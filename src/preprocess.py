"""Preprocessing pipeline: raw corpus files -> cleaned (metadata, text) pairs.

Reads files listed in the manifest (never globs data/corpus/ directly, so
manifest.json itself can never be mistaken for a document), extracts
title/component from frontmatter or API JSON, strips embedded HTML,
INLINES demo source code in place of MUI's {{"demo": ...}} directives,
assigns deterministic doc_ids, and writes docs.jsonl.

Why inline demos instead of stripping them:
  The directives are the only place the prose docs connect a task
  description ("mutually exclusive options") to the code that implements
  it. Dropping them leaves the corpus able to answer "what is a radio
  group" but not "show me one", which is the dominant query intent for
  component-library documentation. Inlining turns each demo into
  retrievable text co-located with its explanatory prose, so a single
  chunk carries both the intent signal and the answer.

Stage ordering is load-bearing:
  1. strip_html      -- must run BEFORE code is inlined, or the tag
                        regex would delete every JSX element.
  2. resolve_directives -- single pass over {{ ... }}: demo directives
                        become fenced code blocks, all others are
                        dropped. Must be one pass, because a second
                        sweep for {{ ... }} would match JSX props like
                        sx={{ mt: 2 }} in the code just inserted.
"""

from __future__ import annotations
import html
from pathlib import Path
import posixpath
import re
import yaml
import json
from bs4 import BeautifulSoup

from config import CORPUS_DIR, MANIFEST_PATH, DOCS_PATH, DEMOS_PATH

# Any {{ ... }} template directive. Applied exactly once, before any code
# is inlined (see module docstring).
DIRECTIVE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)

# Extension -> markdown fence language hint.
FENCE_LANGS = {
    ".js": "jsx",
    ".jsx": "jsx",
    ".ts": "tsx",
    ".tsx": "tsx",
}

# Fallback order when the referenced demo extension is absent from the
# corpus (MUI ships some demos as TS-only or JS-only).
FALLBACK_EXTS = (".js", ".tsx", ".jsx", ".ts")

# Placeholder left in the text where a demo was referenced. Kept on its own
# line so line-oriented chunking can never split one in half. Stripped from
# chunk text and recorded in chunk metadata by the chunking stage.
DEMO_MARKER = "[[DEMO:{demo_id}]]"

# ============================================================
# File reading
# ============================================================


def read_file(file: Path) -> str | dict:
    """Reads a corpus file, parsing JSON files and returning markdown as-is.

    Args:
      file: Path to a .md or .json file.

    Returns:
      The raw markdown text (str) for .md files, or the parsed content
      (dict) for .json files. Empty string for unsupported extensions.
    """
    ext = file.suffix.lower()
    with open(file, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".md":
            return f.read()

    print(f"Unsupported file extension '{ext}' on {file.name}, skipping.")
    return ""


# ============================================================
# Markdown parsing
# ============================================================


def parse_frontmatter(raw_text: str) -> tuple[dict, str]:
    """Splits a markdown file into its YAML frontmatter and body text.

    Frontmatter is the YAML block delimited by '---' at the very start
    of the file, e.g.::

        ---
        title: React Typography component
        components: Typography
        ---

        # Typography
        ...

    Args:
      raw_text: The full, unmodified contents of the markdown file.

    Returns:
      A tuple of (frontmatter, body):
        frontmatter: Parsed key-value pairs from the YAML block. Empty
          dict if no frontmatter block is present.
        body: The markdown content with the frontmatter block removed.
          Unchanged from raw_text if no frontmatter block is present.
    """
    match = re.match(r"^---\s*\n(.*?\n)---\s*\n?", raw_text, re.DOTALL)
    if not match:
        return {}, raw_text

    fm_block = match.group(1)
    body = raw_text[match.end() :]
    frontmatter = yaml.safe_load(fm_block) or {}
    return frontmatter, body


def strip_html(text: str) -> str:
    """Remove embedded HTML while preserving Markdown line structure.

    Markdown headings and paragraph boundaries must remain on separate
    lines so the chunking stage can split documents by headings.

    Must be called BEFORE demo code is inlined: the generic tag regex
    cannot distinguish an HTML wrapper from a JSX element.

    Args:
      text: Markdown body text with directives still intact.

    Returns:
      The text with HTML tags removed and entities decoded.
    """
    cleaned = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================
# Demo resolution
# ============================================================


def build_path_index(manifest: dict) -> dict[str, str]:
    """Inverts the manifest into repo path -> flat corpus filename.

    The corpus is dumped flat, so a demo reference cannot be resolved by
    filename alone (many directories contain a file with the same
    basename). Resolving through the original repo path instead makes
    lookups exact: the reference is joined to the referencing document's
    own directory, which is how MUI's doc site resolves them too.

    Args:
      manifest: Parsed manifest.json, keyed by flat filename.

    Returns:
      A dict mapping each original repo-relative path to its flat filename.
    """
    return {posixpath.normpath(info["url"]): flat for flat, info in manifest.items()}


def _fence_language(path: str) -> str:
    """Picks a markdown fence language hint for a demo source path.

    Args:
      path: Repo path or filename of the demo source file.

    Returns:
      A language tag such as "jsx" or "tsx"; empty string if unknown.
    """
    return FENCE_LANGS.get(posixpath.splitext(path)[1].lower(), "")


def load_demo_source(
    demo_ref: str,
    doc_url: str,
    path_index: dict[str, str],
    corpus_dir: Path,
) -> tuple[str, str] | tuple[None, None]:
    """Resolves a demo reference to its source code.

    Args:
      demo_ref: The directive's demo value, e.g. "RadioButtonsGroup.js".
        May contain relative segments, which are normalised away.
      doc_url: Repo path of the markdown file containing the directive,
        used as the resolution base.
      path_index: Mapping from repo path to flat corpus filename.
      corpus_dir: Directory holding the flat-dumped corpus files.

    Returns:
      A tuple of (resolved_repo_path, source_text), or (None, None) if no
      matching file was extracted.
    """
    base_dir = posixpath.dirname(posixpath.normpath(doc_url))
    primary = posixpath.normpath(posixpath.join(base_dir, demo_ref))

    stem, ext = posixpath.splitext(primary)
    candidates = [primary]
    candidates.extend(stem + alt for alt in FALLBACK_EXTS if alt != ext.lower())

    for candidate in candidates:
        flat = path_index.get(candidate)
        if flat is None:
            continue
        path = corpus_dir / flat
        if path.exists():
            return candidate, path.read_text(encoding="utf-8")

    return None, None


def resolve_directives(
    text: str,
    doc_url: str,
    path_index: dict[str, str],
    corpus_dir: Path,
) -> tuple[str, dict[str, dict], list[str]]:
    """Replaces demo directives with position markers and collects their source.

    MUI's docs use double-curly-brace directives to embed interactive demos
    and components, e.g.::

        {{"demo": "AccordionExpandIcon.js", "bg": true}}
        {{"component": "@mui/internal-core-docs/ComponentLinkHeader"}}

    The referenced source is deliberately NOT inlined into the returned text.
    Demo code is inlined at generation time, not indexing time: every MUI demo
    opens with a near-identical import and export-default block, so hundreds of
    chunks would share a large common component in embedding space, compressing
    the distances retrieval depends on. It also pushes long documents past the
    embedding model's context window, which would force the document-level
    context vector to be computed from a truncated head.

    A marker is left in place of each directive so the code's position within
    the document survives chunking. That lets hydration attach each demo to the
    specific chunk it appeared in, rather than dumping every demo in a document
    onto whichever chunk was retrieved.

    Non-demo directives are doc-site rendering concerns with no retrieval value
    and are removed outright.

    Args:
      text: Markdown body text, HTML already stripped.
      doc_url: Repo path of the markdown file, used to resolve references.
      path_index: Mapping from repo path to flat corpus filename.
      corpus_dir: Directory holding the flat-dumped corpus files.

    Returns:
      A tuple of (resolved_text, demos, missing):
        resolved_text: Text with directives replaced by DEMO markers.
        demos: Mapping of demo_id to a record holding the source code, its
          repo path, and a markdown fence language hint.
        missing: Demo references that resolved to no extracted file.
    """
    demos: dict[str, dict] = {}
    missing: list[str] = []

    def _replace(match: re.Match) -> str:
        payload = match.group(0)[1:-1]  # drop one outer brace pair -> valid JSON
        try:
            directive = json.loads(payload)
        except json.JSONDecodeError:
            return ""

        if not isinstance(directive, dict):
            return ""

        demo_ref = directive.get("demo")
        if not demo_ref:
            return ""  # component/other directive: no retrieval value

        resolved, source = load_demo_source(demo_ref, doc_url, path_index, corpus_dir)
        if source is None:
            missing.append(demo_ref)
            return ""

        demo_id = path_index[resolved]  # flat filename, unique by construction
        demos[demo_id] = {
            "demo_id": demo_id,
            "url": resolved,
            "lang": _fence_language(resolved),
            "code": source.strip(),
        }
        return "\n\n" + DEMO_MARKER.format(demo_id=demo_id) + "\n\n"

    resolved_text = DIRECTIVE_RE.sub(_replace, text)
    resolved_text = re.sub(r"\n{3,}", "\n\n", resolved_text)
    return resolved_text.strip(), demos, missing


# ============================================================
# API JSON parsing
# ============================================================


def parse_api_json(data: dict) -> tuple[str, str]:
    """Extracts title and component name from an API reference JSON doc.

    API docs (e.g. button.json) describe exactly one component, so the
    top-level 'name' field serves as both the document title and the
    component tag.

    Args:
      data: Parsed JSON content of an API reference file.

    Returns:
      A tuple of (title, component), both equal to the component name.

    Raises:
      KeyError: If the JSON has no 'name' field, which would indicate
        an unexpected schema and should not be silently swallowed.
    """
    name = data["name"]
    return name, name


def _clean_prop_type(raw: str) -> str:
    """Decodes HTML entities and tags found in API JSON type descriptions.

    MUI's generated API JSON writes union types like::

        "'inherit'<br>&#124;&nbsp;'primary'<br>&#124;&nbsp;'secondary'"

    which needs to become readable prose rather than raw markup.

    Args:
      raw: The raw 'type.description' or 'type.name' string from the JSON.

    Returns:
      A cleaned string with entities decoded and <br> tags replaced by
      a plain separator.
    """
    text = html.unescape(raw)  # &#124; -> "|", &nbsp; -> " "
    text = text.replace("<br>", " ")  # line-break tags -> space
    text = re.sub(r"\s+", " ", text)  # collapse repeated whitespace
    return text.strip()


def flatten_api_json(data: dict) -> str:
    """Flattens an MUI API reference JSON doc into readable prose text.

    Turns the component's prop table into sentences describing each
    prop's type and default value, so the JSON's factual content
    (exactly what factoid questions need) becomes retrievable text
    rather than raw JSON structure.

    Args:
      data: Parsed JSON content of an API reference file (e.g. the
        dict loaded from button.json).

    Returns:
      Plain text describing the component and each of its props.
    """
    name = data.get("name", "Unknown component")
    lines = [f"{name} component API reference."]

    props: dict = data.get("props", {})
    for prop_name, prop_info in props.items():
        type_info = prop_info.get("type", {})
        type_desc = (
            type_info.get("description") or type_info.get("name") or "unspecified"
        )
        type_desc = _clean_prop_type(type_desc)

        default = prop_info.get("default")
        description = prop_info.get("description", "")
        description = _clean_prop_type(description) if description else ""

        sentence = f"Prop `{prop_name}`: type {type_desc}."
        if default is not None:
            sentence += f" Default: {default}."
        if description:
            sentence += f" {description}"

        lines.append(sentence)

    return "\n".join(lines)


# ============================================================
# Per-file metadata + body extraction
# ============================================================


def parse_metadata(file: Path) -> tuple[dict, str | dict] | tuple[None, None]:
    """Extracts (title, component) metadata and raw body content from a file.

    Dispatches on file extension: markdown files get frontmatter parsed
    (falling back to a title derived from the filename when frontmatter
    is absent); JSON files get their component name read from the
    top-level 'name' field.

    Args:
      file: Path to a .md or .json corpus file.

    Returns:
      A tuple of (metadata, body):
        metadata: dict with 'title' and 'component' keys.
        body: markdown body text (str) for .md files, or the raw
          parsed dict (dict) for .json files, to be flattened later.
      Returns (None, None) for unsupported extensions.
    """
    raw_text = read_file(file)
    ext = file.suffix.lower()

    if ext == ".md":
        if not isinstance(raw_text, str):
            raise TypeError(f"Expected markdown text for {file}, got {type(raw_text)}")
        frontmatter, body = parse_frontmatter(raw_text)
        if frontmatter:
            title = frontmatter.get("title")
            component = frontmatter.get("components")
        else:
            # No frontmatter (e.g. about-the-lab.md): derive title from
            # filename, no component name is recoverable.
            clean_name = re.sub(r"[-_]+", " ", file.stem)
            clean_name = " ".join(clean_name.split())
            title = clean_name.title()
            component = None
        return {"title": title, "component": component}, body

    elif ext == ".json":
        if not isinstance(raw_text, dict):
            raise TypeError(
                f"Expected parsed JSON dict for {file}, got {type(raw_text)}"
            )
        title, component = parse_api_json(raw_text)
        return {"title": title, "component": component}, raw_text

    print(f"Unsupported file extension '{ext}' on {file.name}, skipping.")
    return None, None


# ============================================================
# Doc ID assignment + output
# ============================================================


def assign_doc_id(filename: str, doc_type: str) -> str:
    """Builds a deterministic, human-readable doc_id from a corpus filename.

    IDs are derived from the manifest filename (not randomly generated)
    so they stay stable across reruns, which matters for reproducibility
    and for citation traceability.

    Args:
      filename: The flat filename as it appears in data/corpus/
        (and as keyed in the manifest), e.g. "buttons.md" or "button.json".
      doc_type: Either "prose" or "api", used as an ID prefix so doc_ids
        are self-describing at a glance.

    Returns:
      A doc_id string, e.g. "prose__buttons" or "api__button".
    """
    stem = Path(filename).stem
    return f"{doc_type}__{stem}"


def write_docs_jsonl(docs: list[dict], out_path: Path) -> None:
    """Writes one JSON object per line to out_path, creating parents if needed.

    Args:
      docs: List of {doc_id, text, metadata} dicts, one per document.
      out_path: Destination .jsonl file path (e.g. config.DOCS_PATH).

    Returns:
      None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_demos_jsonl(demos: dict[str, dict], out_path: Path) -> None:
    """Writes the demo source store, one JSON object per line.

    Stored separately from docs.jsonl so the code lives in exactly one place.
    Chunks reference demos by id, so overlapping chunks that share a demo do
    not each carry a copy of it.

    Args:
      demos: Mapping of demo_id to its record.
      out_path: Destination .jsonl path (config.DEMOS_PATH).

    Returns:
      None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for demo_id in sorted(demos):
            f.write(json.dumps(demos[demo_id], ensure_ascii=False) + "\n")


# ============================================================
# Main pipeline
# ============================================================


def preprocess(corpus_dir: Path) -> None:
    """Runs the full preprocessing pipeline over every manifested document.

    Iterates the manifest (not the directory) so files that aren't
    listed there are never mistaken for corpus documents. Manifest
    entries of doc_type "demo" are skipped as documents: they are
    ingredients, consumed by the prose docs that reference them, and
    indexing them separately would duplicate content and split the
    intent signal away from the code.

    Args:
      corpus_dir: Directory containing the flat-dumped corpus files.

    Returns:
      None. Writes data/processed/docs.jsonl as a side effect.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    path_index = build_path_index(manifest)

    docs = []
    all_demos: dict[str, dict] = {}
    skipped = []
    unresolved: dict[str, list[str]] = {}

    for filename, info in manifest.items():
        if info["doc_type"] == "demo":
            continue

        filepath = corpus_dir / filename
        content_meta, body = parse_metadata(filepath)
        if content_meta is None:
            skipped.append(filename)
            continue

        metadata = {**content_meta, "url": info["url"], "doc_type": info["doc_type"]}

        if info["doc_type"] == "prose":
            text = strip_html(body if isinstance(body, str) else "")
            text, demos, missing = resolve_directives(
                text, info["url"], path_index, corpus_dir
            )
            all_demos.update(demos)
            if missing:
                unresolved[filename] = missing
        else:
            text = flatten_api_json(body if isinstance(body, dict) else {})

        doc_id = assign_doc_id(filename, info["doc_type"])
        docs.append({"doc_id": doc_id, "text": text, "metadata": metadata})

    write_docs_jsonl(docs, DOCS_PATH)
    write_demos_jsonl(all_demos, DEMOS_PATH)
    print(f"Wrote {len(all_demos)} demos to {DEMOS_PATH}")

    print(f"Wrote {len(docs)} docs to {DOCS_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped}")
    if unresolved:
        total = sum(len(v) for v in unresolved.values())
        print(f"Unresolved demo references: {total} across {len(unresolved)} docs")
        for name, refs in list(unresolved.items())[:10]:
            print(f"  {name}: {refs}")


if __name__ == "__main__":
    preprocess(CORPUS_DIR)
