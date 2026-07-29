"""Preprocessing pipeline: raw corpus files -> cleaned (metadata, text) pairs.

Reads files listed in the manifest (never globs data/corpus/ directly, so
manifest.json itself can never be mistaken for a document), extracts
title/component from frontmatter or API JSON, strips MUI template
directives and embedded HTML from markdown (or flattens API JSON into
prose), assigns deterministic doc_ids, and writes docs.jsonl.
"""
from __future__ import annotations
import html
from pathlib import Path
import re
import yaml
import json
from bs4 import BeautifulSoup

from config import CORPUS_DIR, MANIFEST_PATH, DOCS_PATH


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
    body = raw_text[match.end():]
    frontmatter = yaml.safe_load(fm_block) or {}
    return frontmatter, body


def strip_template_directives(text: str) -> str:
    """Removes MUI's doc-site template directives from markdown body text.

    MUI's docs use double-curly-brace directives to embed interactive
    demos and components, e.g.::

        {{"demo": "AccordionExpandIcon.js", "bg": true}}
        {{"component": "@mui/internal-core-docs/ComponentLinkHeader"}}

    These have no retrieval value (they reference JS demo files, not
    documentation text) and are stripped entirely.

    Args:
      text: Markdown body text (frontmatter already removed).

    Returns:
      The text with all {{ ... }} directive lines removed, and resulting
      multi-blank-line gaps collapsed.
    """
    without_directives = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", without_directives)
    return cleaned.strip()


def strip_html(text: str) -> str:
    """Remove embedded HTML while preserving Markdown line structure.

    Markdown headings and paragraph boundaries must remain on separate
    lines so the chunking stage can split documents by headings.
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
    text = html.unescape(raw)          # &#124; -> "|", &nbsp; -> " "
    text = text.replace("<br>", " ")   # line-break tags -> space
    text = re.sub(r"\s+", " ", text)   # collapse repeated whitespace
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
        type_desc = type_info.get("description") or type_info.get("name") or "unspecified"
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
            raise TypeError(f"Expected parsed JSON dict for {file}, got {type(raw_text)}")
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


# ============================================================
# Main pipeline
# ============================================================


def preprocess(corpus_dir: Path) -> None:
    """Runs the full preprocessing pipeline over every manifested file.

    Iterates the manifest (not the directory) so files that aren't
    listed there — including manifest.json itself — are never
    mistaken for corpus documents. Cleans markdown body text or
    flattens API JSON into prose, assigns doc_ids, and writes the
    unified output to docs.jsonl.

    Args:
      corpus_dir: Directory containing the flat-dumped corpus files.

    Returns:
      None. Writes data/processed/docs.jsonl as a side effect.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = []
    skipped = []

    for filename, info in manifest.items():
        filepath = corpus_dir / filename
        content_meta, body = parse_metadata(filepath)
        if content_meta is None:
            skipped.append(filename)
            continue

        metadata = {**content_meta, "url": info["url"], "doc_type": info["doc_type"]}

        if info["doc_type"] == "prose":
            text = strip_html(strip_template_directives(body if isinstance(body, str) else ""))
        else:
            text = flatten_api_json(body if isinstance(body, dict) else {})

        doc_id = assign_doc_id(filename, info["doc_type"])
        docs.append({"doc_id": doc_id, "text": text, "metadata": metadata})

    write_docs_jsonl(docs, DOCS_PATH)

    print(f"Wrote {len(docs)} docs to {DOCS_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} files: {skipped}")


if __name__ == "__main__":
    preprocess(CORPUS_DIR)