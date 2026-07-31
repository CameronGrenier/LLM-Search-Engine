"""
Fetch MUI documentation for the RAG corpus.

Pins release v9.2.0 for reproducibility. Downloads the repo tarball once
(via codeload, avoiding the GitHub API rate limit) and dumps the selected
files FLAT into data/corpus (no subfolders).

Two subtrees are collected (see config.KEEP_SUBPATHS):
  - docs/data/material/          markdown prose/usage docs
  - docs/pages/material-ui/api/  API reference JSON (prop tables, defaults)

Within those subtrees, demo source files (config.DEMO_EXTS) are ALSO
extracted. They are not documents themselves; they exist so preprocessing
can resolve markdown directives like {{"demo": "RadioButtonsGroup.js"}}
into the actual source code, which is high-value retrievable content for
a "how do I do X in MUI" corpus.

Every extracted file gets a manifest entry keyed by its flat filename:

    "RadioButtonsGroup.js": {
      "url": "docs/data/material/components/radio-buttons/RadioButtonsGroup.js",
      "doc_type": "demo"
    }

The 'url' is the original repo-relative path. Preprocessing inverts the
manifest (url -> flat filename) to resolve a demo reference relative to
the referencing markdown file's directory, which is collision-proof even
though the on-disk layout is flat.

Run this file using `python -m src.fetch_corpus` from the project root.
"""

from __future__ import annotations
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
import posixpath

from config import (
    REPO,
    TAG,
    KEEP_SUBPATHS,
    DEMO_EXTS,
    CORPUS_DIR,
    PROCESSED_DIR,
    MANIFEST_PATH,
)


def _doc_type_for(subpath: str) -> str:
    """Map a source subtree to a doc_type tag used downstream.

    Args:
      subpath: A key from KEEP_SUBPATHS, e.g. "docs/pages/material-ui/api".

    Returns:
      "api" for API reference JSON, "prose" for markdown documentation.
    """
    # Keyed on the subpath so it stays correct even if extensions change.
    return "api" if "pages/material-ui/api" in subpath else "prose"


def _classify(member_name: str) -> tuple[str, str, str] | None:
    """Decide whether a tar member should be extracted, and as what.

    A member is kept if it lives under one of the KEEP_SUBPATHS subtrees
    and either (a) matches that subtree's document extension, or (b) has a
    demo-source extension. Case (b) is the new behaviour: those files are
    not indexed as documents, but their contents are inlined into the
    markdown that references them.

    Args:
      member_name: Full path of the archive member, including the
        top-level version directory (e.g. "material-ui-9.2.0/docs/...").

    Returns:
      A tuple of (marker, subpath, doc_type), where marker is the subpath
      with a leading slash (used to split off the repo-relative remainder),
      and doc_type is one of "prose", "api", or "demo". None if the member
      falls outside every kept subtree or has an uninteresting extension.
    """
    suffix = Path(member_name).suffix.lower()

    for subpath, doc_ext in KEEP_SUBPATHS.items():
        marker = f"/{subpath}"
        if marker not in member_name:
            continue
        if suffix == doc_ext.lower():
            return marker, subpath, _doc_type_for(subpath)
        if suffix in DEMO_EXTS:
            return marker, subpath, "demo"

    return None


def _download(url: str) -> bytes:
    """Download with a live progress line so the terminal never looks frozen.

    Args:
      url: The codeload tarball URL for the pinned tag.

    Returns:
      The raw gzipped tarball bytes.
    """
    print(f"Downloading {REPO}@{TAG} ...")
    print(f"  {url}")

    with urllib.request.urlopen(url) as resp:
        total = resp.getheader("Content-Length")
        total = int(total) if total else None

        chunk_size = 1 << 20  # 1 MB
        buf = io.BytesIO()
        downloaded = 0

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)

            mb = downloaded / (1024 * 1024)
            if total:
                pct = downloaded / total * 100
                total_mb = total / (1024 * 1024)
                sys.stdout.write(f"\r  {mb:7.1f} / {total_mb:6.1f} MB  ({pct:5.1f}%)")
            else:
                sys.stdout.write(f"\r  {mb:7.1f} MB downloaded")
            sys.stdout.flush()

    sys.stdout.write("\n")
    return buf.getvalue()


def _flat_name(rel: str, used_names: dict[str, int]) -> str:
    """Produce a unique flat filename for a repo-relative path.

    The corpus is dumped flat, but MUI reuses basenames heavily across
    component directories (many folders contain an index.js or a
    BasicButtons.js). Collisions are disambiguated by prefixing the parent
    directory, then by a counter if that still collides.

    Args:
      rel: Path relative to the matched subtree,
        e.g. "components/radio-buttons/RadioButtonsGroup.js".
      used_names: Mutable registry of already-issued flat names, mapping
        each name to the number of times it has been claimed.

    Returns:
      A filename unique within the corpus directory.
    """
    parts = Path(rel).parts
    base = Path(rel).name

    flat = base
    if flat in used_names:
        parent = parts[-2] if len(parts) >= 2 else "root"
        flat = f"{parent}__{base}"
        n = 1
        while flat in used_names:
            n += 1
            flat = f"{parent}_{n}__{base}"

    used_names[flat] = 1
    return flat


def _get_tarball(url: str) -> bytes:
    """Returns the tarball bytes, downloading only on a cache miss.

    The archive is immutable for a pinned tag, so re-downloading on every
    run buys nothing and makes iteration on the extraction logic slow.

    Args:
      url: The codeload tarball URL for the pinned tag.

    Returns:
      The raw gzipped tarball bytes.
    """
    if TARBALL_PATH.exists():
        print(f"Using cached tarball: {TARBALL_PATH}")
        return TARBALL_PATH.read_bytes()

    raw = _download(url)
    TARBALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARBALL_PATH.write_bytes(raw)
    return raw


def fetch_corpus() -> None:
    """Download the pinned MUI tarball and flat-dump docs, API JSON, and demos.

    Returns:
      None. Writes files into CORPUS_DIR and a manifest to MANIFEST_PATH.
    """
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/tags/{TAG}"
    raw = _get_tarball(url)

    # Both the corpus dir and the processed dir (manifest target) must exist.
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, int] = {}
    counts: dict[str, int] = {"prose": 0, "api": 0, "demo": 0}
    manifest: dict[str, dict[str, str]] = {}

    print("Scanning archive and extracting matched files ...")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        scanned = 0
        for member in tar:  # streaming iteration, no full getmembers() upfront
            scanned += 1
            if scanned % 500 == 0:
                kept = sum(counts.values())
                sys.stdout.write(f"\r  scanned {scanned} entries, kept {kept}")
                sys.stdout.flush()

            if not member.isfile():
                continue

            classified = _classify(member.name)
            if classified is None:
                continue
            marker, subpath, doc_type = classified

            # Repo-relative remainder, e.g. components/buttons/buttons.md
            rel = member.name.split(marker, 1)[1]
            # normpath so a trailing slash on a KEEP_SUBPATHS key cannot leak a
            # double separator into the manifest, which would silently break
            # every path-based lookup downstream.
            repo_path = posixpath.normpath(f"{subpath}/{rel}")

            flat = _flat_name(rel, used_names)

            f = tar.extractfile(member)
            if f is None:
                continue
            (CORPUS_DIR / flat).write_bytes(f.read())
            counts[doc_type] += 1

            manifest[flat] = {
                "url": repo_path,
                "doc_type": doc_type,
            }

    sys.stdout.write("\r" + " " * 50 + "\r")  # clear the scan line

    # Write the manifest (sorted keys for stable, diff-friendly output).
    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        json.dump(dict(sorted(manifest.items())), mf, indent=2, ensure_ascii=False)

    total = sum(counts.values())
    summary = ", ".join(f"{n} {kind}" for kind, n in counts.items())
    print(f"Done. Files extracted: {total} ({summary}) -> {CORPUS_DIR}/")
    print(f"Manifest: {len(manifest)} entries -> {MANIFEST_PATH}")


if __name__ == "__main__":
    fetch_corpus()
