"""
Fetch MUI documentation for the RAG corpus.

Pins release v9.2.0 for reproducibility. Downloads the repo tarball once
(via codeload, avoiding the GitHub API rate limit) and dumps the selected
files FLAT into data/corpus (no subfolders).

Two subtrees are collected (see config.KEEP_SUBPATHS):
  - docs/data/material/          markdown prose/usage docs
  - docs/pages/material-ui/api/  API reference JSON (prop tables, defaults)

A manifest mapping each flat filename to its original repo path and doc_type
is written to config.MANIFEST_PATH (outside data/corpus/, so it is never
mistaken for a corpus document). Preprocessing should iterate the manifest
keys rather than globbing the corpus directory.

Run this file using `python -m src.fetch_corpus` from the project root.
"""
from __future__ import annotations
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from config import (
    REPO,
    TAG,
    KEEP_SUBPATHS,
    CORPUS_DIR,
    PROCESSED_DIR,
    MANIFEST_PATH,
)


def _matched_subpath(member_name: str) -> tuple[str, str] | None:
    """Return (marker, expected_ext) if this member falls under a kept subtree."""
    for subpath, ext in KEEP_SUBPATHS.items():
        marker = f"/{subpath}"
        if marker in member_name and member_name.endswith(ext):
            return marker, ext
    return None


def _doc_type_for(subpath: str) -> str:
    """Map a source subtree to a doc_type tag used downstream."""
    # API reference JSON vs prose markdown. Keyed on the subpath so it stays
    # correct even if extensions change.
    return "api" if "pages/material-ui/api" in subpath else "prose"


def _download(url: str) -> bytes:
    """Download with a live progress line so the terminal never looks frozen."""
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


def main() -> None:
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/tags/{TAG}"
    raw = _download(url)

    # Both the corpus dir and the processed dir (manifest target) must exist.
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, int] = {}
    counts: dict[str, int] = {ext: 0 for ext in set(KEEP_SUBPATHS.values())}
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

            matched = _matched_subpath(member.name)
            if matched is None:
                continue
            marker, ext = matched
            subpath = marker.strip("/")  # e.g. docs/pages/material-ui/api

            rel = member.name.split(marker, 1)[1]  # components/buttons/buttons.md
            # Original path inside the repo, minus the top-level version dir.
            url = f"{subpath}/{rel}"
            parts = Path(rel).parts
            base = Path(rel).name  # buttons.md / button.json

            # Flat filename; disambiguate collisions with the parent folder name.
            flat = base
            if flat in used_names:
                parent = parts[-2] if len(parts) >= 2 else "root"
                flat = f"{parent}__{base}"
                while flat in used_names:
                    used_names[flat] = used_names.get(flat, 0) + 1
                    flat = f"{parent}_{used_names[flat]}__{base}"
            used_names[flat] = 1

            f = tar.extractfile(member)
            if f is None:
                continue
            (CORPUS_DIR / flat).write_bytes(f.read())
            counts[ext] += 1

            manifest[flat] = {
                "url": url,
                "doc_type": _doc_type_for(subpath),
            }

    sys.stdout.write("\r" + " " * 50 + "\r")  # clear the scan line

    # Write the manifest (sorted keys for stable, diff-friendly output).
    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        json.dump(dict(sorted(manifest.items())), mf, indent=2, ensure_ascii=False)

    total = sum(counts.values())
    summary = ", ".join(f"{n} {ext}" for ext, n in counts.items())
    print(f"Done. Files extracted: {total} ({summary}) -> {CORPUS_DIR}/")
    print(f"Manifest: {len(manifest)} entries -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
