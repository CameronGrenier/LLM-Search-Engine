"""Chunk processed documents for retrieval.
Reads data/processed/docs.jsonl and writes data/processed/chunks.jsonl.
Chunking rules:
- Markdown documents:
  - combine two consecutive heading sections
  - overlap consecutive chunks by one heading
  - max 500 tokens
  - 100-token overlap when an oversized chunk has to be split.
- API JSON documents:
  - each flattened API document as one chunk
  - no heading based chunking
- need to Preserve doc_id and metadata for traceability and citations

Use this command and run from the project root with:
    python3 -m src.chunk
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from config import CHUNKS_PATH, DOCS_PATH

MAX_CHUNK_TOKEN = 500
CHUNK_TOKEN_OVERLAP = 100
HEADINGS_PER_CHUNK = 2
OVERLAP_HEADING = 1


def document_read(path_input: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with open(path_input, "r", encoding="utf-8") as file_input:
        for line_num, line in enumerate(file_input, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path_input}"
                ) from error
            if "doc_id" not in document:
                raise ValueError(
                    f"Document on line {line_num} is missing doc_id"
                )
            if "text" not in document:
                raise ValueError(
                    f"Document {document['doc_id']} is missing text"
                )
            if "metadata" not in document:
                document["metadata"] = {}
            documents.append(document)
    return documents

def token_split(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def token_count(text: str) -> int:
    return len(token_split(text))

def normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def body_content_check(text: str) -> bool:
    without_heading = re.sub(
        r"^#{1,6}\s+.*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    return bool(without_heading.strip())

def extract_heading(text: str) -> list[dict[str, Any]]:
    pattern_heading = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$",
        flags=re.MULTILINE,
    )
    match_heading = list(pattern_heading.finditer(text))
    if not match_heading:
        return [
            {
                "level": 0,
                "heading": "Document",
                "text": normalize(text),
            }
        ]
    sections: list[dict[str, Any]] = []
    intro_text = normalize(text[: match_heading[0].start()])
    if intro_text:
        sections.append(
            {
                "level": 0,
                "heading": "Introduction",
                "text": intro_text,
            }
        )
    for index_match, heading_match in enumerate(match_heading):
        start_section = heading_match.start()
        end_section = (
            match_heading[index_match + 1].start()
            if index_match +1<len(match_heading)
            else len(text)
        )
        section_text = normalize(text[start_section:end_section])
        sections.append(
            {
                "level": len(heading_match.group(1)),
                "heading": heading_match.group(2).strip(),
                "text": section_text,
            }
        )
    return sections


def large_text_split(
    text: str,
    token_max: int = MAX_CHUNK_TOKEN,
    token_overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[str]:
    token_store = token_split(text)
    if len(token_store) <= token_max:
        return [normalize(text)]
    if token_overlap >= token_max:
        raise ValueError(
            "Token overlap must be smaller than the maximum token count"
        )
    chunk_store: list[str] = []
    place_start = 0
    step_chunk = token_max - token_overlap
    while place_start < len(token_store):
        place_end = min(place_start + token_max, len(token_store))
        temp_chunk = " ".join(token_store[place_start:place_end]).strip()
        if temp_chunk:
            chunk_store.append(temp_chunk)
        if len(token_store)==place_end:
            break
        place_start = place_start + step_chunk
    return chunk_store

def record_chunk(
    document: dict[str, Any],
    index_chunk: int,
    text: str,
    section_heading: list[str],
    index_split: int | None = None,
) -> dict[str, Any]:
    id_document = document["doc_id"]
    if index_split is not None:
        id_chunk = (
                    f"{id_document}__chunk_{index_chunk:04d}"
                    f"__part_{index_split:03d}"
                )
    else:
        id_chunk = f"{id_document}__chunk_{index_chunk:04d}"
    metadata_chunk = dict(document.get("metadata", {}))
    metadata_chunk["headings"] = section_heading
    metadata_chunk["chunk_index"] = index_chunk
    if index_split is not None:
        metadata_chunk["split_index"] = index_split
    return {
        "chunk_id": id_chunk,
        "doc_id": id_document,
        "text": normalize(text),
        "metadata": metadata_chunk,
        "token_count": token_count(text),
    }

def prose_chunk(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    section_heading = extract_heading(document["text"])
    if not section_heading:
        return []
    chunk_store: list[dict[str, Any]] = []
    chunk_index = 1
    if len(section_heading)==1:
        temp = section_heading[0]
        if not body_content_check(temp["text"]):
            return []
        parts = large_text_split(temp["text"])
        for split_index, piece in enumerate(parts, start=1):
            chunk_store.append(
                record_chunk(
                    document=document,
                    index_chunk=chunk_index,
                    text=piece,
                    section_heading=[temp["heading"]],
                    index_split=(
                        split_index if len(parts) > 1 else None
                    ),
                )
            )
        return chunk_store
    step_heading = HEADINGS_PER_CHUNK - OVERLAP_HEADING
    for section_start in range(0, len(section_heading) - 1, step_heading):
        section_select = section_heading[
            section_start : section_start + HEADINGS_PER_CHUNK
        ]
        if not section_select:
            continue
        combined_text = "\n\n".join(
            section["text"]
            for section in section_select
            if section["text"]
        )
        headings = [
            section["heading"]
            for section in section_select
        ]
        if not body_content_check(combined_text):
            continue
        parts = large_text_split(combined_text)
        for split_index, piece in enumerate(parts, start=1):
            chunk_store.append(
                record_chunk(
                    document=document,
                    index_chunk=chunk_index,
                    text=piece,
                    section_heading=headings,
                    index_split=(
                        split_index if len(parts) > 1 else None
                    ),
                )
            )
        chunk_index =chunk_index + 1
    return chunk_store

def api_chunk_doc(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    doc_metadata = document.get("metadata", {})
    comp = doc_metadata.get("component")
    doc_title = doc_metadata.get("title")
    chunk_heading = comp or doc_title or "API reference"
    return [
        record_chunk(
            document=document,
            index_chunk=1,
            text=document["text"],
            section_heading=[str(chunk_heading)],
        )
    ]

def document_chunks(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunk_store: list[dict[str, Any]] = []
    for doc in documents:
        doc_txt = normalize(str(doc.get("text", "")))
        if not doc_txt:
            print(f"Skipping empty document: {doc['doc_id']}")
            continue
        doc["text"] = doc_txt
        doc_type = doc.get("metadata", {}).get("doc_type")
        if "api"!=doc_type:
            temp = prose_chunk(doc)
        else:
            temp = api_chunk_doc(doc)
        chunk_store.extend(temp)
    return chunk_store

def chunk_valid(chunks: list[dict[str, Any]]) -> None:
    existing_id: set[str] = set()
    duplicated_id: list[str] = []
    empty_chunk: list[str] = []
    missing_id_doc: list[str] = []
    oversize_chunk_id: list[str] = []
    for temp in chunks:
        id_chunk = temp.get("chunk_id", "")
        if id_chunk in existing_id:
            duplicated_id.append(id_chunk)
        existing_id.add(id_chunk)
        if not temp.get("doc_id"):
            missing_id_doc.append(id_chunk)
        if not temp.get("text", "").strip():
            empty_chunk.append(id_chunk)
        type_doc = temp.get("metadata", {}).get("doc_type")
        if (
            type_doc == "prose"
            and temp.get("token_count", 0) > MAX_CHUNK_TOKEN
        ):
            oversize_chunk_id.append(id_chunk)
    if duplicated_id:
        raise ValueError(
            f"Duplicate chunk IDs found: {duplicated_id[:10]}"
        )
    if empty_chunk:
        raise ValueError(
            f"Empty chunks found: {empty_chunk[:10]}"
        )
    if missing_id_doc:
        raise ValueError(
            f"Chunks missing doc_id: {missing_id_doc[:10]}"
        )
    if oversize_chunk_id:
        raise ValueError(
            "Prose chunks over the 500-token limit: "
            f"{oversize_chunk_id[:10]}"
        )

def chunk_write(
    chunk_store: list[dict[str, Any]],
    path_output: Path,
) -> None:
    path_output.parent.mkdir(parents=True, exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as file:
        for chunk in chunk_store:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

def statistics_show(
    documents: list[dict[str, Any]],
    chunk_store: list[dict[str, Any]],
) -> None:
    count_token = [
        chunk["token_count"]
        for chunk in chunk_store
    ]
    chunk_prose = [
        chunk
        for chunk in chunk_store
        if chunk["metadata"].get("doc_type") == "prose"
    ]
    chunk_api = [
        chunk
        for chunk in chunk_store
        if chunk["metadata"].get("doc_type") == "api"
    ]
    oversize_api_chunks = [
        chunk
        for chunk in chunk_api
        if chunk["token_count"] > MAX_CHUNK_TOKEN
    ]
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunk_store)}")
    print(f"Prose chunks: {len(chunk_prose)}")
    print(f"API chunks: {len(chunk_api)}")
    if count_token:
        token_avg = sum(count_token) / len(count_token)
        print(f"Average tokens per chunk: {token_avg:.2f}")
        print(f"Minimum tokens: {min(count_token)}")
        print(f"Maximum tokens: {max(count_token)}")
    print(
        "API chunks over 500 tokens "
        f"(kept whole by project plan): {len(oversize_api_chunks)}"
    )

def main() -> None:
    docs = document_read(DOCS_PATH)
    chunks = document_chunks(docs)
    chunk_valid(chunks)
    chunk_write(chunks, CHUNKS_PATH)
    statistics_show(docs, chunks)
    print(f"Wrote chunks to {CHUNKS_PATH}")

if __name__ == "__main__":
    main()