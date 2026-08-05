from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_TEXT_LENGTH = 5000
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest text and markdown files into JSONL.")
    parser.add_argument("--input", required=True, help="Input directory with raw files")
    parser.add_argument("--out", required=True, help="Output directory for docs.jsonl")
    return parser.parse_args()


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def build_docs(source_path: Path, input_root: Path, created_at: str) -> list[dict[str, str]]:
    raw_text = source_path.read_text(encoding="utf-8")
    cleaned_text = clean_whitespace(raw_text)[:MAX_TEXT_LENGTH]
    source = source_path.relative_to(input_root).as_posix()

    parent_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    chunks = chunk_text(cleaned_text)

    docs: list[dict[str, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        doc_id = parent_id if len(chunks) == 1 else f"{parent_id}:{chunk_index}"
        title = chunk[:80] if chunk else source_path.stem
        docs.append(
            {
                "doc_id": doc_id,
                "parent_id": parent_id,
                "chunk_index": str(chunk_index),
                "title": title,
                "text": chunk,
                "source": source,
                "created_at": created_at,
            }
        )

    return docs


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run(input_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "docs.jsonl"
    created_at = datetime.now(timezone.utc).isoformat()

    files = iter_input_files(input_dir)

    with output_file.open("w", encoding="utf-8") as fh:
        for path in files:
            for doc in build_docs(path, input_dir, created_at):
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return output_file


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_file = run(input_dir=input_dir, out_dir=out_dir)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
