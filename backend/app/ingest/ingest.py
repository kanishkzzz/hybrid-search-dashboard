from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_TEXT_LENGTH = 5000
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest text and markdown files into JSONL.")
    parser.add_argument("--input", required=True, help="Input directory with raw files")
    parser.add_argument("--out", required=True, help="Output directory for docs.jsonl")
    return parser.parse_args()


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_doc(source_path: Path, input_root: Path, created_at: str) -> dict[str, str]:
    raw_text = source_path.read_text(encoding="utf-8")
    cleaned_text = clean_whitespace(raw_text)[:MAX_TEXT_LENGTH]
    source = source_path.relative_to(input_root).as_posix()

    title = cleaned_text[:80] if cleaned_text else source_path.stem
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]

    return {
        "doc_id": digest,
        "title": title,
        "text": cleaned_text,
        "source": source,
        "created_at": created_at,
    }


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
            doc = build_doc(path, input_dir, created_at)
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
