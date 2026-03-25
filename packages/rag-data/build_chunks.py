import os
import json
import re
from uuid import uuid4
from typing import List

import argparse

def split_by_delimiter(text: str, delimiter: str, max_chars: int) -> List[str]:
    parts = [p.strip() for p in text.split(delimiter) if p.strip()]
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + len(delimiter) <= max_chars:
            current += part + delimiter
        else:
            if current.strip():
                chunks.append(current.strip())
            current = part + delimiter
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_by_sentences(text: str, max_chars: int) -> List[str]:
    # 간단한 한글 문장 분리 기준 (다., 요., 니다. 등)
    sentences = re.split(r'(?<=[다요니다])[\.\?\!]\s*', text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_text_progressively(text: str, max_chars: int) -> List[str]:
    level1 = split_by_delimiter(text, "\n\n", max_chars)
    final_chunks = []

    for block in level1:
        if len(block) <= max_chars:
            final_chunks.append(block)
        else:
            level2 = split_by_delimiter(block, "\n", max_chars)
            for line in level2:
                if len(line) <= max_chars:
                    final_chunks.append(line)
                else:
                    final_chunks.extend(split_by_sentences(line, max_chars))

    return [chunk.strip() for chunk in final_chunks if chunk.strip()]


def build_chunks(input_path: str, output_path: str, max_chars: int = 500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk_count = 0

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            title = record["title"]
            url = record["url"]
            text = record["text"]
            source_type = record["source_type"]

            chunks = split_text_progressively(text, max_chars=max_chars)

            for chunk_index, chunk_text in enumerate(chunks):
                chunk = {
                    "id": str(uuid4()),
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "title": title,
                    "url": url,
                    "source_type": source_type,
                }
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(f"✅ Chunk 생성 완료: {chunk_count}개 chunk → {output_path}")

def main():
    p = argparse.ArgumentParser(description="Build text chunks from input JSONL")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSONL file path")
    p.add_argument("--output", type=str, default="data/chunks.jsonl",
                   help="Output Chunks JSONL file path")
    p.add_argument("--max_chars", type=int, default=500,
                   help="Maximum characters per chunk")
    args = p.parse_args()
    build_chunks(args.input, args.output, args.max_chars)
    
if __name__ == "__main__":
    main()
