import os
import json
import re
from uuid import uuid4
from typing import List, Tuple, Optional

import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_sections(text: str) -> List[Tuple[int, str]]:
    """텍스트에서 섹션 헤더 위치를 감지합니다.

    감지 방식:
    1. == 헤더 == 또는 === 헤더 === 형식 (위키 스타일)
    2. 단락 첫 줄이 짧고(< 30자) 내용 설명인 경우 (위키 API가 헤더를 제거한 경우)

    Returns: [(char_offset, section_name), ...]
    """
    sections: List[Tuple[int, str]] = []
    offset = 0

    for paragraph in text.split("\n\n"):
        first_line = paragraph.strip().split("\n")[0].strip()

        # 위키 스타일 헤더: == 헤더 == 또는 === 헤더 ===
        wiki_match = re.match(r'^(={2,})\s*(.+?)\s*\1$', first_line)
        if wiki_match:
            sections.append((offset, wiki_match.group(2).strip()))
        # 짧은 단락 첫 줄을 섹션 제목으로 간주
        elif (len(first_line) > 0
              and len(first_line) < 30
              and not first_line.endswith(".")
              and not first_line.endswith("다")
              and len(paragraph.strip()) > len(first_line)):
            sections.append((offset, first_line))

        offset += len(paragraph) + 2  # +2 for "\n\n"

    return sections


def find_section_for_chunk(chunk_start: int, sections: List[Tuple[int, str]]) -> Optional[str]:
    """청크가 속한 섹션 이름을 반환합니다."""
    current_section = None
    for sec_offset, sec_name in sections:
        if sec_offset <= chunk_start:
            current_section = sec_name
        else:
            break
    return current_section


def build_enriched_text(title: str, section: Optional[str], chunk_text: str) -> str:
    """컨텍스트가 주입된 enriched 텍스트를 생성합니다."""
    parts = [f"[문서: {title}]"]
    if section:
        parts.append(f"[섹션: {section}]")
    parts.append(chunk_text)
    return "\n".join(parts)


def build_chunks(input_path: str, output_path: str, chunk_size: int = 500,
                 chunk_overlap: int = 50, context_injection: bool = True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_count = 0

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            title = record["title"]
            url = record["url"]
            text = record["text"]
            source_type = record["source_type"]

            # 섹션 감지
            sections = detect_sections(text) if context_injection else []

            # 청크 분할 (오프셋 추적을 위해 create_documents 대신 수동 매칭)
            chunks = splitter.split_text(text)

            # 각 청크의 시작 위치를 추적
            search_start = 0
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_start = text.find(chunk_text[:50], search_start)
                if chunk_start == -1:
                    chunk_start = search_start

                # enriched 텍스트 생성
                if context_injection:
                    section = find_section_for_chunk(chunk_start, sections)
                    enriched = build_enriched_text(title, section, chunk_text)
                else:
                    enriched = chunk_text

                chunk_data = {
                    "id": str(uuid4()),
                    "chunk_text": chunk_text,
                    "enriched_chunk_text": enriched,
                    "chunk_index": chunk_index,
                    "title": title,
                    "url": url,
                    "source_type": source_type,
                }
                f_out.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                chunk_count += 1

                # 다음 검색 시작 위치 업데이트
                if chunk_start >= search_start:
                    search_start = chunk_start + 1

    print(f"Chunk 생성 완료: {chunk_count}개 chunk -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Build text chunks from input JSONL")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSONL file path")
    p.add_argument("--output", type=str, default="data/chunks.jsonl",
                   help="Output Chunks JSONL file path")
    p.add_argument("--chunk_size", type=int, default=500,
                   help="Maximum characters per chunk")
    p.add_argument("--chunk_overlap", type=int, default=50,
                   help="Overlap characters between chunks")
    p.add_argument("--no_context_injection", action="store_true",
                   help="Disable context injection (title/section prepending)")
    args = p.parse_args()
    build_chunks(args.input, args.output, args.chunk_size, args.chunk_overlap,
                 context_injection=not args.no_context_injection)


if __name__ == "__main__":
    main()
