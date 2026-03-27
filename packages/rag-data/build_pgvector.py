import os
import json
import torch
import argparse
from tqdm import tqdm
import multiprocessing as mp

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text

# 설정
CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db")
COLLECTION_NAME = os.getenv("PG_COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 4  # GPU 메모리에 따라 조정 가능
MAX_WORKERS = min(4, mp.cpu_count())  # CPU 코어 수에 따라 조정

def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_documents(chunks: list[dict]) -> list[Document]:
    documents = []
    MIN_CHUNK_LENGTH = 20
    for i, chunk in enumerate(chunks):
        # enriched_chunk_text가 있으면 임베딩용으로 사용
        page_content = chunk.get("enriched_chunk_text", chunk.get("chunk_text", "")).strip()
        original_chunk_text = chunk.get("chunk_text", "").strip()

        if len(original_chunk_text) > MIN_CHUNK_LENGTH:
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "id": chunk.get("id", str(i)),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "source_type": chunk.get("source_type", "Unknown"),
                    "original_chunk_text": original_chunk_text,
                }
            ))
    return documents

def create_optimized_embeddings(documents, embedding_model, batch_size=BATCH_SIZE):
    """배치 처리로 GPU 사용률을 최적화한 임베딩 생성"""
    print(f"Creating embeddings with batch size: {batch_size}")

    # 텍스트만 추출
    texts = [doc.page_content for doc in documents]

    # 배치 단위로 처리
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_embeddings

def build_fts_index(connection_string: str):
    """PostgreSQL Full-Text Search 인덱스를 생성합니다."""
    print("Building full-text search index...")
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE langchain_pg_embedding
            ADD COLUMN IF NOT EXISTS fts_vector tsvector
        """))
        conn.execute(text("""
            UPDATE langchain_pg_embedding
            SET fts_vector = to_tsvector('simple', document)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_fts_vector
            ON langchain_pg_embedding USING GIN (fts_vector)
        """))
        conn.commit()
    print("FTS index built successfully.")

def main():
    p = argparse.ArgumentParser(description="Build PGVector index from text chunks")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSONL file path containing text chunks")
    p.add_argument("--connection_string", type=str, default=CONNECTION_STRING,
                   help="PostgreSQL connection string (e.g., postgresql+psycopg://user:password@host:port/dbname)")
    p.add_argument("--collection_name", type=str, default=COLLECTION_NAME,
                   help="Name of the PGVector collection")
    p.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL_NAME,
                   help="HuggingFace model name for embeddings")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Batch size for embedding generation")
    p.add_argument("--device", type=str, default='cpu',
                   help="Use cuda for GPU acceleration / mps for Mac GPU / cpu for CPU")
    args = p.parse_args()

    print(f"Loading chunks from: {args.input}")
    chunks = load_chunks(args.input)
    documents = build_documents(chunks)
    print(f"Loaded {len(documents)} documents")

    print(f"Loading embedding model: {args.embedding_model}")
    # GPU 사용 설정 및 최적화
    device = args.device
    print(f"Using device: {device}")

    # GPU 메모리 최적화 설정
    model_kwargs = {
        'device': device,
        'model_kwargs': {
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        }
    }

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs=model_kwargs
    )

    print("Creating optimized embeddings with batch processing...")

    # 최적화된 방식으로 임베딩 생성
    embeddings = create_optimized_embeddings(documents, embedding_model, args.batch_size)

    # PGVector 인덱스 생성 및 삽입
    print(f"Inserting into PGVector (Collection: {args.collection_name})...")
    text_embeddings = list(zip([doc.page_content for doc in documents], embeddings))
    metadatas = [doc.metadata for doc in documents]

    vector_store = PGVector.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=args.collection_name,
        connection=args.connection_string,
        use_jsonb=True, # rag-engine 스타일: JSONB 메타데이터 사용 권장
        pre_delete_collection=True # 기존 컬렉션 데이터 덮어쓰기 (재빌드용)
    )

    print(f"Inserted into PostgreSQL Database at: {args.connection_string.split('@')[-1]}")
    print(f"Insert complete. Total vectors: {len(documents)}")

if __name__ == "__main__":
    main()
