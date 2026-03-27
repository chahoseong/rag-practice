import os
import json
import argparse
import numpy as np
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text

CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db")
COLLECTION_NAME = os.getenv("PG_COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


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
    p = argparse.ArgumentParser(description="Upload pre-computed embeddings to PGVector")
    p.add_argument("--embeddings", type=str, required=True,
                   help="Path to embeddings.npz file (from Colab)")
    p.add_argument("--connection_string", type=str, default=CONNECTION_STRING,
                   help="PostgreSQL connection string")
    p.add_argument("--collection_name", type=str, default=COLLECTION_NAME,
                   help="PGVector collection name")
    p.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL_NAME,
                   help="Embedding model name (must match the model used to generate embeddings)")
    args = p.parse_args()

    # 1) 임베딩 파일 로드
    print(f"Loading embeddings from: {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"].tolist()
    texts = data["texts"].tolist()
    metadata_list = [json.loads(m) for m in data["metadata"].tolist()]
    print(f"Loaded {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # 2) 임베딩 모델 로드 (PGVector.from_embeddings에 필요, 실제 임베딩 생성에는 사용 안 함)
    print(f"Loading embedding model for PGVector compatibility: {args.embedding_model}")
    emb_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cpu"}
    )

    # 3) PGVector에 업로드
    print(f"Uploading to PGVector (Collection: {args.collection_name})...")
    text_embeddings = list(zip(texts, embeddings))

    PGVector.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=emb_model,
        metadatas=metadata_list,
        collection_name=args.collection_name,
        connection=args.connection_string,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    print(f"Uploaded {len(embeddings)} vectors to PGVector")

    # 4) FTS 인덱스 빌드
    build_fts_index(args.connection_string)

    print(f"\nDone! Total vectors: {len(embeddings)}")
    print("Next: restart the backend server to use the new embeddings.")


if __name__ == "__main__":
    main()
