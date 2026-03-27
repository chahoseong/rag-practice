import os
import re
import json
from typing import List, Optional
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.models import DocumentChunk
from sqlalchemy import create_engine, text


def korean_ngram_tokenize(text: str, n: int = 2) -> List[str]:
    """한국어 텍스트를 문자 n-gram으로 토큰화합니다.

    공백 기반 단어 분리 후, 각 단어에서 n-gram을 추출합니다.
    짧은 단어(n 미만)는 그대로 유지합니다.
    """
    words = re.findall(r'[\w]+', text)
    tokens = []
    for word in words:
        if len(word) <= n:
            tokens.append(word)
        else:
            tokens.extend(word[i:i+n] for i in range(len(word) - n + 1))
    return tokens

class PGVectorAdapter:
    def __init__(self, connection_string: str, collection_name: str, embedding_model_name: str, use_cuda_env: str = "USE_CUDA"):
        self.vector_db = 'pgvector'
        self.embedding_model_name = embedding_model_name
        self._collection_name = collection_name
        self._connection_string = connection_string
        self._use_cuda_env = use_cuda_env
        self._emb = None # 지연 로딩용
        self._store: Optional[PGVector] = None
        self._loaded = False

    def _get_embeddings(self):
        """Lazy load the embedding model."""
        if self._emb is None:
            # Configure embedding model device according to fastapi-standard
            device = 'cuda' if os.environ.get(self._use_cuda_env, 'false').lower() == 'true' else 'cpu'
            print(f"Loading embedding model: {self.embedding_model_name} on {device}...")
            self._emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': device}
            )
        return self._emb

    def load(self) -> None:
        """Initialize the connection to the pgvector database."""
        if self._loaded:
            return

        # Depending on the rag-engine standard, use PGVector wrapper (cosine distance natively supported)
        self._store = PGVector(
            connection=self._connection_string,
            embeddings=self._get_embeddings(),
            collection_name=self._collection_name,
            use_jsonb=True, # Recommended for flexible metadata
            # pgvector's default distance_strategy is cosine
        )
        self._loaded = True

    def _ensure_loaded(self):
        """Guard clause for loaded state."""
        if not self._store:
            raise RuntimeError("PGVector store not loaded. Call load() first.")

    def _build_chunk(self, doc) -> DocumentChunk:
        """LangChain Document를 DocumentChunk로 변환합니다."""
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        return DocumentChunk(
            id=meta.get('id', ''),
            chunk_text=doc.page_content,
            original_chunk_text=meta.get('original_chunk_text', doc.page_content),
            chunk_index=meta.get('chunk_index', 0),
            title=meta.get('title', 'Unknown'),
            url=meta.get('url', ''),
            source_type=meta.get('source_type', 'Unknown'),
        )

    def retrieve(self,
                query: str,
                top_k: int = int(os.getenv("TOP_K", "3")),
                source_type: Optional[str] = None,
                title: Optional[str] = None,
                url: Optional[str] = None
            ) -> List[DocumentChunk]:
        """Retrieve similar documents using pgvector."""
        self._ensure_loaded()

        # Construct filter matching pgvector jsonb metadata format
        filter_dict = {}
        if source_type:
            filter_dict['source_type'] = source_type
        if title:
            filter_dict['title'] = title
        if url:
            filter_dict['url'] = url

        retrieved_docs = self._store.similarity_search(query, k=top_k, filter=filter_dict if filter_dict else None)

        return [self._build_chunk(doc) for doc in retrieved_docs]

    def retrieve_bm25(self,
                      query: str,
                      top_k: int = 10,
                      source_type: Optional[str] = None
                      ) -> List[DocumentChunk]:
        """Python BM25로 키워드 기반 검색. 한국어 n-gram 토큰화 사용."""
        engine = create_engine(self._connection_string)
        with engine.connect() as conn:
            params = {"collection": self._collection_name}
            source_filter = ""
            if source_type:
                source_filter = "AND e.cmetadata->>'source_type' = :source_type"
                params["source_type"] = source_type

            sql = text(f"""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :collection {source_filter}
            """)
            results = conn.execute(sql, params).fetchall()

        if not results:
            return []

        # BM25 인덱스 구축 + 검색
        corpus = [korean_ngram_tokenize(row.document) for row in results]
        bm25 = BM25Okapi(corpus)
        query_tokens = korean_ngram_tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # 상위 top_k 추출
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            row = results[idx]
            meta = row.cmetadata if isinstance(row.cmetadata, dict) else json.loads(row.cmetadata)
            chunks.append(DocumentChunk(
                id=meta.get('id', ''),
                chunk_text=row.document,
                original_chunk_text=meta.get('original_chunk_text', row.document),
                chunk_index=meta.get('chunk_index', 0),
                title=meta.get('title', ''),
                url=meta.get('url', ''),
                source_type=meta.get('source_type', ''),
                score=float(round(scores[idx], 4)),
            ))
        return chunks

    def count(self) -> int:
        """Count the number of embeddings in the current collection."""
        if not self._connection_string or not self._collection_name:
            return 0

        try:
            # Note: direct SQL access for efficient counting
            engine = create_engine(self._connection_string)
            with engine.connect() as conn:
                # Query langchain_pg_embedding table matching the collection UUID
                query = text("""
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :name
                """)
                result = conn.execute(query, {"name": self._collection_name}).scalar()
                return int(result) if result else 0
        except Exception as e:
            print(f"Error counting pgvector embeddings: {e}")
            return 0
