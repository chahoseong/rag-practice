import os
import json
from typing import List, Optional
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from app.models import DocumentChunk
from sqlalchemy import create_engine, text

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

    def retrieve_fts(self,
                     query: str,
                     top_k: int = 10,
                     source_type: Optional[str] = None
                     ) -> List[DocumentChunk]:
        """PostgreSQL Full-Text Search로 키워드 기반 검색."""
        engine = create_engine(self._connection_string)
        with engine.connect() as conn:
            # source_type 필터 조건 구성
            source_filter = ""
            params = {
                "query": query,
                "collection": self._collection_name,
                "top_k": top_k,
            }
            if source_type:
                source_filter = "AND e.cmetadata->>'source_type' = :source_type"
                params["source_type"] = source_type

            sql = text(f"""
                SELECT e.document, e.cmetadata,
                       ts_rank_cd(e.fts_vector, plainto_tsquery('simple', :query)) AS rank
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :collection
                  AND e.fts_vector @@ plainto_tsquery('simple', :query)
                  {source_filter}
                ORDER BY rank DESC
                LIMIT :top_k
            """)
            results = conn.execute(sql, params).fetchall()

        chunks = []
        for row in results:
            meta = row.cmetadata if isinstance(row.cmetadata, dict) else json.loads(row.cmetadata)
            chunks.append(DocumentChunk(
                id=meta.get('id', ''),
                chunk_text=row.document,
                original_chunk_text=meta.get('original_chunk_text', row.document),
                chunk_index=meta.get('chunk_index', 0),
                title=meta.get('title', ''),
                url=meta.get('url', ''),
                source_type=meta.get('source_type', ''),
                score=float(row.rank),
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
