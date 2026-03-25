import os
import asyncio
from typing import List, Protocol, Optional
from functools import partial
from app.models import DocumentChunk
from sentence_transformers import CrossEncoder
from .void_retrieval_adapter import VoidRetrievalAdapter
from langfuse import observe

from dotenv import load_dotenv
load_dotenv()


# ───────────────────────────────────────────────────────────────
# Vector store adapters (교체 가능)
# ───────────────────────────────────────────────────────────────
class VectorStoreAdapter(Protocol):
    """All adapters must return (Document-like, similarity[0..1]) pairs."""
    def load(self) -> None: ...
    def retrieve(self,
                query: str,
                top_k: int = int(os.getenv("TOP_K", 3)),  # Default to 3 if not set
                source_type: Optional[str] = None,  # Optional filter by source type
                title: Optional[str] = None,  # Optional filter by title keyword
                url: Optional[str] = None  # Optional filter by URL keyword
            ) -> List[DocumentChunk]: ...
    def count(self) -> int: ...


def make_vector_store_adapter() -> VectorStoreAdapter:
    vector_db = os.getenv("VECTOR_DB", "pgvector").lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    if vector_db == "faiss":
        from app.services.faiss_adapter import FaissAdapter
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        return FaissAdapter(index_dir=index_dir, embedding_model_name=embedding_model)
    elif vector_db == "pgvector":
        from app.services.pgvector_adapter import PGVectorAdapter
        connection_string = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db")
        collection_name = os.getenv("PG_COLLECTION_NAME", "rag_collection")
        return PGVectorAdapter(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_model_name=embedding_model
        )
    print(f"Unsupported VECTOR_DB: {vector_db}")
    return VoidRetrievalAdapter()

# ───────────────────────────────────────────────────────────────
# RetrievalService → List[DocumentChunk] 반환
# ───────────────────────────────────────────────────────────────
class RetrievalService:
    def __init__(self, vector_store: Optional[VectorStoreAdapter] = None):
        self._rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"
        self.vector_store = vector_store or make_vector_store_adapter()
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
        self.cross_encoder_device = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
        self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=self.cross_encoder_device) if self._rerank_enabled else None
        self._initialize()

    def _initialize(self):
        try:
            print("🔌 Loading vector store adapter...")
            self.vector_store.load()
            print(f"✅ Vector store loaded. count={self.vector_store.count()}")
        except Exception as e:
            print(f"❌ Error initializing RetrievalService: {e}")
            self.vector_store = VoidRetrievalAdapter()

    @observe()
    async def retrieve(self,
                    query: str,
                    candidate_k: Optional[int] = None,  # Optional rerank candidates
                    top_k: int = int(os.getenv("TOP_K", 3)),  # Default to 3 if not set
                    source_type: Optional[str] = None,  # Optional filter by source type
                    title: Optional[str] = None,  # Optional filter by title keyword
                    url: Optional[str] = None  # Optional filter by URL keyword
                ) -> List[DocumentChunk]:
        """질문과 유사한 문서 청크를 검색하고, DocumentChunk 리스트로 반환."""
        if not self.vector_store:
            return []
        try:
            retrieved_k = candidate_k if (candidate_k and candidate_k > top_k) else top_k

            # 1) 벡터 DB에서 후보 검색 (★ 키워드 인자)
            chunks = await self._retrieve_vector_db(
                query=query,
                top_k=retrieved_k,
                source_type=source_type,
                title=title,
                url=url,
            )

            # 2) 후보가 더 많으면 Reranking(cross-encoder 재정렬)
            if self._rerank_enabled and self.cross_encoder and (retrieved_k > top_k) and (len(chunks) > top_k):
                pairs = [(query, chunk.chunk_text) for chunk in chunks]
                scores = self.cross_encoder.predict(pairs)
                for chunk, score in zip(chunks, scores):
                    chunk.score = float(round(float(score), 4))
                chunks.sort(key=lambda x: x.score, reverse=True)
                chunks = chunks[:top_k]
                
            if not chunks:
                print("No similar chunks found.")
                return []
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"문서 검색 중 오류 발생: {str(e)}")

    async def _retrieve_vector_db(self,
                    query: str,
                    top_k: int = int(os.getenv("TOP_K", 3)),  # Default to 3 if not set
                    source_type: Optional[str] = None,  # Optional filter by source type
                    title: Optional[str] = None,  # Optional filter by title keyword
                    url: Optional[str] = None  # Optional filter by URL keyword
                ) -> List[DocumentChunk]:
        """질문과 유사한 문서 청크를 검색하고, DocumentChunk 리스트로 반환."""
        if not self.vector_store:
            return []
        try:
            loop = asyncio.get_running_loop()
            func = partial(
                self.vector_store.retrieve,
                query=query,
                top_k=top_k,
                source_type=source_type,
                title=title,
                url=url,
            )
            chunks = await loop.run_in_executor(None, func)
            if not chunks:
                print("No similar chunks found.")
                return []
            return chunks
        except Exception as e:
            raise RuntimeError(f"Vector DB 문서 검색 중 오류 발생: {str(e)}")
        
    def get_index_info(self) -> dict:
        if not self.vector_store._loaded:
            return {"status": "not_loaded", "count": 0}
        try:
            return {
                "status": "loaded",
                "count": self.vector_store.count(),
                "backend": os.getenv("VECTOR_DB", "none"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
                "rerank_enabled": self._rerank_enabled,
                "cross_encoder_model": self.cross_encoder_model if self._rerank_enabled else None,
                "cross_encoder_device": self.cross_encoder_device if self._rerank_enabled else None,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
