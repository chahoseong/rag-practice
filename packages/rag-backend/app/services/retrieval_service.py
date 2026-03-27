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


def make_vector_store_adapter(collection_name: Optional[str] = None) -> VectorStoreAdapter:
    vector_db = os.getenv("VECTOR_DB", "pgvector").lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    if vector_db == "faiss":
        from app.services.faiss_adapter import FaissAdapter
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        return FaissAdapter(index_dir=index_dir, embedding_model_name=embedding_model)
    elif vector_db == "pgvector":
        from app.services.pgvector_adapter import PGVectorAdapter
        connection_string = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db")
        target_collection = collection_name or os.getenv("PG_COLLECTION_NAME", "rag_collection")
        return PGVectorAdapter(
            connection_string=connection_string,
            collection_name=target_collection,
            embedding_model_name=embedding_model
        )
    print(f"Unsupported VECTOR_DB: {vector_db}")
    return VoidRetrievalAdapter()


# ───────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion (RRF)
# ───────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    dense_results: List[DocumentChunk],
    sparse_results: List[DocumentChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[DocumentChunk]:
    """두 검색 결과를 RRF 점수로 병합."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, DocumentChunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.id] = dense_weight / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + sparse_weight / (k + rank + 1)
        if chunk.id not in chunk_map:
            chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    result = []
    for cid in sorted_ids:
        c = chunk_map[cid]
        c.score = round(scores[cid], 6)
        result.append(c)
    return result


# ───────────────────────────────────────────────────────────────
# RetrievalService → List[DocumentChunk] 반환
# ───────────────────────────────────────────────────────────────
class RetrievalService:
    def __init__(self, vector_store: Optional[VectorStoreAdapter] = None):
        self._rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"
        self._hybrid_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
        self._hybrid_dense_weight = float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0"))
        self._hybrid_sparse_weight = float(os.getenv("HYBRID_SPARSE_WEIGHT", "1.0"))

        self.vector_store = vector_store or make_vector_store_adapter()
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
        self.cross_encoder_device = os.getenv("CROSS_ENCODER_DEVICE", "cpu")

        self._initialize()

        # 리랭커 로드
        if self._rerank_enabled:
            print(f"Loading CrossEncoder model: {self.cross_encoder_model}...")
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=self.cross_encoder_device)
            print("CrossEncoder model loaded successfully.")
        else:
            self.cross_encoder = None

    def _initialize(self):
        try:
            print("Loading vector store...")
            self.vector_store.load()
            print(f"Vector store loaded: count={self.vector_store.count()}")
            if self._hybrid_enabled:
                print("Hybrid search enabled (Dense + BM25/FTS)")
        except Exception as e:
            print(f"Error initializing RetrievalService: {e}")
            self.vector_store = VoidRetrievalAdapter()

    @observe()
    async def retrieve(self,
                    query: str,
                    candidate_k: Optional[int] = None,
                    top_k: int = int(os.getenv("TOP_K", 3)),
                    source_type: Optional[str] = None,
                    title: Optional[str] = None,
                    url: Optional[str] = None
                ) -> List[DocumentChunk]:
        """질문과 유사한 문서 청크를 검색하고, DocumentChunk 리스트로 반환."""
        if not self.vector_store:
            return []
        try:
            retrieved_k = candidate_k if (candidate_k and candidate_k > top_k) else top_k

            # 1) Dense 벡터 검색
            loop = asyncio.get_running_loop()
            dense_func = partial(
                self.vector_store.retrieve,
                query=query,
                top_k=retrieved_k,
                source_type=source_type,
                title=title,
                url=url,
            )
            chunks = await loop.run_in_executor(None, dense_func)

            # 2) Hybrid: BM25/FTS 검색 + RRF 병합
            if self._hybrid_enabled and hasattr(self.vector_store, 'retrieve_fts'):
                fts_func = partial(
                    self.vector_store.retrieve_fts,
                    query=query,
                    top_k=retrieved_k,
                    source_type=source_type,
                )
                sparse_chunks = await loop.run_in_executor(None, fts_func)

                chunks = reciprocal_rank_fusion(
                    chunks, sparse_chunks,
                    dense_weight=self._hybrid_dense_weight,
                    sparse_weight=self._hybrid_sparse_weight,
                )
                chunks = chunks[:retrieved_k]

            # 3) CrossEncoder 재정렬
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
                "hybrid_search_enabled": self._hybrid_enabled,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
