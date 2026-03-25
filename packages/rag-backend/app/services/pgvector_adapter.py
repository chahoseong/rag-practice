import os
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
        
        # Configure embedding model device according to fastapi-standard (which relies heavily on efficient async APIs, though HuggingFaceEmbeddings is synchronous primarily, we set devices accordingly)
        device = 'cuda' if os.environ.get(use_cuda_env, 'false').lower() == 'true' else 'cpu'
        self._emb = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device}
        )
        self._store: Optional[PGVector] = None
        self._loaded = False

    def load(self) -> None:
        """Initialize the connection to the pgvector database."""
        if self._loaded:
            return
        
        # Depending on the rag-engine standard, use PGVector wrapper (cosine distance natively supported)
        self._store = PGVector(
            connection=self._connection_string,
            embeddings=self._emb,
            collection_name=self._collection_name,
            use_jsonb=True, # Recommended for flexible metadata
            # pgvector's default distance_strategy is cosine
        )
        self._loaded = True
        
    def _ensure_loaded(self):
        """Guard clause for loaded state."""
        if not self._store:
            raise RuntimeError("PGVector store not loaded. Call load() first.")

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
        
        chunks: List[DocumentChunk] = []
        for doc in retrieved_docs:
            chunk = DocumentChunk(
                id=doc.metadata.get('id', ''),
                chunk_text=doc.page_content,
                chunk_index=doc.metadata.get('chunk_index', 0),
                title=doc.metadata.get('title', 'Unknown'),
                url=doc.metadata.get('url', ''),
                source_type=doc.metadata.get('source_type', 'Unknown')
            )
            chunks.append(chunk)
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
