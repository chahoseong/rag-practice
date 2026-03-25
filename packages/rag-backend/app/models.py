from pydantic import BaseModel
from typing import List, Optional

import dotenv
import os
# Load environment variables from .env file
dotenv.load_dotenv()

class RetrievalRequest(BaseModel):
    query: str
    candidate_k: int = int(os.getenv("RERANK_CANDIDATES", 10))  # Default to 10 if not set
    top_k: int = int(os.getenv("TOP_K", 3))  # Default to 3 if not set
    source_type: Optional[str] = None  # Optional filter by source type
    title: Optional[str] = None  # Optional filter by title keyword
    url: Optional[str] = None  # Optional filter by URL keyword

class DocumentChunk(BaseModel):
    id: str
    chunk_text: str
    chunk_index: int
    title: str
    url: str
    source_type: str
    score: Optional[float] = None  # Optional score field

class RetrievalResponse(BaseModel):
    chunks: List[DocumentChunk]
    elapsed_ms: int

class GenerateRequest(BaseModel):
    query: str
    use_rag: bool = True
    candidate_k: int = int(os.getenv("RERANK_CANDIDATES", 10))  # Default to 10 if not set
    top_k: int = int(os.getenv("TOP_K", 3))  # Default to 3 if not set
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 512))  # Default to 512 if not set
    source_type: Optional[str] = None  # Optional filter by source type
    title: Optional[str] = None  # Optional filter by title keyword
    url: Optional[str] = None  # Optional filter by URL keyword

class GenerateResponse(BaseModel):
    response: str
    reference_documents: Optional[List[DocumentChunk]] = []
    prompt: str
    question: str
    elapsed_ms: int
