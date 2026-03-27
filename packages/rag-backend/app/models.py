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
    original_chunk_text: Optional[str] = None  # 컨텍스트 주입 전 원본 텍스트
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
    retrieval_query: Optional[str] = None  # 검색에 사용할 별도 쿼리 (없으면 query 사용)
    use_rag: bool = True
    candidate_k: int = int(os.getenv("RERANK_CANDIDATES", 10))  # Default to 10 if not set
    top_k: int = int(os.getenv("TOP_K", 3))  # Default to 3 if not set
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 512))  # Default to 512 if not set
    source_type: Optional[str] = None  # Optional filter by source type
    title: Optional[str] = None  # Optional filter by title keyword
    url: Optional[str] = None  # Optional filter by URL keyword
    # Evaluation fields
    choices: Optional[List[str]] = None  # MCQ options
    ground_truth: Optional[str] = None   # General reference answer
    answer: Optional[str] = None         # MCQ reference answer (for compatibility)
    expected_points: Optional[List[str]] = None  # RAG reference points (for compatibility)

class GenerateResponse(BaseModel):
    response: str
    reference_documents: Optional[List[DocumentChunk]] = []
    prompt: str
    question: str
    elapsed_ms: int
    # Langfuse Standard Fields for LLM-as-a-Judge
    input: str
    output: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    choices: Optional[List[str]] = None
