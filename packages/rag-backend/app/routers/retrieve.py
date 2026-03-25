from fastapi import APIRouter, HTTPException
from app.models import RetrievalRequest, RetrievalResponse, DocumentChunk
from app.services.retrieval_service import RetrievalService
import time
import os

from sentence_transformers import CrossEncoder

router = APIRouter()
retrieval_service = RetrievalService()
cross_encoder = CrossEncoder(os.getenv("CROSS_ENCODER_MODEL", "dragonkue/bge-reranker-v2-m3-ko"))

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    사용자의 질문으로 관련 문서 검색
    """
    try:
        start_time = time.time()
        
        chunks = await retrieval_service.retrieve(
            question=request.query,
            candidate_k=request.candidate_k,
            top_k=request.candidate_k,
            source_type=request.source_type,
            title=request.title,
            url=request.url
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return RetrievalResponse(
            chunks=chunks,
            elapsed_ms=elapsed_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")
