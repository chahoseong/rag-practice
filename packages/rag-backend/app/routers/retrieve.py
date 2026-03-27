from fastapi import APIRouter, HTTPException
from app.models import RetrievalRequest, RetrievalResponse, DocumentChunk
import time

router = APIRouter()

def _get_retrieval_service():
    """generate 라우터와 동일한 RetrievalService 인스턴스를 공유합니다."""
    from app.routers.generate import retrieval_service
    return retrieval_service

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    사용자의 질문으로 관련 문서 검색
    """
    try:
        start_time = time.time()

        retrieval_service = _get_retrieval_service()
        chunks = await retrieval_service.retrieve(
            query=request.query,
            candidate_k=request.candidate_k,
            top_k=request.candidate_k,
            source_type=request.source_type,
            title=request.title,
            url=request.url,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RetrievalResponse(
            chunks=chunks,
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")
