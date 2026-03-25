from fastapi import APIRouter
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """
    시스템 전체 상태를 확인합니다.
    """
    retrieval_service = RetrievalService()
    llm_service = LLMService()
    
    retrieval_info = retrieval_service.get_index_info()
    llm_info = llm_service.get_model_info()
    
    return {
        "system": "RAG Backend",
        "version": "1.0.0",
        "retrieval_service": retrieval_info,
        "llm_service": llm_info,
        "status": "healthy" if retrieval_info["status"] == "loaded" else "partial"
    }

@router.get("/retrieval/info")
async def get_search_info():
    """
    검색 서비스 정보를 반환합니다.
    """
    retrieval_service = RetrievalService()
    return retrieval_service.get_index_info()

@router.get("/llm/info")
async def get_llm_info():
    """
    LLM 서비스 정보를 반환합니다.
    """
    llm_service = LLMService()
    return llm_service.get_model_info()
