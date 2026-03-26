from fastapi import APIRouter
from app.routers.generate import retrieval_service, llm_service

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """
    시스템 전체 상태를 확인합니다.
    """
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
    return retrieval_service.get_index_info()

@router.get("/llm/info")
async def get_llm_info():
    """
    LLM 서비스 정보를 반환합니다.
    """
    return llm_service.get_model_info()
