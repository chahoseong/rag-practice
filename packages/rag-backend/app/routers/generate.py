from fastapi import APIRouter, HTTPException
from app.models import GenerateRequest, GenerateResponse
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
import time
from langfuse import observe

router = APIRouter()
retrieval_service = RetrievalService()
llm_service = LLMService()

@router.post("/generate", response_model=GenerateResponse)
@observe(as_type="generation")
async def generate(request: GenerateRequest):
    """
    사용자 질문에 대한 LLM 응답 생성 (RAG 옵션 포함)
    """
    try:
        start_time = time.time()
        
        reference_documents = []
        
        if request.use_rag:
            # RAG를 사용하는 경우 관련 문서 검색
            reference_documents = await retrieval_service.retrieve(
                query=request.query,
                candidate_k=request.candidate_k,
                top_k=request.top_k,
                source_type=request.source_type,
                title=request.title,
                url=request.url
            )
        
        # 프롬프트 생성
        prompt = llm_service.create_prompt(
            question=request.query,
            reference_documents=reference_documents
        )
        
        # LLM을 이용한 답변 생성
        response = await llm_service.llm_generate(prompt, max_tokens=request.max_tokens)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return GenerateResponse(
            response=response,
            reference_documents=reference_documents,
            prompt=prompt.to_string(),
            question=request.query,
            elapsed_ms=elapsed_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")
