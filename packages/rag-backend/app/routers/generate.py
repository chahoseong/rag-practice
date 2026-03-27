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
            # retrieval_query가 별도로 지정된 경우 검색에는 그것을 사용
            search_query = request.retrieval_query or request.query
            reference_documents = await retrieval_service.retrieve(
                query=search_query,
                candidate_k=request.candidate_k,
                top_k=request.top_k,
                source_type=request.source_type,
                title=request.title,
                url=request.url,
            )
        
        # 프롬프트 생성
        prompt = llm_service.create_prompt(
            question=request.query,
            reference_documents=reference_documents
        )
        
        # LLM을 이용한 답변 생성
        response = await llm_service.llm_generate(prompt, max_tokens=request.max_tokens)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Langfuse 평가를 위한 데이터 구성
        input_text = request.query
        if request.choices:
            input_text += f"\n선택지: {', '.join(request.choices)}"
            
        contexts = [doc.original_chunk_text or doc.chunk_text for doc in reference_documents] if reference_documents else []
        
        # 실제 정답(Ground Truth) 결정 로직
        # 1. 직접 ground_truth가 들어오면 사용
        # 2. answer (MCQ) 가 들어오면 사용
        # 3. expected_points (RAG) 가 들어오면 리스트를 문자열로 합쳐서 사용
        final_ground_truth = request.ground_truth
        if not final_ground_truth:
            if request.answer:
                final_ground_truth = request.answer
            elif request.expected_points:
                final_ground_truth = "; ".join(request.expected_points)
        
        return GenerateResponse(
            response=response,
            reference_documents=reference_documents,
            prompt=prompt.to_string(),
            question=request.query,
            elapsed_ms=elapsed_ms,
            # 상세 평가 필드 매핑
            input=input_text,
            output=response,
            contexts=contexts,
            ground_truth=final_ground_truth,
            choices=request.choices
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")
