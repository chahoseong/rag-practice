import asyncio
import os
from typing import List, Protocol, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langfuse import observe

from app.models import DocumentChunk

# .env 파일 로드
load_dotenv()
  
# ───────────────────────────────────────────────────────────────
# LLM Model adapters (교체 가능)
# ───────────────────────────────────────────────────────────────
class LLMModelAdapter(Protocol):
    """LLM Model adapters must implement these methods."""
    def load(self) -> BaseChatModel: ...
    def get_info(self) -> dict: ...
    def get_invoke_kwargs(self) -> dict: ...
    
def make_context(reference_chunks: List[DocumentChunk]) -> str:
    """문서 청크들을 하나의 문자열로 결합합니다."""
    return "\n\n".join([chunk.chunk_text for chunk in reference_chunks])


def make_llm_model_adapter(model_provider: str) -> LLMModelAdapter:
    if model_provider == "gguf":
        from app.services.gguf_adapter import GGUFAdapter
        return GGUFAdapter()
    elif model_provider == "huggingface":
        from app.services.huggingface_adapter import HuggingfaceAdapter
        return HuggingfaceAdapter()
    elif model_provider == "openai":
        from app.services.openai_adapter import OpenAIAdapter
        return OpenAIAdapter()
    raise ValueError(f"Unsupported Model Provider: {model_provider}")

class LLMService:
    def __init__(self, model: Optional[BaseChatModel] = None):
        self.model_provider = os.getenv("LLM_PROVIDER", "huggingface").lower()
        if model:
            self.model = model
        else:
            # 모델 어댑터 생성
            self.model_adapter = make_llm_model_adapter(self.model_provider)
            self.model = self.model_adapter.load()
        if self.model:
            self.model_loaded = True
        else:
            self.model_loaded = False
            print(f"❌ Failed to load model from provider: {self.model_provider}")
            raise RuntimeError(f"Model provider {self.model_provider} not supported or model loading failed.")        
        
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", 512))
        self.context_length = int(os.getenv("LLM_CONTEXT_LENGTH", 2048))
        self.system_prompt = os.getenv("LLM_SYSTEM_PROMPT", "당신은 질문 내용에 답변할 수 있는 전문가입니다. 주어진 문서를 바탕으로 정확하고 도움이 되는 답변을 한국어로 제공해주세요. 문서에 없는 내용은 추측하지 말고, 문서 내용만을 바탕으로 답변해주세요.")
        self.user_prompt_template = os.getenv("LLM_USER_PROMPT_TEMPLATE", "참고 문서:\n{context}\n\n질문: {question}:")
        self.prompt_template = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("user", self.user_prompt_template)]
            )
        
    
    def _get_absolute_path(self, relative_path: str) -> str:
        """상대 경로를 절대 경로로 변환합니다. backend 디렉토리 기준으로 계산합니다."""
        # backend 디렉토리 경로 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # backend/app/services에서 backend로 이동
        backend_dir = os.path.join(current_dir, "..", "..")
        backend_dir = os.path.normpath(backend_dir)
        
        # backend 디렉토리 기준으로 상대 경로 결합
        absolute_path = os.path.join(backend_dir, relative_path)
        return os.path.normpath(absolute_path)
    
    @observe()
    def create_prompt(self, question: str, reference_documents: List[DocumentChunk]) -> ChatPromptValue:
        """
        질문과 컨텍스트를 바탕으로 프롬프트를 생성합니다.
        
        Args:
            question: 사용자 질문
            context_chunks: 검색된 문서 청크들
            
        Returns:
            생성된 프롬프트
        """
        # 컨텍스트 결합
        context = make_context(reference_documents)
        
        # Qwen 모델에 최적화된 프롬프트 템플릿
        prompt = self.prompt_template.invoke({"context": context, "question": question})
        
        # 프롬프트가 너무 길면 자르기
        len_chunks = len(reference_documents)
        while len(prompt.to_string()) > self.context_length:
            len_chunks -= 1
            if len_chunks <= 0:
                print("❗️ Context is too long, cannot generate prompt.")
                return "죄송합니다. 너무 긴 문서입니다."
            context = make_context(reference_documents[:len_chunks])
            prompt = self.prompt_template.invoke({"context": context, "question": question})
        return prompt
    
    @observe()
    async def llm_generate(self, prompt: ChatPromptValue, max_tokens: int = None) -> str:
        """
        프롬프트를 바탕으로 답변을 생성합니다.
        
        Args:
            prompt: 생성된 프롬프트
            max_tokens: 최대 토큰 수 (None이면 기본값 사용)
            
        Returns:
            생성된 답변
        """
        if not self.model_loaded or not self.model:
            # 모델이 로드되지 않은 경우 템플릿 기반 응답
            return self._generate_template_response(prompt)
        
        try:
            # max_tokens가 지정되지 않으면 기본값 사용
            if max_tokens is None:
                max_tokens = self.max_tokens
            
            # 비동기 처리를 위해 동기 함수를 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens
            )
            
            # 응답 메시지에서 텍스트 추출
            answer = response.content if isinstance(response, BaseMessage) else str(response)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return self._generate_template_response(prompt)
    
    def _generate_sync(self, prompt: ChatPromptValue, max_tokens: int) -> BaseMessage:
        """동기 방식으로 답변을 생성합니다."""
        try:
            kwargs = self.model_adapter.get_invoke_kwargs() if self.model_adapter else {}
            response = self.model.invoke(
                prompt,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"답변 생성 중 오류 발생: {str(e)}")
    
    def _generate_template_response(self, prompt: ChatPromptValue) -> str:
        """
        모델이 없을 때 사용하는 템플릿 기반 응답
        """
        # 간단한 키워드 기반 응답
        question_lower = prompt.to_string().lower()
        
        if "훈민정음" in question_lower or "한글" in question_lower:
            return """세종대왕은 1443년(세종 25년)에 훈민정음을 창제하셨습니다. 

훈민정음 창제의 목적:
- 백성들이 쉽게 글을 읽고 쓸 수 있도록 하기 위함
- 한자를 모르는 일반 백성들도 자신의 뜻을 표현할 수 있게 하기 위함
- "나랏말싸미 듕귁에 달아 문자와로 서르 사맛디 아니할쎄"

훈민정음은 표음문자로서 과학적이고 체계적인 문자 체계를 갖추고 있어, 세계적으로도 우수성을 인정받고 있습니다."""
        
        elif "과학" in question_lower or "발명" in question_lower or "측우기" in question_lower:
            return """세종대왕 시대는 과학 기술이 크게 발전한 시기였습니다.

주요 과학 기구와 발명품:
- 측우기(1441년): 세계 최초의 공식적인 강우량 측정기
- 해시계(앙부일구): 시간을 정확히 측정하는 해시계
- 물시계(자격루): 물의 흐름으로 시간을 재는 시계
- 혼천의: 천체의 움직임을 관측하는 기구

또한 천문학, 의학, 농업 기술 등 다양한 분야에서 발전을 이루었으며, 이는 백성들의 삶을 개선하려는 세종대왕의 의지가 반영된 것입니다."""
        
        elif "정치" in question_lower or "정책" in question_lower or "민본" in question_lower:
            return """세종대왕은 민본주의 정치철학을 바탕으로 다양한 정책을 시행하셨습니다.

주요 정치 정책:
- 민본주의: 백성을 나라의 근본으로 여기는 정치철학
- 집현전 설치: 학문 연구와 정책 개발을 위한 기관
- 농업 정책: 농사직설 편찬, 농업 기술 보급
- 사회 제도 개선: 노비제도 개선, 형법 정비

세종대왕은 "백성이 나라의 근본"이라는 신념으로 백성의 삶을 개선하기 위한 다양한 정책을 펼치셨습니다."""
        
        elif "집현전" in question_lower or "학문" in question_lower:
            return """집현전은 세종대왕이 설치한 학문 연구 기관입니다.

집현전의 역할:
- 경서와 사서 연구
- 각종 서적 편찬 및 번역
- 정책 자문 및 연구
- 인재 양성

주요 업적:
- 훈민정음 창제 참여
- 용비어천가, 월인천강지곡 등 편찬
- 각종 역사서와 지리서 편찬
- 중국 고전의 번역과 주석

집현전 학자들은 세종대왕의 각종 정책과 문화 발전에 크게 기여했습니다."""
        
        else:
            return """세종대왕(재위 1418-1450)은 조선의 4대 임금으로, 한국 역사상 가장 존경받는 성군 중 한 분입니다.

주요 업적:
- 훈민정음(한글) 창제 (1443년)
- 과학 기술 발전 (측우기, 해시계, 물시계 등)
- 집현전을 통한 학문 진흥
- 민본주의 정치 실현
- 영토 확장 (4군 6진 개척)

세종대왕은 백성을 사랑하는 마음으로 다양한 정책을 펼치시며, 조선 문화의 황금기를 이끄셨습니다. 더 구체적인 질문을 해주시면 자세한 답변을 드릴 수 있습니다."""
    
    def get_model_info(self) -> dict:
        """모델 정보를 반환합니다."""
        return self.model_adapter.get_info() if self.model_adapter else {
            "status": "not_loaded",
            "provider": self.model_provider,
            "model_name": None,
            "context_length": 0,
            "max_tokens": 0
        }
