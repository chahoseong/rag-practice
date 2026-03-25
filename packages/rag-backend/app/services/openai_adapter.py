from dotenv import load_dotenv
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

# .env 파일 로드
load_dotenv()
class OpenAIAdapter():
    def __init__(self):
        self.model = None
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5-nano")
        self.model_loaded = False
            
        
    def load(self) -> BaseChatModel:
        # Load the model from Huggingface
            
        try:
            api_key=os.getenv("OPENAI_API_KEY", None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            self.model = init_chat_model(
                            self.model_name,
                            model_provider="openai",
                            api_key=api_key,
                        )
            
            self.model_loaded = True
            print(f"✅ model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("💡 Fallback to template-based responses")
            self.model = None
            self.model_loaded = False
            
        return self.model
    
    def get_info(self) -> dict:
        """모델 정보를 반환합니다."""
        return {
            "model_provider": "openai",
            "status": "loaded" if self.model_loaded else "template_mode",
            "model_name": self.model_name
        }
        
    def get_invoke_kwargs(self) -> dict:
        """LLM 호출 시 사용할 추가 인자를 반환합니다."""
        return {}