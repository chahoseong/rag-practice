from dotenv import load_dotenv
import os

from langchain_community.chat_models.llamacpp import ChatLlamaCpp

# .env 파일 로드
load_dotenv()

class GGUFAdapter():
    def __init__(self):
        # Initialize GGUF specific settings
        self.model = None
        self.model_path = os.getenv("GGUF_MODEL_PATH", "data/Qwen2.5-14B-Instruct-Q4_K_S.gguf")
        self.model_name = self.model_path.split("/")[-1].replace(".gguf", "")
        self.context_length = int(os.getenv("LLM_CONTEXT_LENGTH", 2048))
        self.threads = int(os.getenv("GGUF_THREADS", 4))
        self.gpu_layers = int(os.getenv("GGUF_GPU_LAYERS", 0))
        self.batch_size = int(os.getenv("GGUF_BATCH_SIZE", 512))
        
    def load(self) -> ChatLlamaCpp:
        # Load the model from gguf
            
        try:
            print(f"🤖 Loading GGUF model: {self.model_path}")
            print("⏳ This may take a few minutes...")
            
            # GGUF 모델 로드 (환경 변수 사용)
            self.model = ChatLlamaCpp(
                name=self.model_name,
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                n_batch=self.batch_size
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
            "model_provider": "gguf",
            "model_path": self.model_path,
            "status": "loaded" if self.model_loaded else "template_mode",
            "context_length": self.context_length,
            "threads": self.threads,
            "gpu_layers": self.gpu_layers,
            "batch_size": self.batch_size
        }
        
    def get_invoke_kwargs(self) -> dict:
        """LLM 호출 시 사용할 추가 인자를 반환합니다."""
        return {
            "temperature": float(os.getenv("LLM_TEMPERATURE", 0.3)),
            "top_p": float(os.getenv("LLM_TOP_P", 0.8)),
        }