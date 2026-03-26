import os
from dotenv import load_dotenv

# 1. 다른 모듈을 가져오기 전에 환경 변수부터 로드 및 설정 (Langfuse 연동에 중요)
load_dotenv()

def setup_langfuse_env():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = "unknown"
    
    if provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    elif provider == "huggingface":
        model_name = os.getenv("HUGGINGFACE_MODEL_ID", "model")
    elif provider == "gguf":
        path = os.getenv("GGUF_MODEL_PATH", "model.gguf")
        model_name = os.path.basename(path)
    
    env_str = f"{provider}/{model_name}"
    # Langfuse SDK가 자동 인식하는 환경 변수 주입
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = env_str
    print(f"📡 Langfuse Environment set to: {env_str}")

setup_langfuse_env()

# 2. 이제 설정이 완료된 환경 변수를 사용하는 라우터들을 가져옴
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import generate, retrieve, status

app = FastAPI(title="rag-backend")

# CORS 설정 (프론트엔드와의 통신을 위해)
FRONTEND_HOSTS = os.getenv("FRONTEND_HOSTS", "http://localhost:5173")
allow_origins = [host.strip() for host in FRONTEND_HOSTS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG API Server"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# API 라우터 포함
app.include_router(generate.router, prefix="/api")
if os.getenv("EXPOSE_RETRIEVE_ENDPOINT", "false").lower() == "true":
    app.include_router(retrieve.router, prefix="/api")
app.include_router(status.router, prefix="/api")
