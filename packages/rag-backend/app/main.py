import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import generate, retrieve, status
from dotenv import load_dotenv

# Load .env into os.environ at startup
load_dotenv()

# 여러 개 호스트 허용, 콤마로 구분
FRONTEND_HOSTS = os.getenv("FRONTEND_HOSTS", "http://localhost:5173")
allow_origins = [host.strip() for host in FRONTEND_HOSTS.split(",")]

app = FastAPI(title="rag-backend")

# CORS 설정 (프론트엔드와의 통신을 위해)
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

# Always expose /api/generate
app.include_router(generate.router, prefix="/api")

# Optionally expose /api/retrieve (admin/internal)
if os.getenv("EXPOSE_RETRIEVE_ENDPOINT", "false").lower() == "true":
    app.include_router(retrieve.router, prefix="/api")
    
app.include_router(status.router, prefix="/api")
