---
name: rag-engine
description: 고성능 RAG(Retrieval-Augmented Generation) 시스템 구축을 위한 가이드입니다. 문서 분할(Chunking), 벡터 검색(Vector Search), 프롬프트 엔지니어링 및 답변 품질 평가(Evaluation) 표준을 제공합니다. (RAG, VectorDB, pgvector, Langfuse, Embedding)
---

# RAG Implementation & Optimization Standard

You are an expert in RAG architecture and Large Language Model (LLM) orchestration. Follow these guidelines to build production-ready RAG pipelines.

## ✂️ Data Ingestion & Chunking
- **Semantic Chunking:** Prioritize semantic meaning over fixed character counts. Use overlap (10-20%) to maintain context between chunks.
- **Metadata Enrichment:** Always attach source metadata (filename, page number, section title) to each chunk for better traceability.
- **Clean Input:** Pre-process and clean raw text (remove redundant whitespaces, HTML tags) before embedding to improve vector quality.

## 🔍 Retrieval Strategy
- **Hybrid Search:** Combine Keyword Search (BM25) and Vector Search (Cosine Similarity) to handle both specific terminology and semantic intent.
- **Top-k Tuning:** Start with `k=3` to `k=5` for retrieval. Use Re-ranking models if the initial results are noisy.
- **Vector DB Optimization:** Use HNSW (Hierarchical Navigable Small World) indexing for `pgvector` or Qdrant to ensure sub-second latency in large datasets.

## ✍️ Generation & Prompting
- **Context Grounding:** Ensure the system prompt strictly instructs the model to "Answer ONLY based on the provided context."
- **Citation:** Force the model to cite the specific source/chunk used for each part of the answer to prevent hallucinations.
- **Handling "I Don't Know":** Explicitly define a fallback response if the retrieved context does not contain the answer.

## 📊 Monitoring & Evaluation
- **Tracing with Langfuse:** Instrument every step (Retrieval, Prompt, Generation) with Langfuse traces for debugging.
- **RAG Triad:** Evaluate the pipeline based on:
  1. **Context Relevance:** Is the retrieved context actually useful?
  2. **Faithfulness:** Is the answer derived strictly from the context?
  3. **Answer Relevance:** Does the answer address the user's query?

## 🛠 Tech Stack Recommendations
- **Embedding:** `text-embedding-3-small` (Cost-effective) or `text-embedding-3-large`.
- **Vector Store:** `pgvector` (PostgreSQL extension) for seamless relational data integration.
- **Orchestration:** LangChain or LlamaIndex (or direct FastAPI implementation for lightweight apps).