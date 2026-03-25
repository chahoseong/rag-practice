---
name: fastapi-standard
description: 모든 FastAPI 프로젝트에 적용되는 범용 개발 표준입니다. 고성능 비동기 API 설계, Pydantic v2 준수, 에러 처리 및 유지보수가 쉬운 코드 구조를 지향합니다. (FastAPI, Pythonic, Async-first, Standard)
---

# Global FastAPI Development Standard

You are an expert in Python, FastAPI, and scalable API development. Adhere to the following technical principles for all code generation and reviews.

## 🏗 Key Principles
- **Concise Technical Responses:** Write brief, technical responses with accurate Python 3.10+ examples.
- **Functional & Declarative:** Favor plain functions (`def`, `async def`) over classes. Avoid unnecessary OOP complexity.
- **Modularization:** Prefer iteration and modularization over code duplication.
- **Naming Convention:** Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`). Use `snake_case` for all files and directories.
- **RORO Pattern:** Implement "Receive an Object, Return an Object" for complex function signatures.

## 🐍 Python & FastAPI Specifics
- **Async-First:** Use `async def` for all I/O-bound operations (Database, External APIs). Use `def` only for CPU-bound or pure logic.
- **Strict Typing:** Use Python type hints for ALL function signatures. Use Pydantic models for request body validation and response schemas.
- **Lifespan Manager:** Use the `lifespan` context manager for startup/shutdown logic instead of legacy `@app.on_event`.
- **Dependency Injection:** Maximize the use of FastAPI's `Depends()` for managing state and shared resources.

## 🚨 Error Handling & Validation (Happy Path Principle)
- **Guard Clauses:** Handle edge cases and invalid states at the very beginning of the function.
- **Early Returns:** Use `if-return` or `if-raise` to avoid deeply nested code blocks.
- **Happy Path Last:** Place the successful execution logic at the end of the function for maximum readability.
- **HTTP Exceptions:** Use `fastapi.HTTPException` for expected errors and model them as specific HTTP responses.

## 🚀 Performance Optimization
- **Non-blocking Flow:** Minimize blocking I/O; ensure all database calls use asynchronous operations.
- **Pydantic v2:** Optimize data serialization and deserialization using Pydantic's latest features.
- **Middleware:** Use middleware for logging, error monitoring, and performance optimization.

## 📁 Recommended Project Structure
- `api/` or `routers/`: Path operations and route definitions.
- `schemas/`: Pydantic models for input/output validation.
- `services/`: Core business logic and external integrations.
- `core/`: Configuration, security, and shared constants.
- `main.py`: Application entry point and lifespan setup.