from typing import List, Optional
from app.models import DocumentChunk
import os
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

class VoidRetrievalAdapter:
    def __init__(self):
        self._loaded = False

    def load(self) -> None:
        pass

    def retrieve(self,
                query: str,
                top_k: int = int(os.getenv("TOP_K", 3)),  # Default to 3 if not set
                source_type: Optional[str] = None,  # Optional filter by source type
                title: Optional[str] = None,  # Optional filter by title keyword
                url: Optional[str] = None  # Optional filter by URL keyword
            ) -> List[DocumentChunk]:
        return []

    def count(self) -> int:
        return 0