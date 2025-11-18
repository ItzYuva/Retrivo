from typing import List, Optional
from pydantic import BaseModel

class RAGChunkAndSrc(BaseModel):
    chunks: List[str]            # plural — matches what your loader returns
    source: Optional[str] = None # allow None, or supply a string when available

class RAGUpsertResult(BaseModel):
    ingested: int

class RAGSearchResult(BaseModel):
    contexts: List[str]
    sources: List[str]

class RAGQueryResult(BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int
    