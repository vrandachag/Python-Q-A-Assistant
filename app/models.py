from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="How do I reverse a list in Python?")

class SourceDoc(BaseModel):
    title: str
    score: float
    snippet: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]
    latency_ms: float
