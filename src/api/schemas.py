"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the documents")
    search_type: str = Field(default="hybrid", description="'semantic' or 'hybrid'")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class SourceInfo(BaseModel):
    source: str
    page: str | int
    relevance_score: float
    chunk_preview: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    search_type: str


class UploadResponse(BaseModel):
    message: str
    num_chunks: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    version: str