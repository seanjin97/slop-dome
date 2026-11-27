import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .adapters.slack import router as slack_router
from .model import IS_AI_THRESHOLD, inference

logger = logging.getLogger("uvicorn.error")

cors_origins = [
    origin
    for origin in map(str.strip, os.getenv("HOST", "").split(","))
    if origin and origin != "*"
]

app = FastAPI()
app.include_router(slack_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
)


class InferenceRequest(BaseModel):
    text: str


class IsAIResponse(BaseModel):
    is_ai: bool


@app.post("/infer/is-ai")
def run_infer_is_ai(request: InferenceRequest) -> IsAIResponse:
    result = inference(request.text)
    logger.info("POST /infer/is-ai %s", result.model_dump_json())
    return IsAIResponse(is_ai=result.ai_score > IS_AI_THRESHOLD)


@app.get("/health")
def health():
    return {"server": "ok"}
