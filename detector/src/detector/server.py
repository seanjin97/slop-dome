from fastapi import FastAPI
from pydantic import BaseModel

from .model import inference

app = FastAPI()


class InferenceRequest(BaseModel):
    text: str


class IsAIResponse(BaseModel):
    is_ai: bool


IS_AI_THRESHOLD = 0.7


@app.post("/infer/is-ai")
def run_infer_is_ai(request: InferenceRequest) -> IsAIResponse:
    result = inference(request.text)
    return IsAIResponse(is_ai=result.ai_score > IS_AI_THRESHOLD)


@app.get("/health")
def health():
    return {"server": "ok"}
