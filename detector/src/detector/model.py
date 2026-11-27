from functools import cache

import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from .utils import sanitise

MODEL_REPO = "wasitaigeneratedcom/tropa-mini-ai-text-detector"
IS_AI_THRESHOLD = 0.7


class LabelProbabilities(BaseModel):
    human: float
    ai: float
    ai_edited: float
    humanized: float


class InferenceResult(BaseModel):
    probabilities: LabelProbabilities
    ai_score: float


class AIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    _tied_weights_keys = []  # transformers>=5 compatibility

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        n = getattr(config, "detector_num_labels", 1)
        self.classifier = nn.Linear(config.hidden_size, n)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask)
        h = out[0]  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)  # mean pooling over real tokens
        return self.classifier(pooled)


@cache
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AIDetectionModel.from_pretrained(MODEL_REPO).eval()
    return tokenizer, model


def inference(text: str) -> InferenceResult:
    """Return the label probabilities and the overall AI score."""
    tokenizer, model = _load_model()
    encoded = tokenizer(
        sanitise(text), truncation=True, max_length=768, return_tensors="pt"
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded), dim=-1)[0].tolist()

    human, ai, ai_edited, humanized = probabilities
    return InferenceResult(
        probabilities=LabelProbabilities(
            human=human, ai=ai, ai_edited=ai_edited, humanized=humanized
        ),
        ai_score=1.0 - human,
    )
