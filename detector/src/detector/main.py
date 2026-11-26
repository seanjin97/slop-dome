import torch
import torch.nn as nn
import uvicorn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from .server import app
from .utils import is_ai_text, sanitise

class AIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    _tied_weights_keys = []          # transformers>=5 compatibility

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
        h = out[0]                                    # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)      # mean pooling over real tokens
        return self.classifier(pooled)

repo = "wasitaigeneratedcom/tropa-mini-ai-text-detector"
labels = ["human", "ai", "ai_edited", "humanized"]

text = """
uv run fastapi dev
"""



def main() -> None:
    tok = AutoTokenizer.from_pretrained(repo)
    model = AIDetectionModel.from_pretrained(repo).eval()
    if is_ai_text(text):
        print("confirm AI")
    cleaned = sanitise(text)
    enc = tok(cleaned, truncation=True, max_length=768, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(model(**enc), dim=-1)[0]

    ai_score = 1.0 - probs[0].item()
    print({l: round(p.item(), 4) for l, p in zip(labels, probs)}, "| ai_score:", round(ai_score, 4))
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
