import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.data import _get_tokenizer, clean_text
from src.models import FinetunedLLM

LABEL_DECODER = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

_model_cache = {}


def load_model(repo_id: str = None) -> FinetunedLLM:
    """Download (if needed) and load the checkpoint, caching it per process."""
    repo_id = repo_id or os.environ["HF_REPO_ID"]
    if repo_id not in _model_cache:
        checkpoint_dir = snapshot_download(repo_id=repo_id, repo_type="model")
        model = FinetunedLLM.load(
            args_fp=Path(checkpoint_dir, "args.json"),
            state_dict_fp=Path(checkpoint_dir, "model.pt"),
        )
        model.eval()
        _model_cache[repo_id] = model
    return _model_cache[repo_id]


def predict(text: str, repo_id: str = None) -> str:
    model = load_model(repo_id)
    tokenizer = _get_tokenizer()
    encoded = tokenizer([clean_text(text)], return_tensors="pt", padding="longest")
    batch = {"ids": encoded["input_ids"], "masks": encoded["attention_mask"]}
    with torch.inference_mode():
        pred_idx = torch.argmax(model(batch), dim=1).item()
    return LABEL_DECODER[pred_idx]


if __name__ == "__main__":
    load_dotenv()
    text = " ".join(sys.argv[1:]) or "Wall Street stocks rallied after the Federal Reserve meeting."
    print(predict(text))
