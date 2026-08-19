import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel

MODEL_NAME = "allenai/scibert_scivocab_uncased"


class FinetunedLLM(nn.Module):
    def __init__(self, llm, dropout_p, embedding_dim, num_classes):
        super().__init__()
        self.llm = llm
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        seq, pool = self.llm(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        z = self(batch)
        return torch.argmax(z, dim=1).cpu().numpy()

    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            json.dump(
                {
                    "dropout_p": self.dropout_p,
                    "embedding_dim": self.embedding_dim,
                    "num_classes": self.num_classes,
                },
                fp,
                indent=4,
                sort_keys=False,
            )
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp)
        llm = BertModel.from_pretrained(MODEL_NAME, return_dict=False)
        model = cls(llm=llm, **kwargs)
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        return model
