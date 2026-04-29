from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import MAX_LEN, text_to_sequence

ROOT = Path(__file__).resolve().parents[2]


def _resolve(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class TicketDataset(Dataset):
    def __init__(self, csv_file, vocab_file="vocab.json"):
        self.data = pd.read_csv(_resolve(csv_file))

        with open(_resolve(vocab_file), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["cleaned_text"])

        seq = text_to_sequence(text, self.vocab, max_len=MAX_LEN)
        length = min(len(text.split()), MAX_LEN)

        x = torch.tensor(seq, dtype=torch.long)
        y_cls = torch.tensor(row["label"], dtype=torch.long)
        y_reg = torch.tensor(row["urgency_score"], dtype=torch.float32)

        return x, y_cls, y_reg, length
