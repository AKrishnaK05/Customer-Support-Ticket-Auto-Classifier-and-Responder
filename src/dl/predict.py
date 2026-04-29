from pathlib import Path
import json

import pandas as pd
import torch

from .model import TicketLSTM
from .utils import MAX_LEN, EMBED_DIM, HIDDEN_DIM, clean_text, text_to_sequence
from src.nlp.extractor import extract_intent
from src.nlp.responder import generate_response

ROOT = Path(__file__).resolve().parents[2]


with open(ROOT / "data" / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

labels = pd.read_csv(ROOT / "data" / "label_mapping.csv")
num_classes = len(labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TicketLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes).to(device)
model.load_state_dict(torch.load(ROOT / "models" / "best_model.pth", map_location=device, weights_only=True))
model.eval()


def predict_ticket(text, subject=""):
    combined_text = ("subject " + str(subject).strip() + " body " + str(text).strip()).strip()
    text = clean_text(combined_text)

    seq = text_to_sequence(text, vocab, max_len=MAX_LEN)
    length = min(len(text.split()), MAX_LEN)

    x = torch.tensor([seq], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        pred_cls, pred_reg = model(x, lengths)
        category_id = torch.argmax(pred_cls, dim=1).item()
        urgency = pred_reg.item()

    category = labels.iloc[category_id]["Issue_Category"]
    return {"category": category, "urgency": float(round(urgency, 3))}


def map_category_to_simple(dl_category):
    name = str(dl_category).lower()
    if "billing" in name or "returns" in name or "payment" in name:
        return "billing"
    if "technical" in name or "it support" in name or "service outages" in name:
        return "technical"
    if "fraud" in name or "security" in name:
        return "fraud"
    if "human" in name or "hr" in name or "customer service" in name:
        return "general inquiry"
    if "product" in name or "sales" in name:
        return "general inquiry"
    return "general inquiry"


def integrated_predict(text, subject=""):
    dl_out = predict_ticket(text, subject)
    dl_category = dl_out.get("category", "")
    urgency = dl_out.get("urgency", 0.0)

    simple_cat = map_category_to_simple(dl_category)
    subj, action = extract_intent(text)
    response = generate_response(simple_cat, subj, action)

    return {
        "dl_category": dl_category,
        "category": simple_cat,
        "urgency": urgency,
        "subject": subj,
        "action": action,
        "response": response,
    }


if __name__ == "__main__":
    print(predict_ticket("i received a damaged product what should i do with it"))
