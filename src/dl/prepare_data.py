from pathlib import Path
from collections import Counter
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import VOCAB_SIZE

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("Loading data...")
    df = pd.read_csv(ROOT / "data" / "cleaned_customer_support_tickets.csv")

    print(f"Total samples: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print("Building vocabulary from training data...")
    all_words = " ".join(train_df["cleaned_text"].astype(str)).split()
    word_counts = Counter(all_words)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)

    print(f"Vocabulary Size: {len(vocab)}")

    with open(ROOT / "data" / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4)
    print("Saved vocab.json")

    train_df.to_csv(ROOT / "data" / "train.csv", index=False)
    val_df.to_csv(ROOT / "data" / "val.csv", index=False)
    test_df.to_csv(ROOT / "data" / "test.csv", index=False)
    print("Saved train.csv, val.csv, and test.csv")


if __name__ == "__main__":
    main()
