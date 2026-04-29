from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .dataset import TicketDataset
from .model import TicketLSTM
from .utils import EMBED_DIM, HIDDEN_DIM

ROOT = Path(__file__).resolve().parents[2]


def evaluate(model, dataloader, criterion_cls, criterion_reg, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []
    all_reg_preds = []
    all_reg_true = []

    with torch.no_grad():
        for X_batch, y_cls_batch, y_reg_batch, lengths_batch in dataloader:
            X_batch = X_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            pred_cls, pred_reg = model(X_batch, lengths_batch)
            loss_cls = criterion_cls(pred_cls, y_cls_batch)
            loss_reg = criterion_reg(pred_reg, y_reg_batch)
            loss = loss_cls + 0.01 * loss_reg
            total_loss += loss.item()

            all_preds.extend(torch.argmax(pred_cls, dim=1).cpu().numpy())
            all_true.extend(y_cls_batch.cpu().numpy())
            all_reg_preds.extend(pred_reg.cpu().numpy())
            all_reg_true.extend(y_reg_batch.cpu().numpy())

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    f1_per_class = f1_score(all_true, all_preds, average=None)
    mse = mean_squared_error(all_reg_true, all_reg_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, acc, f1, f1_per_class, mse


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_data = TicketDataset(ROOT / "data" / "train.csv")
    val_data = TicketDataset(ROOT / "data" / "val.csv")
    test_data = TicketDataset(ROOT / "data" / "test.csv")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    num_classes = len(pd.read_csv(ROOT / "data" / "label_mapping.csv"))
    model = TicketLSTM(len(train_data.vocab), EMBED_DIM, HIDDEN_DIM, num_classes).to(device)

    y_train_all = train_data.data["label"].values
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_all), y=y_train_all)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion_cls = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 15
    best_val_acc = float("-inf")

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_cls_batch, y_reg_batch, lengths_batch in train_loader:
            X_batch = X_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            optimizer.zero_grad()
            pred_cls, pred_reg = model(X_batch, lengths_batch)
            loss_cls = criterion_cls(pred_cls, y_cls_batch)
            loss_reg = criterion_reg(pred_reg, y_reg_batch)
            loss = loss_cls + 0.01 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, val_mse = evaluate(model, val_loader, criterion_cls, criterion_reg, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val MSE: {val_mse:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ROOT / "models" / "best_model.pth")
            print(">>> Saved new best model by validation accuracy (best_model.pth) <<<")

    print("\nTraining complete. Evaluating on Test Set using best model...")
    model.load_state_dict(torch.load(ROOT / "models" / "best_model.pth", map_location=device, weights_only=True))
    test_loss, test_acc, test_f1, test_f1_per_class, test_mse = evaluate(model, test_loader, criterion_cls, criterion_reg, device)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")
    print(f"Test MSE      : {test_mse:.4f}")

    print("\nPer-Class F1 Scores:")
    labels_df = pd.read_csv(ROOT / "data" / "label_mapping.csv")
    for i, score in enumerate(test_f1_per_class):
        class_name = labels_df[labels_df["label"] == i]["Issue_Category"].values[0]
        print(f"  - {class_name.capitalize()}: {score:.4f}")


if __name__ == "__main__":
    main()
