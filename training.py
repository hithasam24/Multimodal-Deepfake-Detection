from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from utils.dataset import LAVDFMultimodalDataset
from models.cmgan import CMGANOnlyModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "data/LAV-DF"
TRAIN_CSV = "outputs/manifests/train_cmgan_full.csv"
VAL_CSV = "outputs/manifests/val_cmgan_full.csv"
TEST_CSV = "outputs/manifests/test_cmgan_full.csv"

BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 5e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 2
GRAD_CLIP = 1.0

bce_loss = nn.BCEWithLogitsLoss()


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    losses, all_labels, all_probs = [], [], []
    scaler = torch.amp.GradScaler("cuda")

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        frames = batch["frames"].to(DEVICE, non_blocking=True)
        mel = batch["mel"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                outputs = model(frames, mel)
                loss = bce_loss(outputs["logits"], labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

        probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
        losses.append(loss.item())
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=np.mean(losses))

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    train_ds = LAVDFMultimodalDataset(TRAIN_CSV, DATA_ROOT)
    val_ds = LAVDFMultimodalDataset(VAL_CSV, DATA_ROOT)
    test_ds = LAVDFMultimodalDataset(TEST_CSV, DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CMGANOnlyModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -1
    best_path = Path("outputs/checkpoints/best_cmgan_full.pt")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)

        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = run_epoch(model, test_loader)
    print("\nFull CM-GAN Test Metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
