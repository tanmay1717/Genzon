"""
Genzon — BERT Fine-Tuning Script
Fine-tunes bert-base-uncased on the fake review detection task.

Usage (local):
    python -m model.bert.train

Usage (Google Colab):
    Copy this file to Colab and run. See notebooks/train_bert.ipynb.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from model.bert.dataset import create_dataloaders
from model.utils.metrics import compute_metrics, print_metrics


# ─── Config ──────────────────────────────────────────────────

class TrainConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    save_dir: str = "model/checkpoints"
    device: str = "auto"   # "auto", "cuda", "cpu"


# ─── Training ────────────────────────────────────────────────

def get_device(config: TrainConfig) -> torch.device:
    """Pick the best available device."""
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  ✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("  ✓ Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("  ⚠ Using CPU (training will be slow)")
    else:
        device = torch.device(config.device)
        print(f"  Using: {device}")

    return device


def train_one_epoch(
    model, train_loader, optimizer, scheduler, device, scaler=None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            avg = total_loss / (i + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"    Batch {i+1}/{n_batches} | Loss: {avg:.4f} | LR: {lr:.2e}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, val_loader, device) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["val_loss"] = total_loss / len(val_loader)

    return metrics


def train(config: TrainConfig | None = None):
    """
    Full BERT training pipeline.
    
    Steps:
        1. Load data
        2. Create DataLoaders
        3. Load pre-trained BERT
        4. Train with AdamW + linear scheduler
        5. Evaluate after each epoch
        6. Save best model
    """
    if config is None:
        config = TrainConfig()

    print("=" * 60)
    print("  Genzon — BERT Fine-Tuning")
    print("=" * 60)

    # ── Device ──
    device = get_device(config)

    # ── Data ──
    print("\n[1/4] Loading data...")
    # Handle both local and Colab paths
    data_dir = Path("data/processed")
    if not data_dir.exists():
        data_dir = Path("../data/processed")
    if not data_dir.exists():
        data_dir = Path("../../data/processed")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    print("\n[2/4] Creating DataLoaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_df, val_df,
        model_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    # ── Model ──
    print("\n[3/4] Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,}")

    # ── Optimizer + Scheduler ──
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = None
    if config.fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("  ✓ FP16 mixed precision enabled")

    # ── Train ──
    print(f"\n[4/4] Training for {config.epochs} epochs...")
    best_f1 = 0.0
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        print(f"\n  ╔══ Epoch {epoch + 1}/{config.epochs} ══╗")
        start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)

        # Evaluate
        val_metrics = evaluate_model(model, val_loader, device)
        elapsed = time.time() - start

        print(f"\n    Train loss: {train_loss:.4f}")
        print(f"    Val loss:   {val_metrics['val_loss']:.4f}")
        print(f"    Val F1:     {val_metrics['f1']:.4f}")
        print(f"    Val AUC:    {val_metrics['auc_roc']:.4f}")
        print(f"    Time:       {elapsed:.0f}s")

        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(save_path / "bert_best")
            tokenizer.save_pretrained(save_path / "bert_best")
            print(f"    ★ New best model saved (F1={best_f1:.4f})")

    # ── Final ──
    print("\n" + "=" * 60)
    print(f"  ✓ Training complete! Best Val F1: {best_f1:.4f}")
    print(f"  ✓ Model saved to: {save_path / 'bert_best'}")
    print("=" * 60)

    return model, tokenizer


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    train()