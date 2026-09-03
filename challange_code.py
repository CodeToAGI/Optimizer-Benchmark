"""
CodeToAGI — Deep Learning Series EP19 Challenge
================================================
Benchmark Every Optimizer on an EP18-style MLP

Task
----
1. Train the same MLP (BatchNorm + Dropout) with three optimizers:
   - SGD (lr=0.01, momentum=0.9)
   - Adam (lr=1e-3)
   - AdamW (lr=1e-3, weight_decay=1e-4)
2. Record validation accuracy and wall-clock time for each.
3. Plot loss curves and learning-rate schedules.
4. Answer: which converged fastest? which reached the best final accuracy?
5. Post your results table in the YouTube comments.

Run
---
    python ep19_optimizer_benchmark.py

Requirements
------------
    pip install torch torchvision matplotlib tqdm

The script is self-contained (MNIST). If you already have your EP18 model,
swap the MLP class and data loaders — the rest stays the same.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data")
OUT_DIR = Path("./ep19_results")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 12          # short enough for a laptop run, long enough to see differences
NUM_WORKERS = 2
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Model — EP18-style MLP with BatchNorm + Dropout
# ──────────────────────────────────────────────────────────────────────────────
class EP18MLP(nn.Module):
    """Simple but solid MLP matching the style used in EP18."""

    def __init__(self, in_dim: int = 784, hidden: int = 256, num_classes: int = 10, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def get_loaders() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval helpers
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Single-run experiment
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(
    name: str,
    optimizer_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Dict:
    print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")
    model = EP18MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    # OneCycleLR — same schedule style as EP18
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    start = time.perf_counter()
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        best_acc = max(best_acc, va_acc)
        print(
            f"  Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_acc={va_acc:.2f}% | lr={current_lr:.2e}"
        )

    elapsed = time.perf_counter() - start
    print(f"  → Finished in {elapsed:.1f}s | best val_acc = {best_acc:.2f}%")

    return {
        "name": name,
        "history": history,
        "best_acc": best_acc,
        "final_acc": history["val_acc"][-1],
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optimizers under test
# ──────────────────────────────────────────────────────────────────────────────
def make_optimizers():
    return {
        "SGD (momentum=0.9)": lambda params: optim.SGD(
            params, lr=0.01, momentum=0.9, nesterov=True
        ),
        "Adam": lambda params: optim.Adam(params, lr=1e-3),
        "AdamW": lambda params: optim.AdamW(
            params, lr=1e-3, weight_decay=1e-4
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(results: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Validation loss
    ax = axes[0]
    for r in results:
        ax.plot(r["history"]["val_loss"], label=r["name"], linewidth=2)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation accuracy
    ax = axes[1]
    for r in results:
        ax.plot(r["history"]["val_acc"], label=r["name"], linewidth=2)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning rate
    ax = axes[2]
    for r in results:
        ax.plot(r["history"]["lr"], label=r["name"], linewidth=2)
    ax.set_title("Learning Rate (OneCycleLR)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out_path = OUT_DIR / "optimizer_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")
    plt.close()


def print_summary_table(results: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Optimizer':<22} {'Best Acc':>10} {'Final Acc':>11} {'Time (s)':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<22} {r['best_acc']:>9.2f}% {r['final_acc']:>10.2f}% {r['elapsed']:>10.1f}"
        )
    print("=" * 72)

    fastest = min(results, key=lambda x: x["elapsed"])
    best = max(results, key=lambda x: x["best_acc"])
    print(f"\nFastest wall-clock : {fastest['name']} ({fastest['elapsed']:.1f}s)")
    print(f"Best final accuracy: {best['name']} ({best['best_acc']:.2f}%)")
    print("\n→ Copy the table above and post it in the YouTube comments!")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader = get_loaders()
    optimizers = make_optimizers()

    results = []
    for name, opt_fn in optimizers.items():
        res = run_experiment(name, opt_fn, train_loader, val_loader)
        results.append(res)

    plot_results(results)
    print_summary_table(results)

    # Save raw numbers for later analysis
    torch.save(
        {r["name"]: {"best_acc": r["best_acc"], "final_acc": r["final_acc"], "elapsed": r["elapsed"],
                     "history": r["history"]} for r in results},
        OUT_DIR / "results.pt",
    )
    print(f"\nRaw results saved → {OUT_DIR / 'results.pt'}")


if __name__ == "__main__":
    main()
