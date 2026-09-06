"""
Train CharCNN on MNIST (digits) or EMNIST (letters + digits).

Usage:
    python train.py --dataset mnist  --epochs 10
    python train.py --dataset emnist --emnist-split balanced --epochs 15
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CharCNN, count_parameters

NUM_CLASSES = {
    "mnist": 10,
    # EMNIST split -> class count
    "byclass": 62,     # 0-9, A-Z, a-z (unbalanced)
    "bymerge": 47,      # digits + merged-case letters (unbalanced)
    "balanced": 47,   # digits + merged-case letters (balanced) -- recommended
    "letters": 27,       # 1-26 = A-Z (unbalanced), label 0 unused
    "digits": 10,        # EMNIST digits only (balanced, cleaner than MNIST)
    "mnist_split": 10,   # EMNIST's own MNIST-compatible split
}


def fix_emnist_orientation(img):
    """
    EMNIST's raw images are stored transposed relative to how they should
    be displayed/trained (a well-known quirk of the original dataset
    format). This rotates+flips them back to the natural upright reading
    orientation, matching MNIST's convention.
    """
    return transforms.functional.rotate(
        transforms.functional.hflip(img), -90
    )


def get_dataloaders(args):
    if args.dataset == "mnist":
        train_tf = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=test_tf)
        num_classes = NUM_CLASSES["mnist"]

    elif args.dataset == "emnist":
        split = args.emnist_split
        base_tf = [
            transforms.Lambda(fix_emnist_orientation),
        ]
        train_tf = transforms.Compose(base_tf + [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),  # EMNIST mean/std
        ])
        test_tf = transforms.Compose(base_tf + [
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        train_ds = datasets.EMNIST(
            args.data_dir, split=split, train=True, download=True, transform=train_tf
        )
        test_ds = datasets.EMNIST(
            args.data_dir, split=split, train=False, download=True, transform=test_tf
        )
        num_classes = NUM_CLASSES[split]

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, test_loader, num_classes


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train CNN for handwritten character recognition")
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    parser.add_argument("--emnist-split", choices=list(NUM_CLASSES.keys()), default="balanced")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, num_classes = get_dataloaders(args)
    print(f"Dataset: {args.dataset} | Classes: {num_classes} | "
          f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = CharCNN(num_classes=num_classes).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0.0
    ckpt_name = f"{args.dataset if args.dataset == 'mnist' else args.emnist_split}_best.pt"
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_acc)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "dataset": args.dataset,
                "emnist_split": args.emnist_split if args.dataset == "emnist" else None,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {ckpt_path}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
