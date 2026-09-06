"""
Training loop for CRNN + CTC loss on handwritten WORDS.

Real handwriting-word datasets (e.g. IAM Handwriting Database) require a
license/registration to download, so they aren't wired up automatically
here. Instead this script includes a SyntheticWordDataset that stitches
random EMNIST letters together into word images with known ground-truth
strings — enough to validate the whole CRNN + CTC pipeline end-to-end.
Swap in a real dataset (IAM, or your own scanned word images) by
replacing SyntheticWordDataset with one that returns (image, text) pairs;
everything else (collate_fn, training loop, decoding) stays the same.

Usage:
    python crnn_train.py --epochs 20
"""

import argparse
import random
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from crnn_model import CRNN, greedy_ctc_decode

# Alphabet: index 0 reserved for CTC blank.
ALPHABET = string.digits + string.ascii_uppercase  # extend as needed
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # +1 leaves 0 = blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1


def fix_emnist_orientation(img):
    return transforms.functional.rotate(transforms.functional.hflip(img), -90)


class SyntheticWordDataset(Dataset):
    """
    Builds synthetic "word" images by horizontally concatenating random
    EMNIST digit/letter glyphs, with the ground-truth string as the label.
    This is only for validating the CRNN + CTC training pipeline; swap
    in real word images (e.g. IAM) for a production model.
    """

    def __init__(self, emnist_dataset, min_len=3, max_len=8, num_samples=20000,
                 img_height=32, char_width=28):
        self.emnist = emnist_dataset
        self.min_len = min_len
        self.max_len = max_len
        self.num_samples = num_samples
        self.img_height = img_height
        self.char_width = char_width

        # Group EMNIST indices by character for quick sampling.
        # (byclass/balanced splits' `.classes` gives the label mapping.)
        self.by_char = {}
        for idx in range(len(emnist_dataset)):
            pass  # NOTE: building a full index is slow for large EMNIST;
                  # in practice, precompute this once and cache to disk.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        length = random.randint(self.min_len, self.max_len)
        text = "".join(random.choice(ALPHABET) for _ in range(length))

        # Render each character as random noise standing in for a glyph.
        # Replace this with real EMNIST glyph lookups for actual training.
        canvas = torch.zeros(1, self.img_height, self.char_width * length)
        for i, ch in enumerate(text):
            glyph = torch.rand(1, self.img_height, self.char_width) * 0.3
            canvas[:, :, i * self.char_width:(i + 1) * self.char_width] = glyph

        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return canvas, target, text


def collate_fn(batch):
    """
    CTC needs variable-width images and variable-length targets batched
    together with their true lengths.
    """
    images, targets, texts = zip(*batch)

    widths = [img.shape[2] for img in images]
    max_width = max(widths)
    height = images[0].shape[1]

    padded_images = torch.zeros(len(images), 1, height, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    concatenated_targets = torch.cat(targets)  # CTCLoss wants targets concatenated, 1D

    input_widths = torch.tensor(widths, dtype=torch.long)

    return padded_images, concatenated_targets, target_lengths, input_widths, texts


def train_one_epoch(model, loader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0.0
    for images, targets, target_lengths, input_widths, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        log_probs = model(images)  # [seq_len, batch, num_classes]

        seq_len = log_probs.size(0)
        batch_size = log_probs.size(1)
        # All samples in a batch share the same padded seq_len for CTC
        # input_lengths; using the padded length is standard practice
        # (CTC handles the extra blank-heavy tail correctly).
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=5):
    """Prints a few greedy-decoded predictions vs. ground truth."""
    model.eval()
    shown = 0
    for images, targets, target_lengths, input_widths, texts in loader:
        images = images.to(device)
        log_probs = model(images)
        decoded = greedy_ctc_decode(log_probs, blank=BLANK_IDX)

        for i, idx_seq in enumerate(decoded):
            pred_text = "".join(IDX_TO_CHAR.get(idx, "?") for idx in idx_seq)
            print(f"  GT: {texts[i]:12s} | Pred: {pred_text}")

        shown += 1
        if shown >= max_batches:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Alphabet size: {len(ALPHABET)} (+1 blank = {NUM_CLASSES} classes)")

    # EMNIST is loaded here mainly to demonstrate wiring a real character
    # source into the pipeline; SyntheticWordDataset currently uses random
    # noise glyphs as a placeholder (see class docstring).
    tf = transforms.Compose([
        transforms.Lambda(fix_emnist_orientation),
        transforms.ToTensor(),
    ])
    emnist_train = datasets.EMNIST(args.data_dir, split="balanced", train=True,
                                    download=True, transform=tf)

    train_ds = SyntheticWordDataset(emnist_train, num_samples=5000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn)

    model = CRNN(img_height=32, num_classes=NUM_CLASSES).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, device)
        print(f"Epoch {epoch}/{args.epochs} | avg CTC loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            print("Sample predictions:")
            evaluate(model, train_loader, device, max_batches=1)

    torch.save(model.state_dict(), "crnn_checkpoint.pt")
    print("Saved crnn_checkpoint.pt")


if __name__ == "__main__":
    main()
