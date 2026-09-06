"""
Run inference on a single handwritten character image using a trained
checkpoint from train.py.

Usage:
    python infer.py --checkpoint checkpoints/mnist_best.pt --image digit.png --dataset mnist
    python infer.py --checkpoint checkpoints/balanced_best.pt --image letter.png --dataset emnist --emnist-split balanced
"""

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import CharCNN

# EMNIST-balanced label mapping (47 classes): index -> character
# Order matches the official EMNIST "balanced" mapping file.
EMNIST_BALANCED_LABELS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abdefghnqrt"
)


def load_and_preprocess(image_path: str) -> torch.Tensor:
    """
    Loads an arbitrary image, converts to grayscale, resizes to 28x28,
    and inverts colors if needed so the character is white-on-black
    (matching MNIST/EMNIST convention: background=0, ink=high value).
    """
    img = Image.open(image_path).convert("L")

    # Heuristic: if background looks light (mean pixel > 127), assume
    # black-ink-on-white-paper and invert to match training data.
    if sum(img.getdata()) / (img.width * img.height) > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # use EMNIST stats if using EMNIST model
    ])
    return tf(img).unsqueeze(0)  # add batch dimension -> [1, 1, 28, 28]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = CharCNN(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = load_and_preprocess(args.image).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_idx = pred_idx.item()
    conf = conf.item()

    if args.dataset == "mnist":
        label = str(pred_idx)
    else:
        label = EMNIST_BALANCED_LABELS[pred_idx] if pred_idx < len(EMNIST_BALANCED_LABELS) else str(pred_idx)

    print(f"Predicted character: '{label}'  (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
