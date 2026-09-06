# Handwritten Character Recognition

A complete pipeline for recognizing handwritten digits and characters using
CNNs, with a built-in extension path to full word/sentence recognition via
a CRNN (CNN + BiLSTM + CTC).

## Results

Trained and evaluated on Kaggle (NVIDIA T4 GPU):

| Dataset | Classes | Val Accuracy | Epochs |
|---|---|---|---|
| MNIST (digits) | 10 | 99.07%+ | 10 |
| EMNIST-balanced (digits + letters) | 47 | 88.27% | 15 |

The CNN architecture (`model.py`) is a compact VGG-style network (~880K
parameters) — two Conv-BN-ReLU blocks followed by fully connected layers.

### A real finding from testing on my own handwriting

After training, I tested the model on phone photos of my own handwritten
letters and initially got poor, low-confidence predictions (~30-50%
confidence, often the same 1-2 letters predicted repeatedly regardless of
what was actually written). Investigating this surfaced two separate
issues, both fixed in this repo:

1. **Normalization mismatch** — the original inference script normalized
   all inputs using MNIST's pixel statistics, even for the EMNIST model,
   which expects different statistics. Fixed by using dataset-specific
   normalization constants.
2. **Domain shift between clean benchmark data and real photos** — EMNIST
   images are tightly cropped, centered, and evenly lit. My phone photos
   had large empty margins, off-center writing, and strong lighting
   gradients from the camera flash/angle. Standard preprocessing
   (grayscale + resize + simple brightness-based inversion) isn't robust
   to this.

To confirm the *model* itself was sound and isolate this as a
preprocessing/data problem, I re-ran inference on genuine EMNIST test-set
images (same format as training data) and got consistently high
confidence (many predictions 95-99.9%), matching the 88.27% held-out
validation accuracy. This confirmed the model generalizes well within its
training distribution, and that robust real-world deployment would need
either a proper bounding-box crop + adaptive thresholding step, or
fine-tuning on a small set of representative real photos — a good example
of the gap between benchmark accuracy and real-world deployment accuracy
that shows up often in applied ML.

## Project layout
