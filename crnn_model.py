"""
CRNN (Convolutional Recurrent Neural Network) for handwritten WORD/LINE
recognition — the extension path beyond single-character CNN classification.

Why a different architecture is needed:
    A single-character CNN (model.py) takes one fixed-size image and
    outputs one label. A word is a variable number of characters at
    variable positions, so we need a model that:
      1. Extracts a spatial feature map from the whole word image (CNN)
      2. Reads that feature map left-to-right as a SEQUENCE of columns
      3. Predicts a character (or "blank") at each sequence step (RNN)
      4. Aligns the variable-length predicted sequence to the target
         text WITHOUT needing per-character bounding boxes (CTC loss)

Architecture (standard CRNN, Shi et al. 2015):
    Input:  1 x 32 x W  (grayscale word image, fixed height, variable width)
    CNN:    7 conv layers -> feature map of shape [C, 1, W']
    Reshape: squeeze height -> sequence of W' feature vectors
    RNN:    2-layer bidirectional LSTM over the W' timesteps
    Output: [W', batch, num_classes]  log-probabilities per timestep
    Loss:   CTCLoss aligns this to the target character string
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height: int = 32, num_channels: int = 1,
                 num_classes: int = 80, rnn_hidden: int = 256):
        """
        num_classes should be (alphabet size + 1) — the extra class is
        the CTC "blank" token, conventionally index 0.
        """
        super().__init__()
        assert img_height % 16 == 0, "img_height must be a multiple of 16"

        # CNN feature extractor. Progressively downsamples height to 1
        # while preserving most of the width resolution (so each
        # timestep in the output sequence corresponds to a narrow
        # vertical strip of the original image).
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H: /2, W: /2

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H: /4, W: /4

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: /8, W unchanged

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: /16, W unchanged

            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            # H: (img_height/16 - 1) -> collapses to 1 for img_height=32
        )

        # Sequence modeling: bidirectional LSTM reads the width-wise
        # sequence of CNN feature vectors.
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=rnn_hidden, num_layers=2,
            bidirectional=True, batch_first=False, dropout=0.2,
        )

        # Project BiLSTM output (2 * rnn_hidden, because bidirectional)
        # to per-timestep class scores.
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, height, width]
        returns: [seq_len, batch, num_classes] log-probabilities,
                 the format expected by nn.CTCLoss
        """
        conv_out = self.cnn(x)  # [batch, 512, 1, W']
        b, c, h, w = conv_out.size()
        assert h == 1, f"Expected CNN output height 1, got {h}. Check img_height/pooling."

        conv_out = conv_out.squeeze(2)          # [batch, 512, W']
        conv_out = conv_out.permute(2, 0, 1)     # [W', batch, 512]  (seq_len, batch, features)

        rnn_out, _ = self.rnn(conv_out)          # [W', batch, 2*rnn_hidden]
        logits = self.fc(rnn_out)                 # [W', batch, num_classes]
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs


def greedy_ctc_decode(log_probs: torch.Tensor, blank: int = 0):
    """
    Simple greedy CTC decoder: take argmax at each timestep, then
    collapse repeated consecutive labels and drop blanks.

    log_probs: [seq_len, batch, num_classes]
    returns: list of length `batch`, each a list of predicted class indices
    """
    # [seq_len, batch]
    best_path = log_probs.argmax(dim=2)
    seq_len, batch = best_path.shape

    decoded = []
    for b in range(batch):
        prev = None
        chars = []
        for t in range(seq_len):
            idx = best_path[t, b].item()
            if idx != blank and idx != prev:
                chars.append(idx)
            prev = idx
        decoded.append(chars)
    return decoded


if __name__ == "__main__":
    # Smoke test: word image of height 32, width 128 -> sequence length 31
    model = CRNN(img_height=32, num_classes=80)
    dummy = torch.randn(4, 1, 32, 128)  # batch=4
    out = model(dummy)
    print(f"Output shape [seq_len, batch, num_classes]: {out.shape}")

    decoded = greedy_ctc_decode(out)
    print(f"Greedy-decoded label indices per sample: {decoded}")
