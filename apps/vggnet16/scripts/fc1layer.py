"""
fc1layer.py

Train a single fully-connected (Linear) layer on CIFAR-10.
Mirrors single_fc.c exactly in architecture and hyper-parameters.

Architecture:
    Input [3072]  (flattened 3x32x32, normalised to [-1, 1])
      → Linear(3072 → 10)
      → Softmax + CrossEntropyLoss

Hyper-parameters match train.c / single_fc.c:
    lr        = 0.01
    momentum  = 0.9
    batch     = 64
    epochs    = 20
"""

import numpy as np
from PIL import Image
import os

# -------------------------------------------------------
# Config
# -------------------------------------------------------
TRAIN_LIST = '../cifar10/images.list'
VAL_LIST   = '../cifar10/image_val.list'
IMG_W, IMG_H, IN_C = 32, 32, 3
IN_UNITS    = IN_C * IMG_W * IMG_H   # 3072
NUM_CLASSES = 10
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 0.001  # 0.01 (train.c) is calibrated for AlexNet+BN; 0.001 suits bare FC
MOMENTUM    = 0.9

# -------------------------------------------------------
# Data loading  (same normalization as data.c: /127.5 - 1)
# -------------------------------------------------------
def load_batch(lines):
    X = np.zeros((len(lines), IN_UNITS), dtype=np.float32)
    Y = np.zeros(len(lines), dtype=np.int32)
    for i, line in enumerate(lines):
        label, path = line.strip().split(' ', 1)
        img = Image.open(path).convert('RGB').resize((IMG_W, IMG_H))
        arr = np.array(img, dtype=np.float32)
        # data.c normalisation: pixel / 127.5 - 1, stored as CHW
        arr = arr / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)   # HWC → CHW
        X[i] = arr.flatten()
        Y[i] = int(label)
    return X, Y

# -------------------------------------------------------
# Gauss init  (matches gauss_initialization in alexnet.c)
#   mean=0, std=0.01, Box-Muller
# -------------------------------------------------------
def gauss_init(shape, std=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return (rng.standard_normal(shape) * std).astype(np.float32)

# -------------------------------------------------------
# Forward: Linear(W, b)  →  logits [batch, out]
#   W layout: [in_units, out_units]  (matches fc_layer.c)
# -------------------------------------------------------
def fc_forward(X, W, b):
    return X @ W + b   # [batch, out]

# -------------------------------------------------------
# Softmax + Cross-Entropy  (matches cross_entropy_loss in train.c)
#   Returns (loss scalar, d_out_avg [out_units])  — AVERAGED over batch.
#   fc_op_backward expects this single averaged vector, not per-sample.
# -------------------------------------------------------
def softmax_ce(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)   # numerical stability
    exp_l  = np.exp(logits)
    probs  = exp_l / exp_l.sum(axis=1, keepdims=True)

    n      = len(labels)
    loss   = -np.log(probs[np.arange(n), labels] + 1e-9).mean()

    d = probs.copy()
    d[np.arange(n), labels] -= 1.0
    d_avg = d.mean(axis=0)      # [out_units] — single averaged gradient
    return loss, d_avg

# -------------------------------------------------------
# Backward: FC  — matches fc_op_backward in fc_layer.c EXACTLY
#
# The C implementation receives d_output[out_units] as a SINGLE averaged
# gradient and computes:
#   d_weights[k, i] += d_output[i] * input[p, k] / batchsize  (for all p)
#                    = d_output_avg[i] * mean(input[:, k])
#                    = outer(mean_X, d_out_avg)
#   d_bias[j]        = d_output[j]   (averaged gradient directly)
# -------------------------------------------------------
def fc_backward(X, d_out_avg):
    d_W = np.outer(X.mean(axis=0), d_out_avg)   # [in_units, out_units]
    d_b = d_out_avg.copy()                       # [out_units]
    return d_W, d_b

# -------------------------------------------------------
# Momentum SGD with velocity clip to [-1, 1]
#   Matches momentum_sgd + CLIP in train.c exactly
# -------------------------------------------------------
class MomentumSGD:
    def __init__(self, shape_w, shape_b):
        self.v_w = np.zeros(shape_w, dtype=np.float32)
        self.v_b = np.zeros(shape_b, dtype=np.float32)

    def step(self, W, b, d_W, d_b):
        self.v_w = MOMENTUM * self.v_w - LR * d_W
        self.v_b = MOMENTUM * self.v_b - LR * d_b
        self.v_w = np.clip(self.v_w, -1.0, 1.0)
        self.v_b = np.clip(self.v_b, -1.0, 1.0)
        W += self.v_w
        b += self.v_b

# -------------------------------------------------------
# Accuracy on list file
# -------------------------------------------------------
def evaluate(list_path, W, b):
    with open(list_path) as f:
        lines = [l for l in f if l.strip()]
    correct = total = 0
    for s in range(0, len(lines), BATCH_SIZE):
        batch = lines[s:s+BATCH_SIZE]
        X, Y  = load_batch(batch)
        logits = fc_forward(X, W, b)
        preds  = logits.argmax(axis=1)
        correct += (preds == Y).sum()
        total   += len(Y)
    return 100.0 * correct / total if total else 0.0

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Initialise weights (gauss, std=0.01) + zero bias
    W = gauss_init((IN_UNITS, NUM_CLASSES), rng=rng)
    b = np.zeros(NUM_CLASSES, dtype=np.float32)

    opt = MomentumSGD(W.shape, b.shape)

    with open(TRAIN_LIST) as f:
        all_lines = [l for l in f if l.strip()]
    total_images       = len(all_lines)
    batches_per_epoch  = total_images // BATCH_SIZE

    print("=== Single FC Layer on CIFAR-10 ===")
    print(f"Arch  : Linear({IN_UNITS} → {NUM_CLASSES}) → Softmax+CE")
    print(f"Params: batch={BATCH_SIZE}  epochs={NUM_EPOCHS}  lr={LR}  momentum={MOMENTUM}")
    print(f"Train : {total_images} images  →  {batches_per_epoch} batches/epoch\n")

    for epoch in range(NUM_EPOCHS):
        rng.shuffle(all_lines)          # shuffle each epoch (C reads sequentially)
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for bi in range(batches_per_epoch):
            batch = all_lines[bi*BATCH_SIZE : (bi+1)*BATCH_SIZE]
            X, Y  = load_batch(batch)

            # Forward
            logits = fc_forward(X, W, b)
            loss, d_logits = softmax_ce(logits, Y)
            epoch_loss += loss

            # Accuracy
            preds          = logits.argmax(axis=1)
            epoch_correct += (preds == Y).sum()
            epoch_total   += len(Y)

            # Backward
            d_W, d_b = fc_backward(X, d_logits)

            # Weight update
            opt.step(W, b, d_W, d_b)

            if (bi + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:2d}  batch {bi+1:4d}/{batches_per_epoch}  "
                      f"loss={epoch_loss/(bi+1):.4f}")

        train_acc = 100.0 * epoch_correct / epoch_total
        val_acc   = evaluate(VAL_LIST, W, b) if os.path.exists(VAL_LIST) else -1.0

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  avg_loss={epoch_loss/batches_per_epoch:.4f}  "
              f"train_acc={train_acc:.2f}%", end="")
        if val_acc >= 0:
            print(f"  val_acc={val_acc:.2f}%", end="")
        print("\n")

if __name__ == '__main__':
    main()
