"""
minivgg_torch.py — PyTorch reference for the MiniVGGNet implemented in C.

Mirrors kernel/alexnet.h and main.c layer for layer, so the two can be compared:

    CONV -> ACT -> BN   (note: activation BEFORE batch norm, per the reference
                         MiniVGGNet table), and NO dropout anywhere.

    input        3 x 32 x 32
    conv1  C1    3x3 s1 p1  -> relu -> bn1
    conv2  C1    3x3 s1 p1  -> relu -> bn2
    pool1        2x2 s2                        -> C1 x 16 x 16
    conv3  C2    3x3 s1 p1  -> relu -> bn3
    conv4  C2    3x3 s1 p1  -> relu -> bn4
    pool2        2x2 s2                        -> C2 x 8 x 8
    flatten                                    -> C2 * 64
    fc1    FC1              -> relu -> bn5
    fc2    OUT   (logits; softmax lives in the loss)

Widths come from a uniform multiplier alpha on the reference 32/32/64/64/512:
    alpha = 1.0    -> 32 / 64 / 512   -> 2,215,940 params (CIFAR-100)
    alpha = 0.375  -> 12 / 24 / 192   ->   324,400 params (the C default)

Usage:
    python3 scripts/minivgg_torch.py --summary            # param count only
    python3 scripts/minivgg_torch.py --alpha 1.0 --summary
    python3 scripts/minivgg_torch.py --train --epochs 50  # train on CIFAR-100
    python3 scripts/minivgg_torch.py --export minivgg.weights
        ^ writes a checkpoint in the exact binary layout save_alexnet() uses,
          so the C build can load it (and vice versa).
"""

import argparse
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference MiniVGGNet widths, before the multiplier.
BASE_C1, BASE_C2, BASE_FC = 32, 64, 512


def widths(alpha: float):
    """Uniform width multiplier, matching the C build's -DC1_CHANNELS etc."""
    return round(BASE_C1 * alpha), round(BASE_C2 * alpha), round(BASE_FC * alpha)


class MiniVGG(nn.Module):
    def __init__(self, alpha: float = 0.375, num_classes: int = 100):
        super().__init__()
        c1, c2, fc = widths(alpha)
        self.c1, self.c2, self.fc, self.num_classes = c1, c2, fc, num_classes

        # Block 1
        self.conv1 = nn.Conv2d(3, c1, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c1)

        # Block 2
        self.conv3 = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv4 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(c2)

        # Classifier
        self.fc1 = nn.Linear(c2 * 8 * 8, fc)
        self.bn5 = nn.BatchNorm1d(fc)
        self.fc2 = nn.Linear(fc, num_classes)

        self._init_weights()

    def _init_weights(self):
        # He normal on the weights, zero bias, BN gamma=1 beta=0 — same as
        # he_initialization() in main.c.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # CONV -> ACT -> BN in every block.
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)
        x = self.bn5(F.relu(self.fc1(x)))
        return self.fc2(x)   # logits; CrossEntropyLoss applies the softmax


def param_table(model: MiniVGG):
    rows, total = [], 0
    for name, p in model.named_parameters():
        rows.append((name, tuple(p.shape), p.numel()))
        total += p.numel()
    return rows, total


def export_weights(model: MiniVGG, path: str):
    """Write the checkpoint layout that save_alexnet()/load_alexnet_from_file() use.

    Raw fp32, no header, in this order:
      conv1.w conv1.b  conv2.w conv2.b  conv3.w conv3.b  conv4.w conv4.b
      fc1.w   fc1.b    fc2.w   fc2.b
      bn1.g bn1.b  bn2.g bn2.b  bn3.g bn3.b  bn4.g bn4.b  bn5.g bn5.b

    Layout note: the C conv kernels store weights as [in_ch*k*k, out_ch]
    (see conv_op_backward_full: d_weights[i * oc + j]), whereas PyTorch stores
    [out_ch, in_ch, k, k]. So conv/linear weights are transposed on the way out.
    """
    def conv_w(m):
        # (out, in, kh, kw) -> (in*kh*kw, out)
        oc = m.weight.shape[0]
        return m.weight.detach().reshape(oc, -1).t().contiguous().flatten()

    def lin_w(m):
        # (out, in) -> (in, out), matching weights[i * out_units + j]
        return m.weight.detach().t().contiguous().flatten()

    tensors = [
        conv_w(model.conv1), model.conv1.bias.detach(),
        conv_w(model.conv2), model.conv2.bias.detach(),
        conv_w(model.conv3), model.conv3.bias.detach(),
        conv_w(model.conv4), model.conv4.bias.detach(),
        lin_w(model.fc1), model.fc1.bias.detach(),
        lin_w(model.fc2), model.fc2.bias.detach(),
        model.bn1.weight.detach(), model.bn1.bias.detach(),
        model.bn2.weight.detach(), model.bn2.bias.detach(),
        model.bn3.weight.detach(), model.bn3.bias.detach(),
        model.bn4.weight.detach(), model.bn4.bias.detach(),
        model.bn5.weight.detach(), model.bn5.bias.detach(),
    ]

    n = 0
    with open(path, "wb") as f:
        for t in tensors:
            flat = t.flatten().cpu().numpy().astype("<f4")
            f.write(flat.tobytes())
            n += flat.size
    print(f"wrote {path}: {n} floats ({n * 4} bytes)")
    return n


def train(model, epochs, batch_size, lr, device):
    from torchvision import datasets, transforms

    # Match kernel/data.c load_image(): pixel/127.5 - 1, i.e. uint8 -> [-1, 1].
    # ToTensor() already gives [0, 1], so mean=std=0.5 reproduces it exactly.
    # No augmentation either, since the C loader does none — keeps the two
    # implementations comparable.
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR100("./data", train=True, download=True, transform=tf)
    test_set = datasets.CIFAR100("./data", train=False, download=True, transform=tf)

    train_ld = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_ld = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model.to(device)
    # Same optimiser settings as train.c: SGD, momentum 0.9, weight decay 5e-4,
    # step decay x0.1 every 20 epochs.
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    crit = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        tot_loss = correct = seen = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            tot_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
        sched.step()

        model.eval()
        t_correct = t_seen = 0
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                t_correct += (model(x).argmax(1) == y).sum().item()
                t_seen += y.size(0)

        print(f"epoch {e+1}/{epochs}  lr={sched.get_last_lr()[0]:.6f}  "
              f"loss={tot_loss/len(train_ld):.4f}  "
              f"train acc={correct/seen:.4f}  test acc={t_correct/t_seen:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.375,
                    help="uniform width multiplier (default 0.375, the C default)")
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--summary", action="store_true", help="print the layer table and exit")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--export", metavar="PATH", help="write a checkpoint for the C build")
    args = ap.parse_args()

    model = MiniVGG(alpha=args.alpha, num_classes=args.classes)
    c1, c2, fc = widths(args.alpha)

    rows, total = param_table(model)
    print(f"MiniVGGNet  alpha={args.alpha}  widths {c1}/{c2}/{fc}  classes={args.classes}")
    print(f"{'layer':14s} {'shape':>22s} {'params':>10s}")
    for name, shape, n in rows:
        print(f"{name:14s} {str(shape):>22s} {n:>10,}")
    print(f"{'TOTAL':14s} {'':>22s} {total:>10,}")

    if args.export:
        export_weights(model, args.export)

    if args.summary and not args.train:
        return

    if args.train:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\ntraining on {device}")
        train(model, args.epochs, args.batch_size, args.lr, device)


if __name__ == "__main__":
    main()
