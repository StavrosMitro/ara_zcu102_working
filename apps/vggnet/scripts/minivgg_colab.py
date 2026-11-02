# =============================================================================
# MiniVGGNet on CIFAR-100 — PyTorch reference for the C implementation
# Paste this whole thing into one Google Colab cell.
# Runtime -> Change runtime type -> GPU (T4 is enough; ~5-10 min for 30 epochs)
#
# Mirrors kernel/alexnet.h + main.c + train.c:
#   CONV -> ACT -> BN  (activation BEFORE batch norm), NO dropout
#   alpha = 0.375 uniform width multiplier -> 12 / 24 / 192 -> 324,400 params
#   SGD lr=0.01, momentum=0.9, weight_decay=5e-4 (weights only, not biases/BN)
#   LR x0.1 every 12 epochs, batch 64, 30 epochs
#   inputs scaled to [-1, 1] (pixel/127.5 - 1), no data augmentation
#   40k train / 10k held-out eval split of the 50k CIFAR-100 train set
# =============================================================================

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------- config ----
ALPHA         = 0.375      # width multiplier; 1.0 = reference MiniVGGNet
NUM_CLASSES   = 100
BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
LR_STEP       = 12         # x0.1 every LR_STEP epochs
LR_GAMMA      = 0.1
EVAL_PERCENT  = 20         # held-out fraction of the 50k train set
SEED          = 42

# Regularisation — matches the C build's Makefile knobs (AUGMENT, DROP*, WD).
AUGMENT       = True       # random horizontal flip on training batches
DROPOUT1      = 0.25       # after pool1
DROPOUT2      = 0.25       # after pool2
DROPOUT_FC    = 0.50       # after the FC head

BASE_C1, BASE_C2, BASE_FC = 32, 64, 512   # reference MiniVGGNet widths

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
if device == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")
    # Speed knobs, harmless if unsupported. TF32 lets the T4/A100 use faster
    # matmuls; cudnn.benchmark picks the best conv algorithm for a fixed shape.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("=" * 60)
    print("WARNING: running on CPU — this will be very slow.")
    print("Colab menu -> Runtime -> Change runtime type -> Hardware")
    print("accelerator: GPU (T4), then Runtime -> Restart and run all.")
    print("=" * 60)

# ----------------------------------------------------------------- model ----
class MiniVGG(nn.Module):
    def __init__(self, alpha=ALPHA, num_classes=NUM_CLASSES):
        super().__init__()
        c1 = round(BASE_C1 * alpha)
        c2 = round(BASE_C2 * alpha)
        fc = round(BASE_FC * alpha)
        self.dims = (c1, c2, fc)

        # Block 1: conv -> relu -> bn, conv -> relu -> bn, pool
        self.conv1 = nn.Conv2d(3,  c1, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(c1)

        # Block 2
        self.conv3 = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(c2)
        self.conv4 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(c2)

        # Classifier: fc -> relu -> bn -> dropout -> fc
        self.fc1 = nn.Linear(c2 * 8 * 8, fc)
        self.bn5 = nn.BatchNorm1d(fc)
        self.fc2 = nn.Linear(fc, num_classes)

        # Dropout after each pool block and the FC head (reference MiniVGGNet).
        self.drop1 = nn.Dropout(DROPOUT1)
        self.drop2 = nn.Dropout(DROPOUT2)
        self.drop_fc = nn.Dropout(DROPOUT_FC)

        # He normal + zero bias + BN(gamma=1, beta=0) — same as he_initialization()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))     # CONV -> ACT -> BN
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)               # -> c1 x 16 x 16
        x = self.drop1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)               # -> c2 x 8 x 8
        x = self.drop2(x)

        x = torch.flatten(x, 1)
        x = self.bn5(F.relu(self.fc1(x)))
        x = self.drop_fc(x)
        return self.fc2(x)                      # logits; softmax is in the loss


model = MiniVGG().to(device)
c1, c2, fc = model.dims
total = sum(p.numel() for p in model.parameters())
print(f"\nMiniVGGNet  alpha={ALPHA}  widths {c1}/{c2}/{fc}  classes={NUM_CLASSES}")
for name, p in model.named_parameters():
    print(f"  {name:14s} {str(tuple(p.shape)):>22s} {p.numel():>9,}")
print(f"  {'TOTAL':14s} {'':>22s} {total:>9,}")
assert total == 324400 or ALPHA != 0.375, f"expected 324,400 params, got {total:,}"

# ------------------------------------------------------------------ data ----
# pixel/127.5 - 1  ==  ToTensor() [0,1] then Normalize(0.5, 0.5)  ==  [-1, 1]
# Training uses random horizontal flip only, matching the C build's AUGMENT
# (dataset_set_augment -> load_image is_h_flip). No random crop, to stay
# faithful. Eval/test are never augmented.
norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_tf = transforms.Compose(
    ([transforms.RandomHorizontalFlip()] if AUGMENT else []) +
    [transforms.ToTensor(), norm])
tf = transforms.Compose([transforms.ToTensor(), norm])   # eval / test

# torchvision fetches CIFAR-100 from www.cs.toronto.edu, which is heavily
# throttled from Colab (often <100 KB/s -> ~1 hour for 169 MB). Pull it from
# the Hugging Face CDN instead: same data, seconds instead of an hour.
# Falls back to torchvision if HF is unavailable.
try:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "datasets"],
                   check=True)
    from datasets import load_dataset

    raw = load_dataset("uoft-cs/cifar100")
    cols = raw["train"].column_names
    IMG = "img" if "img" in cols else "image"
    LBL = "fine_label" if "fine_label" in cols else "label"

    class HFCifar100(torch.utils.data.Dataset):
        def __init__(self, split, transform):
            self.ds, self.tf = raw[split], transform
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            r = self.ds[i]
            return self.tf(r[IMG]), r[LBL]

    # Two views of the same train split: augmented (train) and clean (eval).
    train_aug   = HFCifar100("train", train_tf)
    train_clean = HFCifar100("train", tf)
    test_set    = HFCifar100("test",  tf)
    print("data: Hugging Face CDN")

except Exception as err:
    print(f"data: HF path failed ({err}); falling back to torchvision")
    train_aug   = datasets.CIFAR100("./data", train=True,  download=True, transform=train_tf)
    train_clean = datasets.CIFAR100("./data", train=True,  download=True, transform=tf)
    test_set    = datasets.CIFAR100("./data", train=False, download=True, transform=tf)

# Same 40k/10k split the C build makes from the 50k train set. The eval subset
# is drawn from the CLEAN view so held-out images are never flipped.
n_total = len(train_aug)                          # 50,000
n_eval  = n_total * EVAL_PERCENT // 100           # 10,000
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(n_total, generator=g).tolist()
train_set = Subset(train_aug,   perm[n_eval:])    # 40,000, augmented
eval_set  = Subset(train_clean, perm[:n_eval])    # 10,000 held out, clean

# drop_last: BN uses batch statistics, so a short trailing batch would skew them.
train_ld = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)
eval_ld  = DataLoader(eval_set,  batch_size=256, shuffle=False,
                      num_workers=2, pin_memory=True)
test_ld  = DataLoader(test_set,  batch_size=256, shuffle=False,
                      num_workers=2, pin_memory=True)

print(f"\ndataset split: {len(train_set)} train / {len(eval_set)} eval "
      f"(held out) / {len(test_set)} official test")

# ------------------------------------------------------------- optimiser ----
# train.c applies weight decay to weight MATRICES only — not to biases or to BN
# gamma/beta, where it would fight the normalisation. PyTorch's SGD would decay
# everything, so split the parameters into two groups.
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim == 1:            # biases, BN gamma, BN beta
        no_decay.append(p)
    else:                      # conv and fc weight matrices
        decay.append(p)

opt = torch.optim.SGD(
    [{"params": decay,    "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, momentum=MOMENTUM)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)
crit = nn.CrossEntropyLoss()

print(f"train: lr={LR}  momentum={MOMENTUM}  wd={WEIGHT_DECAY}  "
      f"epochs={EPOCHS}  steps/epoch={len(train_ld)}\n")


@torch.no_grad()
def evaluate(loader, batch_stats=False):
    """batch_stats=True reproduces the C build, whose BN always normalises with
    batch statistics (it keeps no running mean/var). batch_stats=False is
    standard PyTorch eval using the running statistics."""
    model.train() if batch_stats else model.eval()
    correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        seen += y.size(0)
    return correct / seen


# ------------------------------------------------------------------ train ----
for e in range(EPOCHS):
    model.train()
    tot_loss = correct = seen = 0
    for x, y in train_ld:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()

        tot_loss += loss.item()
        correct  += (out.argmax(1) == y).sum().item()
        seen     += y.size(0)

    lr_now = opt.param_groups[0]["lr"]
    sched.step()

    eval_acc = evaluate(eval_ld)
    print(f"epoch {e+1}/{EPOCHS}  lr={lr_now:.6f}  "
          f"loss={tot_loss/len(train_ld):.4f}  "
          f"train acc={correct/seen:.4f}  eval acc={eval_acc:.4f}")

# ----------------------------------------------------------------- results ---
print("\n--- final ---")
print(f"held-out eval (10k, running BN stats): {evaluate(eval_ld):.4f}")
print(f"held-out eval (10k, batch BN stats)  : {evaluate(eval_ld, batch_stats=True):.4f}"
      "   <- comparable to the C build")
print(f"official CIFAR-100 test set (10k)    : {evaluate(test_ld):.4f}")


# ------------------------------------------------- export for the C build ----
def export_weights(model, path="minivgg.weights"):
    """Write the exact binary layout save_alexnet() / load_alexnet_from_file() use:
    raw fp32, no header. Conv weights are stored [in_ch*kh*kw, out_ch] in C but
    [out_ch, in_ch, kh, kw] in PyTorch, and Linear is [in, out] vs [out, in],
    so both are transposed on the way out."""
    def conv_w(m):
        oc = m.weight.shape[0]
        return m.weight.detach().reshape(oc, -1).t().contiguous().flatten()

    def lin_w(m):
        return m.weight.detach().t().contiguous().flatten()

    tensors = [
        conv_w(model.conv1), model.conv1.bias.detach(),
        conv_w(model.conv2), model.conv2.bias.detach(),
        conv_w(model.conv3), model.conv3.bias.detach(),
        conv_w(model.conv4), model.conv4.bias.detach(),
        lin_w(model.fc1),    model.fc1.bias.detach(),
        lin_w(model.fc2),    model.fc2.bias.detach(),
        model.bn1.weight.detach(), model.bn1.bias.detach(),
        model.bn2.weight.detach(), model.bn2.bias.detach(),
        model.bn3.weight.detach(), model.bn3.bias.detach(),
        model.bn4.weight.detach(), model.bn4.bias.detach(),
        model.bn5.weight.detach(), model.bn5.bias.detach(),
        # BN running stats trailer (mean then var per layer), matching the C
        # build's save_alexnet(). These drive inference-mode BN in C.
        model.bn1.running_mean.detach(), model.bn1.running_var.detach(),
        model.bn2.running_mean.detach(), model.bn2.running_var.detach(),
        model.bn3.running_mean.detach(), model.bn3.running_var.detach(),
        model.bn4.running_mean.detach(), model.bn4.running_var.detach(),
        model.bn5.running_mean.detach(), model.bn5.running_var.detach(),
    ]
    n = 0
    with open(path, "wb") as f:
        for t in tensors:
            arr = t.flatten().cpu().numpy().astype("<f4")
            f.write(arr.tobytes())
            n += arr.size
    print(f"\nwrote {path}: {n:,} floats = {n*4:,} bytes")
    return path


path = export_weights(model)
try:                       # Colab: pull the checkpoint down to your machine
    from google.colab import files
    files.download(path)
except Exception:
    pass
