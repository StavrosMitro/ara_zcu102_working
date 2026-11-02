# MiniVGGNet (FP32) — CIFAR-100 on Ara / Spike

MiniVGGNet written in plain C with no ML framework: forward pass, backward pass
and SGD are all in this tree. The hot layers (conv, fc, BN, pooling, ReLU, SGD)
run on RISC-V vector instructions; every one of them also has a scalar reference
path so the two can be A/B'd against each other.

Two build paths:

| Path | Where | What it produces |
|------|-------|------------------|
| **Spike / Ara** (primary) | `apps/Makefile`, run from `apps/` | `bin/vggnet.spike`, the vectorized RV64GCV build |
| Native reference | [Makefile](Makefile) in this directory | `minivgg_train` etc., scalar gcc build for comparison |

The FP16 twin of this app lives in [../vggnet16/](../vggnet16/).

---

## Architecture

Constants in [kernel/alexnet.h](kernel/alexnet.h), forward pass in
[main.c:204-378](main.c#L204).

```
INPUT     3 x 32 x 32
  conv1  C1 x 32 x 32   3x3 s1 p1
  relu1
  bn1                   spatial BN over C1 channels
  conv2  C1 x 32 x 32   3x3 s1 p1
  relu2
  bn2
  pool1  C1 x 16 x 16   2x2 s2
  dropout(DROPOUT1)                       [training only]
  conv3  C2 x 16 x 16   3x3 s1 p1
  relu3
  bn3
  conv4  C2 x 16 x 16   3x3 s1 p1
  relu4
  bn4
  pool2  C2 x  8 x  8   2x2 s2
  dropout(DROPOUT2)                       [training only]
  flatten C2*8*8                          (no-op: pool2 is already contiguous)
  fc1    FC1_LAYER
  relu5
  bn5                   1-D BN over FC1_LAYER units
  dropout(DROPOUT_FC)                     [training only]
  fc2    OUT_LAYER      raw logits
```

Notes that are easy to get wrong:

- The order inside a block is **CONV → ACT → BN**, not conv→bn→act.
- Pooling happens after every *second* conv, so there are 4 convs and 2 pools.
- There is exactly **one hidden FC layer**. `fc2` emits logits; softmax is fused
  into the cross-entropy loss in [kernel/train.c](kernel/train.c).
- Every conv is 3×3 / stride 1 / pad 1 and every pool is 2×2 / stride 2, so the
  spatial sizes above are fixed by construction.

### Width variants

A uniform multiplier over the reference 32/32/64/64/512 configuration:

| VARIANT | C1 | C2 | FC1_LAYER | params (OUT_LAYER=100) |
|---------|----|----|-----------|------------------------|
| `alpha` (default) | 12 | 24 | 192 | 324,400 |
| `full` | 32 | 64 | 512 | 2,215,940 |

Widths come in as `-DC1_CHANNELS/-DC2_CHANNELS/-DFC1_LAYER`; class count as
`-DOUT_LAYER=10|100`. `FC1_IN_UNITS` is derived (`C2 * 8 * 8`).

---

## Building and running on Spike

Run these from `apps/` (not from this directory):

```bash
# build + run, default knobs: alpha, batch 32, 2 steps, fine-tune mode
make spike-run-vggnet

# build only
make bin/vggnet.spike

# full training run (random init, every layer trainable) on the full net
make spike-run-vggnet VGGNET_FINETUNE=0 VGGNET_VARIANT=full VGGNET_BATCHSIZE=8
```

`spike-run-vggnet` tees its output to `apps/spike_runs/spike-run-vggnet`.

### Knobs

All are `make` variables on the `apps/Makefile` command line.

| Variable | Default | Meaning |
|----------|---------|---------|
| `VGGNET_VARIANT` | `alpha` | `alpha` or `full` (see table above) |
| `VGGNET_BATCHSIZE` | 32 | Mini-batch; also sets `ALEXNET_STATIC_MAX_BATCH` |
| `VGGNET_EPOCHS` | 1 | Epochs |
| `VGGNET_MAX_STEPS` | 2 | Cap on steps/epoch so a Spike run finishes; 0 = full epoch |
| `VGGNET_FINETUNE` | 1 | 1 = freeze conv stack, adapt `fc1/bn5/fc2` from the embedded weights. 0 = train everything from He init |
| `VGGNET_DROP1` / `DROP2` / `DROPFC` | 0.25 / 0.25 / 0.5 | Dropout after pool1 / pool2 / bn5. 0 disables |
| `VGGNET_AUGMENT` | 1 | Random horizontal flip on training batches |
| `VGGNET_INIT_GAIN` | — | Scales the He std-dev (`WEIGHT_INIT_GAIN`) |

Bisection / diagnostic knobs:

| Variable | Effect |
|----------|--------|
| `VGGNET_SCALAR_CONV=1` | `-DUSE_IM2COL_CONV` — conv forward via the scalar im2col path |
| `VGGNET_SCALAR_FC=1` | `-DUSE_SCALAR_FC` — fc forward via the scalar path |
| `VGGNET_SGD_SCALAR=1` | `-DSGD_SCALAR` — scalar reference `sgd_step` instead of the vectorized one |
| `VGGNET_DUMP_TRACE=1` | One-shot per-layer checksum of the first training forward pass |
| `VGGNET_DUMP_CONV1=1` | One-shot dump of conv1's output for image 0 |

Use the scalar switches to isolate whether a divergence comes from the vector
conv (`fconv3d`) or the vector fc (`fmatmul`). The trace/dump code is guarded by
plain `-DDUMP_TRACE` / `-DDUMP_CONV1` ([main.c:232](main.c#L232)), so the same
output can be obtained from the native reference build by adding those defines.

### Fine-tune mode on Spike

Spike is bare-metal with no filesystem, so `save_alexnet()` /
`load_alexnet_from_file()` are no-ops there. Under `FINETUNE_MODE + SPIKE`
([main.c:932-946](main.c#L932)) the network binds directly to the arrays in
[kernel/weights.c](kernel/weights.c) — those *are* the pretrained checkpoint —
freezes conv1-4 and bn1-4, and trains only `fc1`, `bn5`, `fc2` in place. No
random re-init happens.

---

## Native reference build

From this directory:

```bash
make                          # train, alpha variant, CIFAR-100
make VARIANT=full             # the 32/64/512 net
make run                      # build + run training
make finetune                 # freeze the conv stack, adapt the head
make infer IDX=0              # single-image inference
make asan                     # build + run under AddressSanitizer + UBSan
make gdb                      # build and drop into gdb
make clean
```

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCHSIZE` | 64 | Mini-batch size |
| `EPOCHS` | 30 | Training epochs |
| `DATASET` | `cifar100` | `cifar10` or `cifar100` (selects the `.S` blob and `OUT_LAYER`) |
| `VARIANT` | `alpha` | `alpha` or `full` |
| `FINETUNE` | 0 | 1 → load `minivgg.weights`, freeze the conv stack |
| `LR_STEP` | 12 | LR ×0.1 every this many epochs |
| `AUGMENT` | 1 | Random horizontal flip |
| `DROP1` / `DROP2` / `DROPFC` | 0.25 / 0.25 / 0.5 | Dropout probabilities |
| `WD` | 0.0005 | L2 weight decay, applied to weight matrices only |
| `IDX` | 0 | Image index for `make infer` |
| `OPT` | `-O3 -march=native -funroll-loops` | Override (e.g. `OPT=-O2`) to un-pin the binary from this CPU |

Binaries: `minivgg_train`, `minivgg_infer`, `minivgg_debug` (the debug one is
`-O0 -g3 -fsanitize=address,undefined`). The checkpoint written by
`save_alexnet()` is `minivgg.weights`.

Use `make -B` to force a rebuild when only Makefile variables changed.

> `make infer` runs BN with a batch of 1, which has zero per-channel variance,
> so single-image inference is only meaningful once BN has running statistics
> loaded. See the note at [main.c:1009](main.c#L1009).

---

## Dataset

The network reads three blobs from `kernel/`, embedded into the binary by an
`.incbin` in a `.S` file:

| Dataset | `.S` file | Blobs |
|---------|-----------|-------|
| CIFAR-100 / CIFAR-100-C | [data_cifar100.S](data_cifar100.S) | `kernel/cifar100_{data,labels,offsets}.bin` |
| CIFAR-10 | `data_cifar10.S.excluded` | `kernel/cifar10_{data,labels,offsets}.bin` |

Both files export the same symbols (`cifar10_data`, `cifar10_offsets`,
`cifar10_labels`), so only one may take part in a build. The Spike build globs
`*.S`, which is why the CIFAR-10 one is parked under a `.excluded` suffix and
`data.S` is left empty; restore it into `data.S` to build the CIFAR-10 side.

Sizes are declared in [kernel/cifar10_dataset.h](kernel/cifar10_dataset.h).
**The currently embedded set is a 64-image subset** for fast Spike bring-up; the
full 50,000-image blobs are backed up as `kernel/cifar100_*.bin.full`.

### Preparing data

Raw datasets are not kept in this tree. [scripts/dataset_root.py](scripts/dataset_root.py)
resolves a raw input path against `$VGGNET_DATA_ROOT`, falling back to
`~/vggnet`, so the prep scripts work without copying multi-GB files in.

```bash
# CIFAR-100 (writes kernel/*.bin and updates kernel/cifar10_dataset.h)
python3 scripts/prepare_cifar100.py cifar-100-binary/test.bin    # 10k images
python3 scripts/prepare_cifar100.py cifar-100-binary/train.bin   # 50k images

# CIFAR-100-C, for test-time adaptation — all 19 corruptions at severity 2
python3 scripts/prepare_cifar100c.py CIFAR-100-C/ --all --severity 2

# a single corruption / severity
python3 scripts/prepare_cifar100c.py CIFAR-100-C/ gaussian_noise --severity 3
```

Both scripts convert CHW→HWC (interleaved RGB, which is what
[kernel/data.c](kernel/data.c) expects), write int32 labels and uint32 offsets,
and rewrite `cifar10_count` / `cifar10_total_bytes` in `cifar10_dataset.h`.

### Train / eval split

`EVAL_PERCENT` (default 20, [kernel/train.c:83](kernel/train.c#L83)) of the
dataset is held out and never trained on. The split is created once, before the
baseline accuracy pass, and printed as `dataset split: N train / M eval`.
Augmentation is applied to training batches only. The partial trailing batch is
dropped, because BN normalises with batch statistics.

---

## Weights

`kernel/weights.c` holds the parameter arrays; `kernel/weights.h` declares their
sizes in terms of `C1_CHANNELS` / `C2_CHANNELS` / `FC1_LAYER` / `OUT_LAYER`:
conv1-4 (weights + bias), fc1-2 (weights + bias), bn1-5 (gamma + beta).
`verify_weight_array_shapes()` in [main.c:732](main.c#L732) checks every array
size at startup, which catches a `weights.c` built for a different variant.

On Spike, `weights.c` is the only way a pretrained checkpoint gets in.

### Checkpoint format

`save_alexnet()` writes raw fp32 with no header, in this order:

```
conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b,
fc1_w,   fc1_b,   fc2_w,   fc2_b,
bn1_gamma, bn1_beta, ... bn5_gamma, bn5_beta,
bn1_run_mean, bn1_run_var, ... bn5_run_mean, bn5_run_var    ← trailer
```

The BN running-stats trailer is what inference-mode BN uses.
`load_alexnet_from_file()` validates the file size against the build's parameter
count, so an architecture/checkpoint mismatch is caught rather than silently
misread; a file without the trailer is accepted with a warning.

### Embedding a checkpoint

```bash
python3 scripts/binary2text_weights.py minivgg.weights alpha 100
#                                      ^file            ^variant ^classes
```

This regenerates `kernel/weights.c`. The running mean/var trailer is *not*
emitted — it lives in [kernel/batchnorm_layer.c](kernel/batchnorm_layer.c) and
only matters for inference-mode BN, while the Spike fine-tune runs with batch
statistics.

> `scripts/gen_zero_weights.py` targets an older 5-conv / 3-FC network
> (`conv1b…conv5b`, `fc3`, `bn1…bn7`) and does not match `kernel/weights.h`.
> Use `binary2text_weights.py` to produce `weights.c`.

---

## Training details

Set in [kernel/train.c](kernel/train.c); all are `#ifndef`-guarded so they can be
overridden from the build.

| | Value |
|---|---|
| Init | He/Kaiming normal, `std = WEIGHT_INIT_GAIN * sqrt(2/fan_in)` ([main.c:644](main.c#L644)) |
| Optimizer | SGD with momentum 0.9 |
| LR | 0.01 from scratch, 1e-3 in fine-tune mode |
| LR schedule | ×0.1 every `LR_STEP_EPOCHS` epochs |
| Weight decay | 5e-4, on weight matrices only — never on biases or BN gamma/beta |
| Loss | Cross-entropy with softmax fused in; gradient is the batch mean |
| Seed | `RANDOM_SEED` = 1 |

---

## Verification

[scripts/check_vggnet.sh](scripts/check_vggnet.sh) builds for Spike, runs, and
compares exact per-layer gradient checksums, post-update weight checksums and
the max-pool element-wise audit against a recorded baseline — end-to-end loss is
too chaotic to be a useful oracle past epoch 1. Exit status gates a commit.

```bash
scripts/check_vggnet.sh record [MAKEVAR=VAL ...]   # save baseline
scripts/check_vggnet.sh check  [MAKEVAR=VAL ...]   # diff against baseline
scripts/check_vggnet.sh ab VAR=VAL [MAKEVAR=...]   # run with VAR set vs unset
scripts/check_vggnet.sh run    [MAKEVAR=VAL ...]   # just build + run
```

`ab` is the one to reach for when proving a refactor is a no-op.

---

## File map

| File | Purpose |
|------|---------|
| [main.c](main.c) | Forward pass, setup, weight init/save/load, `main()` |
| [kernel/alexnet.h](kernel/alexnet.h) | Architecture constants and the `alexnet` struct |
| [kernel/train.c](kernel/train.c) | Backward pass, loss, SGD, training loop, eval sweep |
| [kernel/convolution_layer.c](kernel/convolution_layer.c) | Conv forward/backward (vector `fconv3d` + scalar im2col/col2img) |
| [kernel/fc_layer.c](kernel/fc_layer.c) | FC forward/backward, `fmatmul` blocking |
| [kernel/batchnorm_layer.c](kernel/batchnorm_layer.c) | BN forward/backward, running stats |
| [kernel/maxpooling_layer.c](kernel/maxpooling_layer.c) | Max pool forward/backward with saved argmax indices |
| [kernel/activation_layer.c](kernel/activation_layer.c) | ReLU, softmax |
| [kernel/dropout_layer.c](kernel/dropout_layer.c) | Inverted dropout forward/backward |
| [kernel/fconv3c_3x3x3_f32.c](kernel/fconv3c_3x3x3_f32.c) | `fconv3d_CHx3x3_f32` — the vector 3×3 conv |
| [kernel/fmatmul.c](kernel/fmatmul.c) | Vector matmul family: `fmatmul_32`, `_fused_32`, `_nt_32`, `_tn_32`, 4×4 tiled variants |
| [kernel/matrix_vec.c](kernel/matrix_vec.c) | `matrix_multiply_*_32` wrappers over `fmatmul` |
| [kernel/matrix.c](kernel/matrix.c) | Scalar reference matmul |
| [kernel/data.c](kernel/data.c) | Batch loading, HWC decode, split, horizontal flip |
| [kernel/weights.c](kernel/weights.c) / [.h](kernel/weights.h) | Parameter arrays (regenerated by `binary2text_weights.py`) |
| [kernel/rng_spike.c](kernel/rng_spike.c) | `rand()` for the bare-metal build |
| [data_cifar100.S](data_cifar100.S) | `.incbin` of the CIFAR-100 blobs |
| [scripts/](scripts/) | Data prep, weight conversion, PyTorch reference, verification |
