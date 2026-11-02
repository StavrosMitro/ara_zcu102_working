# MiniVGGNet (FP16) — CIFAR-100 on Ara / Spike

The pure-FP16 twin of [../vggnet/](../vggnet/). Same MiniVGGNet architecture,
same training loop, same data — but every activation, weight, gradient and
momentum buffer is `_Float16`, and the vector kernels are the `zvfh` half-precision
ones. **There are no FP32 master weights**: momentum is the sole accumulator.

Built for `rv64gcv_zfh_zvfh`. If you want the FP32 reference numbers to compare
against, build [../vggnet/](../vggnet/) with the same knobs.

---

## Architecture

Identical to the FP32 app. Constants in [kernel/alexnet.h](kernel/alexnet.h),
forward pass in [main.c](main.c).

```
INPUT     3 x 32 x 32
  conv1  C1 x 32 x 32   3x3 s1 p1   →  relu1  →  bn1
  conv2  C1 x 32 x 32   3x3 s1 p1   →  relu2  →  bn2
  pool1  C1 x 16 x 16   2x2 s2      →  dropout(DROPOUT1)
  conv3  C2 x 16 x 16   3x3 s1 p1   →  relu3  →  bn3
  conv4  C2 x 16 x 16   3x3 s1 p1   →  relu4  →  bn4
  pool2  C2 x  8 x  8   2x2 s2      →  dropout(DROPOUT2)
  flatten C2*8*8
  fc1    FC1_LAYER                  →  relu5  →  bn5  →  dropout(DROPOUT_FC)
  fc2    OUT_LAYER                  raw logits
```

The order inside a block is **CONV → ACT → BN**. Pooling comes after every
*second* conv, so there are 4 convs and 2 pools, and there is exactly one hidden
FC layer. Softmax is fused into the cross-entropy loss. Dropout is training-only.

| VARIANT | C1 | C2 | FC1_LAYER | params (OUT_LAYER=100) |
|---------|----|----|-----------|------------------------|
| `alpha` (default) | 12 | 24 | 192 | 324,400 |
| `full` | 32 | 64 | 512 | 2,215,940 |

---

## What differs from the FP32 app

### Types

`_Float16` throughout: [kernel/weights.h](kernel/weights.h) (all conv/fc/bn
parameters), the `alexnet` struct's `input`/`output`, and every layer op struct
in `kernel/*.h`, including BN's `gamma`/`beta`/`x_norm`/`avg`/`var` and the
running statistics.

### FP32 islands

Only two places compute in `float`, both for precision reasons:

- **Softmax** ([kernel/activation_layer.c:96](kernel/activation_layer.c#L96)) —
  logits arrive in FP16; max / exp / sum / divide run in FP32 and only the final
  probability is narrowed back.
- **The cross-entropy gradient seed** — built in the `ce_grad_f32` scratch, then
  narrowed to FP16 with a vector `vfncvt.f.f.w`
  ([kernel/train.c:247](kernel/train.c#L247)).

### Dynamic loss scaling

Applied once for the whole deep backward pass
([kernel/train.c:225-292](kernel/train.c#L225)), using the algorithm from
`fc_layer16only`:

1. `CLEAR_FFLAGS` before the loss. The gradient seed multiply and every backward
   matmul then accumulate overflow/underflow into `fflags`.
2. **Overflow** → halve the scale and **skip the optimizer for that step** (the
   gradients are inf).
3. **Clean or underflow** → the optimizer unscales each gradient by `1/scale`.
   The scale doubles after `UF_DEBOUNCE_STEPS` (8) consecutive underflow steps,
   or after 1000 consecutive clean steps.

| Constant | Value |
|----------|-------|
| Initial scale | 1024 |
| `MIN_LOSS_SCALE` | 1 |
| `MAX_LOSS_SCALE` | 32768 (2¹⁵, headroom below fp16 max 65504) |
| `UF_DEBOUNCE_STEPS` | 8 |

The rounding mode is pinned to round-to-nearest-even (`SET_FRM_RNE()`).

### Optimizer

FP16 momentum SGD with no master copy
([kernel/train.c:376-410](kernel/train.c#L376)): unscale by `1/loss_scale`, add
`wd*w`, then `v = MOMENTUM*v - lr*t`, clip to `[-1, 1]`, `w += v`. Momentum is
what carries sub-epsilon updates that would otherwise vanish when added directly
to an FP16 weight.

### Vector kernels

The FP32 app keeps its whole matmul family in one `fmatmul.c`; here each form is
its own file, all FP16:

| File | Entry point | Used by |
|------|-------------|---------|
| [kernel/fmatmul.c](kernel/fmatmul.c) | `fmatmul_32` | general C = A·B |
| [kernel/fmatmul_fused.c](kernel/fmatmul_fused.c) | `fmatmul_fused_32` | FC forward (bias fused into the accumulator) |
| [kernel/fmatmul_nt.c](kernel/fmatmul_nt.c) | `fmatmul_nt_32` | C = A·Bᵀ |
| [kernel/fmatmul_tn.c](kernel/fmatmul_tn.c) | `fmatmul_tn_32` | C = Aᵀ·B (weight gradients) |
| [kernel/fmatmul_deferred.c](kernel/fmatmul_deferred.c) | deferred-accumulate variant | |
| [kernel/fconv3c_3x3x3_f32.c](kernel/fconv3c_3x3x3_f32.c) | `fconv3d_CHx3x3_f16` | conv forward and the dgrad path |

`fconv3d_CHx3x3_f16` expects a **pre-padded** input (rows `out_w+2` wide); the
padding scratch is set up in [kernel/convolution_layer.c](kernel/convolution_layer.c).

---

## Building and running

Run from `apps/` (not from this directory):

```bash
# build + run, default knobs: alpha, batch 32, 2 steps, fine-tune mode
make spike-run-vggnet16

# full training run (random init, every layer trainable)
make spike-run-vggnet16 VGGNET16_FINETUNE=0 VGGNET16_BATCHSIZE=4 VGGNET16_EPOCHS=4

# build only
make bin/vggnet16.spike
```

Output is teed to `apps/spike_runs/spike-run-vggnet16`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `VGGNET16_VARIANT` | `alpha` | `alpha` or `full` |
| `VGGNET16_BATCHSIZE` | 32 | Mini-batch; also sets `ALEXNET_STATIC_MAX_BATCH` |
| `VGGNET16_EPOCHS` | 1 | Epochs |
| `VGGNET16_MAX_STEPS` | 2 | Cap on steps/epoch; 0 = full epoch |
| `VGGNET16_FINETUNE` | 1 | 1 = freeze conv stack, adapt `fc1/bn5/fc2` from the embedded weights. 0 = train everything from He init |
| `VGGNET16_DROP1` / `DROP2` / `DROPFC` | 0.25 / 0.25 / 0.5 | Dropout after pool1 / pool2 / bn5 |
| `VGGNET16_AUGMENT` | 1 | Random horizontal flip on training batches |
| `VGGNET16_INIT_GAIN` | — | Scales the He std-dev (`WEIGHT_INIT_GAIN`) |
| `VGGNET16_INPUT_SCALE` | — | Multiplies the FP16 input batch. Experiment knob only: bn1 normalises conv1's output, so a global input scale is neutralised past the first BN |
| `VGGNET16_SGD_SCALAR` | 0 | 1 → `-DSGD_SCALAR`, the scalar reference optimizer |
| `VGGNET16_SELFTEST` | 0 | 1 → `-DCONV_SELFTEST`: run the isolated FP16-vs-FP32 conv unit test and exit |
| `VGGNET16_DUMP_TRACE` | 0 | 1 → one-shot per-layer checksum of the first training forward pass |

`VGGNET16_SELFTEST=1` runs `conv_selftest()`
([kernel/convolution_layer.c:1027](kernel/convolution_layer.c#L1027)) — a small
3→4 channel, 8×8 conv with deterministic inputs, checked forward and backward
against the scalar reference — then stops before any training.

Note there is no `SCALAR_CONV` / `SCALAR_FC` switch here; use `VGGNET16_DUMP_TRACE`
and diff against the FP32 app's trace instead.

This directory also carries the native gcc [Makefile](Makefile) inherited from the
FP32 app — same targets and variables, documented in the
[FP32 README](../vggnet/README.md#native-reference-build). The Spike path above is
how this app is built and run.

### Fine-tune mode on Spike

Spike has no filesystem, so checkpoint save/load are no-ops. Under
`FINETUNE_MODE + SPIKE` the network binds directly to the arrays in
[kernel/weights.c](kernel/weights.c) — those *are* the pretrained checkpoint —
freezes conv1-4 and bn1-4, and trains only `fc1`, `bn5`, `fc2` in place.

---

## Weights

`kernel/weights.c` holds `_Float16` arrays with the same names and sizes as the
FP32 app. Generate it from an FP32 checkpoint:

```bash
python3 scripts/binary2text_weights_fp16.py minivgg.weights alpha 100
#                                           ^fp32 file      ^variant ^classes
```

Every value is rounded to IEEE-754 binary16 via `struct 'e'`, so the emitted
decimals are exactly what the compiler will store. Values outside fp16's ±65504
range are clamped and reported.

The input checkpoint layout is the one `save_alexnet()` writes on the FP32 side:

```
conv1_w, conv1_b, … conv4_w, conv4_b, fc1_w, fc1_b, fc2_w, fc2_b,
bn1_gamma, bn1_beta, … bn5_gamma, bn5_beta,
bn1_run_mean, bn1_run_var, … bn5_run_mean, bn5_run_var    ← trailer, not emitted
```

`scripts/binary2text_weights.py` (the FP32 emitter) is also present, for when you
want to feed the same checkpoint to [../vggnet/](../vggnet/).

---

## Dataset

`kernel/cifar*_{data,labels,offsets}.bin` are **symlinks into
`../../vggnet/kernel/`** — the two apps share one copy of the prepared data.
Re-running a prep script in `vggnet/` therefore changes what this app trains on
too.

Sizes are declared in [kernel/cifar10_dataset.h](kernel/cifar10_dataset.h) (this
one is a real file, not a symlink). The currently embedded set is a 64-image
subset for fast Spike bring-up. See the [FP32 README](../vggnet/README.md) for
the prep scripts, `$VGGNET_DATA_ROOT`, and the CIFAR-100-C flow.

`data_cifar100.S` carries the `.incbin`. `data.S` is empty and the CIFAR-10
version is parked as `data_cifar10.S.excluded`, because both export the same
symbols and the Spike build globs `*.S`.

`EVAL_PERCENT` (default 20) of the dataset is held out and never trained on;
augmentation applies to training batches only; the partial trailing batch is
dropped because BN normalises with batch statistics.

---

## Training details

Same schedule as the FP32 app, on top of the FP16 machinery above:

| | Value |
|---|---|
| Init | He/Kaiming normal, `std = WEIGHT_INIT_GAIN * sqrt(2/fan_in)` |
| Optimizer | FP16 momentum SGD, momentum 0.9, update clipped to `[-1, 1]` |
| LR | 0.01 from scratch, 1e-3 in fine-tune mode |
| LR schedule | ×0.1 every `LR_STEP_EPOCHS` epochs |
| Weight decay | 5e-4, on weight matrices only — never on biases or BN gamma/beta |
| Rounding | RNE, pinned via `SET_FRM_RNE()` |
| Seed | `RANDOM_SEED` = 1 |

---

## File map

| File | Purpose |
|------|---------|
| [main.c](main.c) | Forward pass, setup, weight init, `main()` |
| [kernel/alexnet.h](kernel/alexnet.h) | Architecture constants and the `alexnet` struct |
| [kernel/train.c](kernel/train.c) | Backward pass, loss, loss scaling, FP16 SGD, training loop, eval |
| [kernel/convolution_layer.c](kernel/convolution_layer.c) | Conv forward/backward + `conv_selftest()` |
| [kernel/fc_layer.c](kernel/fc_layer.c) | FC forward/backward, `fmatmul` blocking |
| [kernel/batchnorm_layer.c](kernel/batchnorm_layer.c) | BN forward/backward, running stats |
| [kernel/maxpooling_layer.c](kernel/maxpooling_layer.c) | Max pool forward/backward with saved argmax indices |
| [kernel/activation_layer.c](kernel/activation_layer.c) | ReLU, softmax (FP32 island) |
| [kernel/dropout_layer.c](kernel/dropout_layer.c) | Inverted dropout forward/backward |
| [kernel/fmatmul*.c](kernel/) | FP16 vector matmul family (see the table above) |
| [kernel/fconv3c_3x3x3_f32.c](kernel/fconv3c_3x3x3_f32.c) | `fconv3d_CHx3x3_f16` — the vector 3×3 conv |
| [kernel/matrix.c](kernel/matrix.c) | Scalar reference matmul + `matrix_multiply_*_32` wrappers |
| [kernel/data.c](kernel/data.c) | Batch loading, HWC decode, split, horizontal flip |
| [kernel/weights.c](kernel/weights.c) / [.h](kernel/weights.h) | `_Float16` parameter arrays |
| [scripts/binary2text_weights_fp16.py](scripts/binary2text_weights_fp16.py) | FP32 checkpoint → FP16 `weights.c` |
