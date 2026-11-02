# fc_layer_test — FP16 (mixed precision) vs FP32 equivalence suite

Compares the **mixed-precision** FC-layer kernels (FP16 inputs/weights, FP32
accumulate — from `apps/fc_layer`) against the **pure FP32** reference
(from `apps/fc_layer32`), kernel by kernel and then end-to-end, on identical
inputs taken from `data.S`.

## How it is wired

A single Spike binary links the three kernels under test **from both trees**:

```
fc_layer_test/
├── main.c            # the test driver (also carries copied loss + optimizer)
├── data.S            # dataset (.incbin of generated_data/*.bin), FP32
├── generated_data/   # inputs.bin / targets.bin / labels.bin
├── fp16/             # snapshot of apps/fc_layer/kernel  : fmatmul.c matrix.c fc_layer.c (+headers)
└── fp32/             # snapshot of apps/fc_layer32/kernel: same 3 modules, every external symbol _32
```

To let both trees coexist in one binary, **every externally-linked function and
global of the FP32 tree was suffixed with `_32`** (done in `apps/fc_layer32`
itself, so that tree still builds standalone). Compile-time-only names
(struct tags, header guards, macros) and `event_trigger` (a linker-script symbol)
are intentionally left unchanged because they never collide at link time.

The driver includes the FP16 headers directly and hand-declares the `_32`
interfaces (`fc_op_32`, `matrix_multiply_32`, `fc_op_forward_32`, …) to avoid
header-guard / struct-tag clashes.

Per project decision, the **loss and optimizer are copied** into `main.c`
(`harness_*`), because they are `static` inside each `train.c` and cannot be
linked. They are byte-for-byte copies of the real RVV implementations, so they
run the actual vector asm.

## What is tested (`BATCH=2`, `IN=2048`, `OUT=512`)

Part 1 — one by one:
- `matrix_multiply`, `_fused`, `_nt`, `_nt_deferred`, `_tn`
- `fc_op_forward` (full layer)
- `fc_op_backward_full_profile` → `d_input`, `d_weights`, `d_bias`
- `mse_loss_vec`, `cross_entropy_loss` (run on each path's forward output)
- momentum SGD update: FP16-weight (`vfwcvt`/`vfncvt`) vs FP32-weight

Part 2 — in flow: forward → MSE → backward → weight update, for several steps,
reporting how far the FP16 and FP32 weights diverge.

> `BATCH` must be 2: `calc_bias_gradient_vec_batch2` is hardcoded to two rows.

Each line prints `max_abs`, `max_rel`, `mean_abs`. `PASS` means `max_rel ≤ REL_TOL`
(3e-2). `CHECK` is not necessarily a bug — FP16 rounding can legitimately exceed
the tolerance on ill-conditioned (cancelling) reductions; inspect the numbers.

## Build / run (Spike)

From `apps/`:

```
make bin/fc_layer_test.spike
```

The Makefile stanza gives this app `AUTOVECTORIZE=0`, points the assembler at
`fc_layer_test/generated_data` for `data.S`, and compiles `main.c` with
`-O0 -fno-vectorize` (it carries the hand-copied vector `momentum_sgd` asm,
mirroring the `train.c` workaround). Run the resulting binary with your usual
Spike command; `bin/fc_layer_test.spike.dump` is the disassembly.

## Refreshing the snapshots

`fp16/` and `fp32/` are **copies**. If you change `apps/fc_layer` (mixed) or
`apps/fc_layer32`, re-copy the three modules (`fmatmul.c matrix.c fc_layer.c`
plus headers) into the matching subdir.
