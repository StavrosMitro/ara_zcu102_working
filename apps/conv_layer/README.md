# fc_layer

This app is a focused AlexNet FC-layer training/inference benchmark. It also includes per-batch cycle profiling for forward/backward/update steps when running on Ara/Spike.

## Cycle timing (why we use local cycle reads)
The training code and FC backward profiling use a local cycle counter helper (e.g. `fc_cycle_count_local()` and `alexnet_cycle_count_local()`) instead of calling `start_timer()/stop_timer()/get_timer()` from runtime.h directly. The reasons are:

- On SPIKE, runtime.h defines the timer helpers as no-ops, which would always report 0 cycles.
- `train.c` is compiled with `-O0` for SPIKE, and inlined helpers from runtime.h may not emit symbols; calling them can fail to link or be unreliable.
- Local cycle deltas avoid nested timer interference. The profiling uses deltas from CSR `cycle` so each measurement is per-batch and independent.

The local helpers still read the same CSR as runtime.h (`fence; csrr cycle`), so the measurements are consistent with the platform timing model.

## Autovectorization
The build system enables autovectorization for fc_layer by default (see apps/Makefile). This improves performance of the forward path and other vectorizable kernels.

At the same time, the SPIKE build compiles `kernel/train.c` with `-O0 -fno-vectorize -mllvm -scalable-vectorization=off` to avoid a known SPIKE corruption issue in the training update path (momentum_sgd). That file is intentionally kept scalar even when autovectorization is enabled for the rest of the app.

### Build with autovectorization (default)
From the repo root:

- `make -C apps bin/fc_layer.spike`

### Build without autovectorization
Option 1 (override from the command line):

- `make -C apps bin/fc_layer.spike AUTOVECTORIZE=0`

Option 2 (edit the target-specific setting):

- Remove or comment the `AUTOVECTORIZE = 1` assignments for fc_layer in apps/Makefile.

Note: Even with AUTOVECTORIZE=0, the dedicated SPIKE rule for `kernel/train.c` still forces `-O0 -fno-vectorize` to avoid the simulator issue in training.

## Related files
- apps/common/runtime.h
- apps/Makefile
- apps/fc_layer/kernel/train.c
- apps/fc_layer/kernel/fc_layer.c
