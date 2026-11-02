# Copyright 2019 ETH Zurich and University of Bologna.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Matheus Cavalcante, ETH Zurich
#         Samuel Riedel, ETH Zurich

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
APPS_DIR := $(ROOT_DIR)
COMMON_DIR := $(ROOT_DIR)/common
TESTS_DIR := $(ROOT_DIR)/riscv-tests/isa

# This will overwrite the ROOT_DIR variable from the included makefile
include $(COMMON_DIR)/runtime.mk
include $(COMMON_DIR)/riscv_tests.mk
include $(COMMON_DIR)/default_args.mk

# Per-app source finder: skip app-local virtualenv and git metadata.
find_app_sources = $(shell find -L $(1) \( -path "*/.venv" -o -path "*/.git" -o -path "*/sshfs_dir" \) -prune -o \( -name "*.c" -o -name "*.cpp" -o -name "*.S" \) -print)

APPS := $(patsubst $(APPS_DIR)/%/main.c,%,$(shell find $(APPS_DIR) -name "sshfs_dir" -prune -o -name "main.c" -print))
APPS += $(patsubst $(APPS_DIR)/%/main.cpp,%,$(shell find $(APPS_DIR) -name "sshfs_dir" -prune -o -name "main.cpp" -print))
# Don't automatically compile the benchmark app, which is only a wrapper around the other apps
BINARIES := $(filter-out bin/benchmarks bin/rivec-bmarks% bin/_%, $(addprefix bin/,$(APPS)))

CVA6_EXTENSIONS := rv64ui rv64uc rv64um rv64uf rv64ud rv64si
# Atomics are messy, since there is currently no memory region capable of handling them
# CVA6_EXTENSIONS := rv64ua
CVA6_BINARIES := $(addprefix bin/, $(cva6_tests))
ARA_EXTENSIONS := rv64uv
ARA_BINARIES := $(addprefix bin/, $(ara_tests))

# Suffix for binaries
ifeq ($(LINUX),1)
BIN_SUFFIX := -linux
else
BIN_SUFFIX :=
endif

# FFT requires special treatment because of its header files
ifeq ($(ENV_DEFINES),)
bin/fft:  ENV_DEFINES += -DFFT_SAMPLES=$(subst ",,$(firstword $(def_args_fft)))
bin/fft%: ENV_DEFINES += -DFFT_SAMPLES=$(subst ",,$(firstword $(def_args_fft)))
endif

# Optional training step cap for AlexNet smoke runs.
ALEXNET_MAX_STEPS ?= 0
# Optional operator timing prints for long Spike runs.
ALEXNET_SHOW_OP_TIME ?= 0

ifeq ($(ALEXNET_SHOW_OP_TIME),1)
ALEXNET_TIMING_DEFINE := -DSHOW_OP_TIME
else
ALEXNET_TIMING_DEFINE :=
endif

# # AlexNet default profile: training, batchsize=25, static batch=25, epochs=1
# bin/alexnet:  ENV_DEFINES += -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=1 -DALEXNET_STATIC_MAX_BATCH=1 -DALEXNET_EPOCHS=1 -DALEXNET_MAX_STEPS=$(ALEXNET_MAX_STEPS) $(ALEXNET_TIMING_DEFINE)
# bin/alexnet%: ENV_DEFINES += -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=1 -DALEXNET_STATIC_MAX_BATCH=1 -DALEXNET_EPOCHS=1 -DALEXNET_MAX_STEPS=$(ALEXNET_MAX_STEPS) $(ALEXNET_TIMING_DEFINE)

# fc_layer profile controls (compatible with apps/fc_layer/gccMakefile semantics)
FC_LAYER_MODE ?= train
FC_LAYER_LOSS ?= MSE
FC_LAYER_BATCHSIZE ?= 12
FC_LAYER_EPOCHS ?= 1
FC_LAYER_INFER_IDX ?= -1
FC_LAYER_STATIC_MAX_BATCH ?= $(FC_LAYER_BATCHSIZE)
FC_LAYER_SAMPLES ?= $(FC_LAYER_BATCHSIZE)

ifeq ($(FC_LAYER_MODE),train)
FC_LAYER_MODE_DEFINES := -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=$(FC_LAYER_BATCHSIZE) -DALEXNET_STATIC_MAX_BATCH=$(FC_LAYER_STATIC_MAX_BATCH) -DALEXNET_EPOCHS=$(FC_LAYER_EPOCHS) -DFC_TOTAL_SAMPLES=$(FC_LAYER_SAMPLES)
else ifeq ($(FC_LAYER_MODE),inference)
FC_LAYER_MODE_DEFINES := -DALEXNET_MODE_INFERENCE -DALEXNET_INFER_IDX=$(FC_LAYER_INFER_IDX)
else
$(error Unsupported FC_LAYER_MODE='$(FC_LAYER_MODE)'. Use train or inference)
endif

ifeq ($(FC_LAYER_LOSS),MSE)
FC_LAYER_LOSS_DEFINES := -DSET_MSE
else ifeq ($(FC_LAYER_LOSS),CEL)
FC_LAYER_LOSS_DEFINES := -DSET_CEL
else
$(error Unsupported FC_LAYER_LOSS='$(FC_LAYER_LOSS)'. Use MSE or CEL)
endif

FC_LAYER_DEFINES := $(FC_LAYER_MODE_DEFINES) $(FC_LAYER_LOSS_DEFINES)

# fc_layer needs: (1) compile-time train/loss knobs, (2) autovectorization off,
# (3) assembler include root for data.S .incbin paths.
# NOTE: use bin/fc_layer.% (dot separator) to avoid matching bin/fc_layer16only, bin/fc_layer32, etc.
bin/fc_layer:   $(APPS_DIR)/Makefile
bin/fc_layer.%: $(APPS_DIR)/Makefile
bin/fc_layer:   ENV_DEFINES += $(FC_LAYER_DEFINES)
bin/fc_layer.%: ENV_DEFINES += $(FC_LAYER_DEFINES)
bin/fc_layer:   AUTOVECTORIZE = 0
bin/fc_layer.%: AUTOVECTORIZE = 0
bin/fc_layer:   RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer
bin/fc_layer.%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer
bin/fc_layer:   RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer
bin/fc_layer.%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer
bin/fc_layer:   RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer
bin/fc_layer.%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer

# Workaround: keep app-level autovectorization on, but compile training update
# code without vectorization to avoid Spike runtime corruption in momentum_sgd.
fc_layer/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# ---------------------------------------------------------------------------
# FC BENCHMARK PARITY KNOBS
#
# fc_layer32 (FP32) and fc_layer16only (FP16) are a precision A/B pair: for the
# comparison to mean anything they must run the same shape, the same number of
# steps, and the same data. Both apps derive their per-app knobs from these, so
# a single override moves both:
#
#   make spike-run-fc_layer32      FC_BENCH_BATCHSIZE=8
#   make spike-run-fc_layer16only  FC_BENCH_BATCHSIZE=8
#
# BATCHSIZE was 12 (FP32) vs 4 (FP16) and EPOCHS 1 vs 20. Beyond the obvious
# step-count skew, batch 12 pads to 16 in fmatmul_row_block() while batch 4 does
# not -- so the two apps were not even executing the same forward code path.
# Batch 4 is the padding-free choice for both.
FC_BENCH_BATCHSIZE ?= 4
FC_BENCH_EPOCHS    ?= 1
# SHAPE = vggnet's LARGEST fc layer (fc1), so this microbenchmark measures the
# layer that actually dominates the real workload rather than a synthetic square:
#   vggnet(alpha) fc1: FC1_IN_UNITS = C2_CHANNELS*POOLING2_L^2 = 24*8*8 = 1536
#                      FC1_LAYER                                       =  192
#   vggnet(alpha) fc2: 192 -> 100   (15x fewer MACs -- not the interesting one)
# Both fit the static caps in fc_layer.h: in <= FC_MAX_IN_UNITS (2048),
# out <= FC_MAX_INTERNAL (512).
#
# The old 256x256 was chosen so BOTH precisions saturate their lanes on the m4
# fmatmul path (VLMAX(e16,m4) = 2*VL_bytes, so >= 256 keeps FP16 100% occupied
# up to VL=128 B and yields the honest 2:1 strip ratio vs FP32). 1536 is well
# above that threshold, so the saturation argument still holds -- and holds up
# to VL=384 B, past the whole lane/VLEN sweep.
FC_BENCH_IN_UNITS  ?= 1536
FC_BENCH_OUT_UNITS ?= 192
FC_BENCH_SEED      ?= 42
FC_BENCH_LOSS      ?= MSE

# ---------------------------------------------------------------------------
# fc_layer32: pure FP32 fully-connected layer (mirrors fc_layer semantics)
# ---------------------------------------------------------------------------
FC_LAYER32_MODE ?= train
FC_LAYER32_LOSS ?= $(FC_BENCH_LOSS)
FC_LAYER32_BATCHSIZE ?= $(FC_BENCH_BATCHSIZE)
FC_LAYER32_EPOCHS ?= $(FC_BENCH_EPOCHS)
FC_LAYER32_INFER_IDX ?= -1
FC_LAYER32_STATIC_MAX_BATCH ?= $(FC_LAYER32_BATCHSIZE)
FC_LAYER32_SAMPLES ?= $(FC_LAYER32_BATCHSIZE)

# ---- benchmark data is GENERATED FROM THESE KNOBS ----
# generated_data/ used to be a checked-in 4-sample blob while FC_LAYER32_BATCHSIZE
# defaulted to 12. Nothing tied the two together, and train.c's batch fetch copies
# batchsize*FC_INPUT_UNITS floats regardless of FC_TOTAL_SAMPLES -- so it read
# 12*128 = 1536 floats out of a 512-float array, ran off inputs.bin into
# targets.bin/labels.bin, and reported an inf loss on the very first forward
# (before any weight update). Regenerating from the same knobs that size the run
# makes that mismatch impossible rather than merely detectable.
# These reproduce the previously checked-in blobs bit-for-bit at the defaults.
FC_LAYER32_IN_UNITS  ?= $(FC_BENCH_IN_UNITS)
FC_LAYER32_OUT_UNITS ?= $(FC_BENCH_OUT_UNITS)
FC_LAYER32_SEED      ?= $(FC_BENCH_SEED)

FC_LAYER32_DATA_STAMP  := fc_layer32/generated_data/.params
FC_LAYER32_DATA_PARAMS := samples=$(FC_LAYER32_SAMPLES) in=$(FC_LAYER32_IN_UNITS) out=$(FC_LAYER32_OUT_UNITS) seed=$(FC_LAYER32_SEED)

# The recipe runs every build but regenerates -- and touches the stamp -- ONLY
# when the parameter set actually changed, so dependents rebuild only then
# instead of on every invocation.
.PHONY: fc_layer32-data-force
fc_layer32-data-force:

$(FC_LAYER32_DATA_STAMP): fc_layer32-data-force
	@mkdir -p $(@D)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(FC_LAYER32_DATA_PARAMS)" ]; then \
	    echo "  GEN     fc_layer32/generated_data ($(FC_LAYER32_DATA_PARAMS))"; \
	    ( cd fc_layer32 && $(PYTHON) scripts/generate_fc_test_vectors.py \
	        --samples $(FC_LAYER32_SAMPLES) --inputs $(FC_LAYER32_IN_UNITS) \
	        --outputs $(FC_LAYER32_OUT_UNITS) --seed $(FC_LAYER32_SEED) \
	        --out-dir generated_data --weights-c kernel/weights.c ) > /dev/null; \
	    echo "$(FC_LAYER32_DATA_PARAMS)" > $@; \
	fi

ifeq ($(FC_LAYER32_MODE),train)
FC_LAYER32_MODE_DEFINES := -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=$(FC_LAYER32_BATCHSIZE) -DALEXNET_STATIC_MAX_BATCH=$(FC_LAYER32_STATIC_MAX_BATCH) -DALEXNET_EPOCHS=$(FC_LAYER32_EPOCHS) -DFC_TOTAL_SAMPLES=$(FC_LAYER32_SAMPLES)
else ifeq ($(FC_LAYER32_MODE),inference)
FC_LAYER32_MODE_DEFINES := -DALEXNET_MODE_INFERENCE -DALEXNET_INFER_IDX=$(FC_LAYER32_INFER_IDX)
else
$(error Unsupported FC_LAYER32_MODE='$(FC_LAYER32_MODE)'. Use train or inference)
endif

ifeq ($(FC_LAYER32_LOSS),MSE)
FC_LAYER32_LOSS_DEFINES := -DSET_MSE
else ifeq ($(FC_LAYER32_LOSS),CEL)
FC_LAYER32_LOSS_DEFINES := -DSET_CEL
else
$(error Unsupported FC_LAYER32_LOSS='$(FC_LAYER32_LOSS)'. Use MSE or CEL)
endif

# Dimensions must reach the COMPILER too, not just the data generator: the app
# sizes its static buffers and the op descriptor from these macros. Passing them
# only to the generator would regenerate data at the new shape while the binary
# stayed at 128 -- a silent buffer overrun.
FC_LAYER32_DIM_DEFINES := -DFC_INPUT_UNITS=$(FC_LAYER32_IN_UNITS) -DFC_OUTPUT_UNITS=$(FC_LAYER32_OUT_UNITS)

FC_LAYER32_DEFINES := $(FC_LAYER32_MODE_DEFINES) $(FC_LAYER32_LOSS_DEFINES) $(FC_LAYER32_DIM_DEFINES)

bin/fc_layer32:  $(APPS_DIR)/Makefile
bin/fc_layer32%: $(APPS_DIR)/Makefile
bin/fc_layer32:  ENV_DEFINES += $(FC_LAYER32_DEFINES)
bin/fc_layer32%: ENV_DEFINES += $(FC_LAYER32_DEFINES)
bin/fc_layer32:  AUTOVECTORIZE = 0
bin/fc_layer32%: AUTOVECTORIZE = 0
bin/fc_layer32:  RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer32
bin/fc_layer32%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer32
bin/fc_layer32:  RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer32
bin/fc_layer32%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer32
bin/fc_layer32:  RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer32
bin/fc_layer32%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer32

fc_layer32/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# Rebuild fc_layer32 objects when a Makefile knob changes, and when the generated
# data changes. Only `bin/fc_layer32%` depended on the Makefile before, which is
# the LINK step -- so changing FC_LAYER32_BATCHSIZE relinked stale objects still
# compiled with the old -DALEXNET_BATCHSIZE, and main.c kept printing the
# previous batch size. Same explicit-list form as conv_layer32 below (a
# recipe-less pattern rule would be a silent no-op here).
FC_LAYER32_SRCS := $(call find_app_sources,fc_layer32)
$(addsuffix .o,       $(FC_LAYER32_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER32_DATA_STAMP)
$(addsuffix .o.spike, $(FC_LAYER32_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER32_DATA_STAMP)
$(addsuffix .o.fpga,  $(FC_LAYER32_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER32_DATA_STAMP)
$(addsuffix .o.fpga16,$(FC_LAYER32_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER32_DATA_STAMP)

# ---------------------------------------------------------------------------
# fc_layer_test: FP16(mixed) vs FP32 equivalence harness. Links the 3 kernels
# under test from both trees (fp16/ from fc_layer, fp32/ from fc_layer32 with
# _32-suffixed externals). NOTE: bin/fc_layer% already supplies FC_LAYER_DEFINES
# and AUTOVECTORIZE=0; here we (1) point the assembler at this app's own
# generated_data for data.S, and (2) build main.c (which carries the hand-copied
# vector momentum_sgd asm) without autovectorization, mirroring the train.c rule.
# ---------------------------------------------------------------------------
bin/fc_layer_test:   $(APPS_DIR)/Makefile
bin/fc_layer_test%: $(APPS_DIR)/Makefile
bin/fc_layer_test:   ENV_DEFINES += -DALEXNET_STATIC_MAX_BATCH=2 -DALEXNET_BATCHSIZE=2
bin/fc_layer_test%: ENV_DEFINES += -DALEXNET_STATIC_MAX_BATCH=2 -DALEXNET_BATCHSIZE=2
bin/fc_layer_test:   AUTOVECTORIZE = 0
bin/fc_layer_test%: AUTOVECTORIZE = 0
bin/fc_layer_test:   RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer_test
bin/fc_layer_test%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer_test
bin/fc_layer_test:   RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer_test
bin/fc_layer_test%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer_test
bin/fc_layer_test:   RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer_test
bin/fc_layer_test%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer_test

fc_layer_test/main.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
fc_layer_test/main.c.o:       RISCV_CCFLAGS       += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# ---------------------------------------------------------------------------
# fc_layer16only: pure FP16 fully-connected layer
# ---------------------------------------------------------------------------
FC_LAYER16ONLY_MODE ?= train
FC_LAYER16ONLY_LOSS ?= $(FC_BENCH_LOSS)
FC_LAYER16ONLY_BATCHSIZE ?= $(FC_BENCH_BATCHSIZE)
FC_LAYER16ONLY_EPOCHS ?= $(FC_BENCH_EPOCHS)
FC_LAYER16ONLY_INFER_IDX ?= -1
FC_LAYER16ONLY_STATIC_MAX_BATCH ?= $(FC_LAYER16ONLY_BATCHSIZE)
FC_LAYER16ONLY_SAMPLES ?= $(FC_LAYER16ONLY_BATCHSIZE)
# Learning rate and momentum, for sweeping the FP16 update-swamping threshold:
#   make spike-run-fc_layer16only FC_LAYER16ONLY_LR=0.01 FC_LAYER16ONLY_MOMENTUM=0
# NOTE: the default MUST equal fc_layer32's LEARNING_RATE (0.00001f, set in
# fc_layer32/kernel/train.c) or the two apps take differently-sized steps.
FC_LAYER16ONLY_LR ?= 0.00001f
FC_LAYER16ONLY_MOMENTUM ?= 1

# ---- benchmark data is GENERATED FROM THESE KNOBS (mirrors fc_layer32) ----
# generated_data/ was a checked-in 4-sample blob with no rule tying it to
# FC_LAYER16ONLY_BATCHSIZE -- exactly the hazard that was already fixed on the
# fc_layer32 side. Any batch size > 4 would have read off the end of inputs.bin
# into targets.bin/labels.bin. Generating from the same knobs that size the run
# makes that impossible, and driving both generators with the same
# --samples/--inputs/--outputs/--seed keeps the FP16 arrays the exact fp16
# rounding of the FP32 ones.
FC_LAYER16ONLY_IN_UNITS  ?= $(FC_BENCH_IN_UNITS)
FC_LAYER16ONLY_OUT_UNITS ?= $(FC_BENCH_OUT_UNITS)
FC_LAYER16ONLY_SEED      ?= $(FC_BENCH_SEED)

FC_LAYER16ONLY_DATA_STAMP  := fc_layer16only/generated_data/.params
FC_LAYER16ONLY_DATA_PARAMS := samples=$(FC_LAYER16ONLY_SAMPLES) in=$(FC_LAYER16ONLY_IN_UNITS) out=$(FC_LAYER16ONLY_OUT_UNITS) seed=$(FC_LAYER16ONLY_SEED)

.PHONY: fc_layer16only-data-force
fc_layer16only-data-force:

$(FC_LAYER16ONLY_DATA_STAMP): fc_layer16only-data-force
	@mkdir -p $(@D)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(FC_LAYER16ONLY_DATA_PARAMS)" ]; then \
	    echo "  GEN     fc_layer16only/generated_data ($(FC_LAYER16ONLY_DATA_PARAMS))"; \
	    ( cd fc_layer16only && $(PYTHON) scripts/generate_fc_test_vectors.py \
	        --samples $(FC_LAYER16ONLY_SAMPLES) --inputs $(FC_LAYER16ONLY_IN_UNITS) \
	        --outputs $(FC_LAYER16ONLY_OUT_UNITS) --seed $(FC_LAYER16ONLY_SEED) \
	        --out-dir generated_data --weights-c kernel/weights.c ) > /dev/null; \
	    echo "$(FC_LAYER16ONLY_DATA_PARAMS)" > $@; \
	fi

ifeq ($(FC_LAYER16ONLY_MODE),train)
FC_LAYER16ONLY_MODE_DEFINES := -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=$(FC_LAYER16ONLY_BATCHSIZE) -DALEXNET_STATIC_MAX_BATCH=$(FC_LAYER16ONLY_STATIC_MAX_BATCH) -DALEXNET_EPOCHS=$(FC_LAYER16ONLY_EPOCHS) -DFC_TOTAL_SAMPLES=$(FC_LAYER16ONLY_SAMPLES)
else ifeq ($(FC_LAYER16ONLY_MODE),inference)
FC_LAYER16ONLY_MODE_DEFINES := -DALEXNET_MODE_INFERENCE -DALEXNET_INFER_IDX=$(FC_LAYER16ONLY_INFER_IDX)
else
$(error Unsupported FC_LAYER16ONLY_MODE='$(FC_LAYER16ONLY_MODE)'. Use train or inference)
endif

ifeq ($(FC_LAYER16ONLY_LOSS),MSE)
FC_LAYER16ONLY_LOSS_DEFINES := -DSET_MSE
else ifeq ($(FC_LAYER16ONLY_LOSS),CEL)
FC_LAYER16ONLY_LOSS_DEFINES := -DSET_CEL
else
$(error Unsupported FC_LAYER16ONLY_LOSS='$(FC_LAYER16ONLY_LOSS)'. Use MSE or CEL)
endif

FC_LAYER16ONLY_OPT_DEFINES := '-DLEARNING_RATE=((_Float16)$(FC_LAYER16ONLY_LR))' -DALEXNET_USE_MOMENTUM=$(FC_LAYER16ONLY_MOMENTUM)

FC_LAYER16ONLY_DIM_DEFINES := -DFC_INPUT_UNITS=$(FC_LAYER16ONLY_IN_UNITS) -DFC_OUTPUT_UNITS=$(FC_LAYER16ONLY_OUT_UNITS)

FC_LAYER16ONLY_DEFINES := $(FC_LAYER16ONLY_MODE_DEFINES) $(FC_LAYER16ONLY_LOSS_DEFINES) $(FC_LAYER16ONLY_OPT_DEFINES) $(FC_LAYER16ONLY_DIM_DEFINES)

bin/fc_layer16only:  $(APPS_DIR)/Makefile
bin/fc_layer16only%: $(APPS_DIR)/Makefile
bin/fc_layer16only:  ENV_DEFINES += $(FC_LAYER16ONLY_DEFINES)
bin/fc_layer16only%: ENV_DEFINES += $(FC_LAYER16ONLY_DEFINES)
bin/fc_layer16only:  AUTOVECTORIZE = 0
bin/fc_layer16only%: AUTOVECTORIZE = 0
bin/fc_layer16only:  RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer16only
bin/fc_layer16only%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer16only
bin/fc_layer16only:  RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer16only
bin/fc_layer16only%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer16only
bin/fc_layer16only:  RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer16only
bin/fc_layer16only%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer16only

fc_layer16only/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# Ensure fc_layer16only objects rebuild when Makefile-defined knobs change.
# Without this, `make spike-run-fc_layer16only FC_LAYER16ONLY_EPOCHS=3` relinks
# STALE objects and silently runs with the previous epoch count / LR.
#
# NB: this must be an explicit target list, not `fc_layer16only/%.c.o.spike: Makefile`.
# A pattern rule with no recipe does not add prerequisites to the pattern rule in
# runtime.mk that actually builds the object -- make just ignores it. (The
# conv_layer/conv_layer32 guards further down are written that way and are no-ops.)
FC_LAYER16ONLY_SRCS := $(call find_app_sources,fc_layer16only)
$(addsuffix .o,       $(FC_LAYER16ONLY_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER16ONLY_DATA_STAMP)
$(addsuffix .o.spike, $(FC_LAYER16ONLY_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER16ONLY_DATA_STAMP)
$(addsuffix .o.fpga,  $(FC_LAYER16ONLY_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER16ONLY_DATA_STAMP)
$(addsuffix .o.fpga16,$(FC_LAYER16ONLY_SRCS)): $(APPS_DIR)/Makefile $(FC_LAYER16ONLY_DATA_STAMP)

# ---------------------------------------------------------------------------
# fc_layer16_batch12: assembler include root for data.S .incbin paths.
# Had no dedicated rule at all, so .incbin "generated_data/..." couldn't
# resolve for any build flavor (normal/spike/fpga).
# ---------------------------------------------------------------------------
bin/fc_layer16_batch12:  $(APPS_DIR)/Makefile
bin/fc_layer16_batch12%: $(APPS_DIR)/Makefile
bin/fc_layer16_batch12:  RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer16_batch12
bin/fc_layer16_batch12%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/fc_layer16_batch12
bin/fc_layer16_batch12:  RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer16_batch12
bin/fc_layer16_batch12%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/fc_layer16_batch12
bin/fc_layer16_batch12:  RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer16_batch12
bin/fc_layer16_batch12%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/fc_layer16_batch12

# Compile data.S from within fc_layer16only/ so that .incbin resolves
# "generated_data/..." relative to the app dir, not the apps/ CWD.
# Without this, a stray apps/generated_data/ on the server confuses LLVM.
fc_layer16only/data.S.o: fc_layer16only/data.S
	cd $(APPS_DIR)/fc_layer16only && $(RISCV_CC) $(RISCV_CCFLAGS) -c data.S -o $(APPS_DIR)/fc_layer16only/data.S.o

fc_layer16only/data.S.o.spike: fc_layer16only/data.S patch-spike-crt0
	cd $(APPS_DIR)/fc_layer16only && $(RISCV_CC) $(RISCV_CCFLAGS_SPIKE) -c data.S -o $(APPS_DIR)/fc_layer16only/data.S.o.spike

# conv_layer profile controls (mirrors fc_layer semantics)
CONV_LAYER_MODE ?= train
CONV_LAYER_BATCHSIZE ?= 4
CONV_LAYER_EPOCHS ?= 1
CONV_LAYER_INFER_IDX ?= -1
CONV_LAYER_STATIC_MAX_BATCH ?= $(CONV_LAYER_BATCHSIZE)

# Spatial / channel dimensions — override to benchmark different configs.
# Kernel stays 3x3; same-padding so OUT_H=IN_H, OUT_W=IN_W -- identical to every
# conv in vggnet (3x3 / stride 1 / pad 1), so a vggnet layer maps here exactly.
#
# SHAPE = vggnet's LARGEST conv layer (conv2). The alpha variant's four convs:
#   conv1:  3->12 @ 32x32   0.33 M MAC   X_col  27,648
#   conv2: 12->12 @ 32x32   1.33 M MAC   X_col 110,592   <-- this one
#   conv3: 12->24 @ 16x16   0.66 M MAC   X_col  27,648
#   conv4: 24->24 @ 16x16   1.33 M MAC   X_col  55,296
# conv2 and conv4 TIE on MACs. conv2 is the tie-break because it is strictly
# larger on everything the lane/VLEN/LLC sweep actually varies: 2x the
# activation and im2col traffic, and N = out_h*out_w = 1024 vs conv4's 256 --
# at VLEN=2048/e16/m4, VLMAX = 512, so conv2 runs two full strips while conv4
# leaves half the lanes idle. conv2 is therefore also the slower of the two in
# vggnet, i.e. the runtime-dominant conv. To benchmark conv4 instead:
#   make ... CONV_LAYER_C_IN=24 CONV_LAYER_C_OUT=24 CONV_LAYER_H=16 CONV_LAYER_W=16
#
# out_h*out_w = 1024 is exactly CONV_MAX_OWOH in convolution_layer.h -- at the
# cap, not over it. Do not raise H/W without raising that too.
# generated_data/ + kernel/weights.c are regenerated from these knobs by the
# CONV_*_DATA_STAMP rules below; no manual generator run is needed.
CONV_LAYER_C_IN  ?= 12
CONV_LAYER_C_OUT ?= 12
CONV_LAYER_H     ?= 32
CONV_LAYER_W     ?= 32

# im2col scratch for layer 1 (one image): C_IN * 9 * H * W
CONV_LAYER_XCOL_ELEMS := $(shell expr $(CONV_LAYER_C_IN) \* 9 \* $(CONV_LAYER_H) \* $(CONV_LAYER_W))

# ---- conv benchmark data is GENERATED FROM THESE KNOBS ----
# Same rule the fc_layer32 / fc_layer16only pair already has, and for the same
# reason -- except here the hazard had already fired. kernel/weights.c is a
# CHECKED-IN generated file: weights.h declares conv1_weights[CONV1_WEIGHT_ELEMS]
# from the -D dims, while weights.c defines whatever size it was last generated
# at. C does not check array bounds across translation units, so a stale
# weights.c is a silent over-read, not a link error. conv_layer16only ran for
# weeks with weights.c generated at C_IN=24 C_OUT=24 H=16 W=16 (5,184 elements)
# against a declaration of 18,432: a 13,248-element over-read that poisoned
# 252 weights with NaN, made 3,674/4,096 outputs NaN, and printed a loss of
# 2147483647 that was misread as an FP16 overflow for several sessions.
# inputs.bin/targets.bin/labels.bin are the same story via data.S's .incbin.
#
# So: regenerate from the very knobs that size the run, and stamp the parameter
# set so it happens exactly when a knob changes and never otherwise.
CONV_LAYER_SEED ?= 42
CONV_LAYER_DATA_PARAMS := batch=$(CONV_LAYER_BATCHSIZE) c_in=$(CONV_LAYER_C_IN) c_out=$(CONV_LAYER_C_OUT) h=$(CONV_LAYER_H) w=$(CONV_LAYER_W) seed=$(CONV_LAYER_SEED)

# All three conv apps share CONV_LAYER_* by design (benchmark parity), so they
# share one generator invocation shape too. Each generator writes its own
# precision: conv_layer/conv_layer32 emit float, conv_layer16only emits
# _Float16, from the same seed and the same RNG call order -- so at matching
# dims the FP16 arrays are the exact fp16 rounding of the FP32 ones and an
# accuracy delta is attributable to precision alone.
CONV_DATA_APPS := conv_layer conv_layer32 conv_layer16only

define CONV_DATA_STAMP_RULE
.PHONY: $(1)-data-force
$(1)-data-force:

$(1)/generated_data/.params: $(1)-data-force
	@mkdir -p $$(@D)
	@if [ "$$$$(cat $$@ 2>/dev/null)" != "$(CONV_LAYER_DATA_PARAMS)" ]; then \
	    echo "  GEN     $(1)/generated_data ($(CONV_LAYER_DATA_PARAMS))"; \
	    ( cd $(1) && $(PYTHON) scripts/generate_conv_test_vectors.py \
	        --batch $(CONV_LAYER_BATCHSIZE) --c-in $(CONV_LAYER_C_IN) \
	        --c-out $(CONV_LAYER_C_OUT) --h $(CONV_LAYER_H) --w $(CONV_LAYER_W) \
	        --seed $(CONV_LAYER_SEED) \
	        --out-dir generated_data --weights-c kernel/weights.c ) > /dev/null; \
	    echo "$(CONV_LAYER_DATA_PARAMS)" > $$@; \
	fi
endef
$(foreach a,$(CONV_DATA_APPS),$(eval $(call CONV_DATA_STAMP_RULE,$(a))))

ifeq ($(CONV_LAYER_MODE),train)
CONV_LAYER_MODE_DEFINES := -DALEXNET_MODE_TRAIN -DALEXNET_BATCHSIZE=$(CONV_LAYER_BATCHSIZE) -DALEXNET_STATIC_MAX_BATCH=$(CONV_LAYER_STATIC_MAX_BATCH) -DALEXNET_EPOCHS=$(CONV_LAYER_EPOCHS)
else ifeq ($(CONV_LAYER_MODE),inference)
CONV_LAYER_MODE_DEFINES := -DALEXNET_MODE_INFERENCE -DALEXNET_INFER_IDX=$(CONV_LAYER_INFER_IDX)
else
$(error Unsupported CONV_LAYER_MODE='$(CONV_LAYER_MODE)'. Use train or inference)
endif

CONV_LAYER_DIM_DEFINES := \
    -DCONV1_IN_CHANNELS=$(CONV_LAYER_C_IN)   \
    -DCONV1_OUT_CHANNELS=$(CONV_LAYER_C_OUT)  \
    -DCONV1_IN_H=$(CONV_LAYER_H) -DCONV1_IN_W=$(CONV_LAYER_W) \
    -DCONV1_OUT_H=$(CONV_LAYER_H) -DCONV1_OUT_W=$(CONV_LAYER_W) \
    -DCONV1_XCOL_ELEMS=$(CONV_LAYER_XCOL_ELEMS)

CONV_LAYER_DEFINES := $(CONV_LAYER_MODE_DEFINES) $(CONV_LAYER_DIM_DEFINES)

# conv_layer needs: (1) compile-time train knobs, (2) autovectorization off,
# (3) assembler include root for data.S .incbin paths.
# NOTE: use bin/conv_layer.% (dot separator) to avoid matching bin/conv_layer32.
bin/conv_layer:   $(APPS_DIR)/Makefile
bin/conv_layer.%: $(APPS_DIR)/Makefile
bin/conv_layer:   ENV_DEFINES += $(CONV_LAYER_DEFINES)
bin/conv_layer.%: ENV_DEFINES += $(CONV_LAYER_DEFINES)
bin/conv_layer:   AUTOVECTORIZE = 0
bin/conv_layer.%: AUTOVECTORIZE = 0
bin/conv_layer:   RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer
bin/conv_layer.%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer
bin/conv_layer:   RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer
bin/conv_layer.%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer
bin/conv_layer:   RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer
bin/conv_layer.%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer

# train.c must be built -O0: its inline asm declares no "memory" / vector-reg
# clobbers, so at -O3 the compiler reorders around it and the app segfaults.
# NOTE: this is masking the real bug (missing clobbers), not fixing it.
#
# Applied to BOTH conv_layer and conv_layer32, and to all three flavors:
#   .o        -> plain / Verilator RTL sim
#   .o.spike  -> Spike
#   .o.fpga   -> FPGA
# Previously only conv_layer's .o.spike had it, which (a) left the Verilator and
# FPGA builds at -O3 (same latent segfault), and (b) meant conv_layer's train.c
# was -O0 while conv_layer32's was -O3 -- making every train.c-resident stage
# (prep / loss / zero_d_input / update) incomparable between the two.
conv_layer/kernel/train.c.o:         RISCV_CCFLAGS       += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer/kernel/train.c.o.spike:   RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer/kernel/train.c.o.fpga:    RISCV_CCFLAGS_FPGA  += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer32/kernel/train.c.o:       RISCV_CCFLAGS       += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer32/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer32/kernel/train.c.o.fpga:  RISCV_CCFLAGS_FPGA  += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# conv_layer32: same profile knobs as conv_layer, but must NOT reuse conv_layer's
# -Wa,-I path. bin/conv_layer% used to match bin/conv_layer32 too, silently
# pointing .incbin at conv_layer/generated_data/ instead of conv_layer32's own
# (much larger) dataset, and never set RISCV_CCFLAGS_FPGA at all.
bin/conv_layer32:   $(APPS_DIR)/Makefile
bin/conv_layer32.%: $(APPS_DIR)/Makefile
bin/conv_layer32:   ENV_DEFINES += $(CONV_LAYER_DEFINES)
bin/conv_layer32.%: ENV_DEFINES += $(CONV_LAYER_DEFINES)
bin/conv_layer32:   AUTOVECTORIZE = 0
bin/conv_layer32.%: AUTOVECTORIZE = 0
bin/conv_layer32:   RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer32
bin/conv_layer32.%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer32
bin/conv_layer32:   RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer32
bin/conv_layer32.%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer32
bin/conv_layer32:   RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer32
bin/conv_layer32.%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer32

# ---------------------------------------------------------------------------
# conv_layer16only: FP16 sibling of conv_layer32.
#
# This app previously had NO block at all. conv_layer's rules are scoped
# `bin/conv_layer.%` (with a dot) and conv_layer32's `bin/conv_layer32.%`, so
# bin/conv_layer16only matched NEITHER: it was built with no -DCONV1_* defines
# (silently falling back to the 43/43/8/8 header defaults, which disagree with
# the config kernel/weights.c was generated for) and with no -Wa,-I, so data.S
# could not resolve .incbin "generated_data/..." and the build failed outright.
#
# BENCHMARK PARITY: these dims MUST track CONV_LAYER_* above (which conv_layer32
# is built with). C_OUT was 16 here against conv_layer32's 32, so the FP16 app
# executed exactly half the FLOPs and half the output-side memory traffic --
# every cycle comparison between the two was meaningless. Keep them equal.
#
# Defaults match the config recorded in kernel/weights.c, regenerate with:
#   cd conv_layer16only && python3 scripts/generate_conv_test_vectors.py \
#       --batch 1 --c-in 64 --c-out 32 --h 1 --w 128
# The generator uses the same seed and the same RNG call order as the FP32 one,
# so at matching dims the FP16 arrays are the exact fp16 rounding of the FP32
# arrays -- accuracy deltas are then attributable to precision alone.
# ---------------------------------------------------------------------------
CONV_LAYER16ONLY_BATCHSIZE ?= $(CONV_LAYER_BATCHSIZE)
CONV_LAYER16ONLY_EPOCHS ?= $(CONV_LAYER_EPOCHS)
CONV_LAYER16ONLY_STATIC_MAX_BATCH ?= $(CONV_LAYER16ONLY_BATCHSIZE)
CONV_LAYER16ONLY_C_IN  ?= $(CONV_LAYER_C_IN)
CONV_LAYER16ONLY_C_OUT ?= $(CONV_LAYER_C_OUT)
CONV_LAYER16ONLY_H     ?= $(CONV_LAYER_H)
CONV_LAYER16ONLY_W     ?= $(CONV_LAYER_W)
CONV_LAYER16ONLY_XCOL_ELEMS := $(shell expr $(CONV_LAYER16ONLY_C_IN) \* 9 \* $(CONV_LAYER16ONLY_H) \* $(CONV_LAYER16ONLY_W))

CONV_LAYER16ONLY_DEFINES := \
    -DALEXNET_MODE_TRAIN \
    -DALEXNET_BATCHSIZE=$(CONV_LAYER16ONLY_BATCHSIZE) \
    -DALEXNET_STATIC_MAX_BATCH=$(CONV_LAYER16ONLY_STATIC_MAX_BATCH) \
    -DALEXNET_EPOCHS=$(CONV_LAYER16ONLY_EPOCHS) \
    -DCONV1_IN_CHANNELS=$(CONV_LAYER16ONLY_C_IN) \
    -DCONV1_OUT_CHANNELS=$(CONV_LAYER16ONLY_C_OUT) \
    -DCONV1_IN_H=$(CONV_LAYER16ONLY_H) -DCONV1_IN_W=$(CONV_LAYER16ONLY_W) \
    -DCONV1_OUT_H=$(CONV_LAYER16ONLY_H) -DCONV1_OUT_W=$(CONV_LAYER16ONLY_W) \
    -DCONV1_XCOL_ELEMS=$(CONV_LAYER16ONLY_XCOL_ELEMS)

bin/conv_layer16only:   $(APPS_DIR)/Makefile
bin/conv_layer16only.%: $(APPS_DIR)/Makefile
bin/conv_layer16only:   ENV_DEFINES += $(CONV_LAYER16ONLY_DEFINES)
bin/conv_layer16only.%: ENV_DEFINES += $(CONV_LAYER16ONLY_DEFINES)
bin/conv_layer16only:   AUTOVECTORIZE = 0
bin/conv_layer16only.%: AUTOVECTORIZE = 0
bin/conv_layer16only:   RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer16only
bin/conv_layer16only.%: RISCV_CCFLAGS += -Wa,-I$(CURDIR)/conv_layer16only
bin/conv_layer16only:   RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer16only
bin/conv_layer16only.%: RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/conv_layer16only
bin/conv_layer16only:   RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer16only
bin/conv_layer16only.%: RISCV_CCFLAGS_FPGA += -Wa,-I$(CURDIR)/conv_layer16only

conv_layer16only/kernel/train.c.o:       RISCV_CCFLAGS       += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer16only/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
conv_layer16only/kernel/train.c.o.fpga:  RISCV_CCFLAGS_FPGA  += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# Ensure conv objects rebuild when Makefile-defined knobs change; otherwise a
# CONV_LAYER_W change silently relinks stale objects built with the old
# -DCONV1_* dims. The generated_data/.params stamp is listed alongside the
# Makefile so a knob change ALSO drags in the freshly regenerated weights.c --
# without it, weights.c is just another source file whose mtime happens to be
# old, and the stale-array over-read comes straight back.
#
# These MUST be explicit target lists. The previous form here --
#     conv_layer/%.c.o.spike: $(APPS_DIR)/Makefile
# -- was a silent no-op: a recipe-less pattern rule does not add a prerequisite to
# the pattern rule in common/runtime.mk that actually builds the object; make just
# ignores it. Same fix as fc_layer16only above.
#
# .o.fpga16 is in the list: it was missing, so define-only changes shipped FPGA
# FP16 binaries built from objects that predated the change.
CONV_LAYER_SRCS       := $(call find_app_sources,conv_layer)
CONV_LAYER32_SRCS     := $(call find_app_sources,conv_layer32)
CONV_LAYER16ONLY_SRCS := $(call find_app_sources,conv_layer16only)

define CONV_OBJ_DEPS_RULE
$$(addsuffix .o,        $(2)): $(APPS_DIR)/Makefile $(1)/generated_data/.params
$$(addsuffix .o.spike,  $(2)): $(APPS_DIR)/Makefile $(1)/generated_data/.params
$$(addsuffix .o.fpga,   $(2)): $(APPS_DIR)/Makefile $(1)/generated_data/.params
$$(addsuffix .o.fpga16, $(2)): $(APPS_DIR)/Makefile $(1)/generated_data/.params

# data.S .incbin's generated_data/*.bin, but the %.S.o rules only depend on the
# .S file -- regenerating the data would NOT rebuild these. Make it explicit.
# The stamp is the reliable trigger here; the wildcard is parse-time and yields
# nothing on a clean tree where the .bin files do not exist yet.
$(1)/data.S.o $(1)/data.S.o.spike $(1)/data.S.o.fpga $(1)/data.S.o.fpga16: \
    $(1)/generated_data/.params $$(wildcard $(APPS_DIR)/$(1)/generated_data/*.bin)
endef
$(eval $(call CONV_OBJ_DEPS_RULE,conv_layer,$(CONV_LAYER_SRCS)))
$(eval $(call CONV_OBJ_DEPS_RULE,conv_layer32,$(CONV_LAYER32_SRCS)))
$(eval $(call CONV_OBJ_DEPS_RULE,conv_layer16only,$(CONV_LAYER16ONLY_SRCS)))

# ---------------------------------------------------------------------------
# vggnet: MiniVGGNet fp32 classifier-head fine-tuning / test-time adaptation on
# CIFAR-100 (and CIFAR-100-C), using the vectorized fc_layer32 + conv_layer32
# kernels. Built via the generic app template below (same pattern as fc_layer32).
#
# Prerequisites (run once, from apps/vggnet):
#   python3 scripts/prepare_cifar100c.py CIFAR-100-C/ --all --severity 2   # adaptation data
#   # embed the pretrained checkpoint into kernel/weights.c (no filesystem on Spike)
#
# Usage:
#   make spike-run-vggnet                                    # alpha, bs=32, 2 steps
#   make bin/vggnet.spike VGGNET_VARIANT=full VGGNET_MAX_STEPS=1
#
# NOTES:
#   - The cifar10 data.S is renamed out of the *.S glob (both data.S files export
#     the same symbols); data_cifar100.S carries the embedded dataset.
#   - The stray scripts/im2col_test.c (a second main) is renamed out of the glob.
#   - FINETUNE_MODE + SPIKE (see main.c) skips file I/O and adapts the embedded
#     weights.c in place, training only fc1/bn5/fc2 with the conv stack frozen.
# ---------------------------------------------------------------------------
VGGNET_BATCHSIZE        ?= 32
VGGNET_EPOCHS           ?= 1
VGGNET_STATIC_MAX_BATCH ?= $(VGGNET_BATCHSIZE)
# Cap steps/epoch so a Spike run finishes in reasonable time; 0 = full epoch.
VGGNET_MAX_STEPS        ?= 2
# variant: alpha (~324K params, C1=12 C2=24 FC1=192, default) | full (32/64/512)
VGGNET_VARIANT          ?= alpha
# Regularisation knobs (override to match a scalar reference run, e.g. all 0 for
# a deterministic scalar-vs-vector loss comparison).
VGGNET_DROP1            ?= 0.25
VGGNET_DROP2            ?= 0.25
VGGNET_DROPFC          ?= 0.5
VGGNET_AUGMENT         ?= 1
# 1 (default) = freeze the conv stack and adapt only fc1/bn5/fc2 from the
# embedded weights.c. 0 = ALEXNET_MODE_TRAIN proper: random init, every layer
# trainable, so conv backward and the bn1-bn4 backward (spatial 1024/256)
# actually run -- matches `make FINETUNE=0` on the native scalar reference.
VGGNET_FINETUNE        ?= 1

ifeq ($(VGGNET_VARIANT),full)
VGGNET_VARIANT_DEFINES := -DC1_CHANNELS=32 -DC2_CHANNELS=64 -DFC1_LAYER=512
else
VGGNET_VARIANT_DEFINES := -DC1_CHANNELS=12 -DC2_CHANNELS=24 -DFC1_LAYER=192
endif

# -DFINETUNE_MODE is no longer hardcoded here; it is appended below when
# VGGNET_FINETUNE=1 (the default), so VGGNET_FINETUNE=0 gives a full training run.
VGGNET_DEFINES := \
    -DALEXNET_MODE_TRAIN \
    -DALEXNET_BATCHSIZE=$(VGGNET_BATCHSIZE) \
    -DALEXNET_STATIC_MAX_BATCH=$(VGGNET_STATIC_MAX_BATCH) \
    -DALEXNET_EPOCHS=$(VGGNET_EPOCHS) \
    -DALEXNET_MAX_STEPS=$(VGGNET_MAX_STEPS) \
    -DOUT_LAYER=100 \
    -DDROPOUT1=$(VGGNET_DROP1)f -DDROPOUT2=$(VGGNET_DROP2)f -DDROPOUT_FC=$(VGGNET_DROPFC)f \
    -DAUGMENT=$(VGGNET_AUGMENT) \
    $(VGGNET_VARIANT_DEFINES)

ifeq ($(VGGNET_FINETUNE),1)
VGGNET_DEFINES += -DFINETUNE_MODE
endif
ifneq ($(VGGNET_INIT_GAIN),)
VGGNET_DEFINES += -DWEIGHT_INIT_GAIN=$(VGGNET_INIT_GAIN)f
endif

# Bisection knobs: route conv/fc forward through the scalar reference paths to
# isolate whether the vector conv (fconv3d) or vector fc (fmatmul) diverges.
ifeq ($(VGGNET_SCALAR_CONV),1)
VGGNET_DEFINES += -DUSE_IM2COL_CONV
endif
ifeq ($(VGGNET_SCALAR_FC),1)
VGGNET_DEFINES += -DUSE_SCALAR_FC
endif
# VGGNET_DUMP_CONV1=1 -> one-shot conv1 output dump for image 0 (diagnostic).
ifeq ($(VGGNET_DUMP_CONV1),1)
VGGNET_DEFINES += -DDUMP_CONV1
endif
# VGGNET_DUMP_TRACE=1 -> one-shot per-layer checksum of the first training
# forward pass, for locating the first layer that diverges from the native
# scalar reference (build that side with `make TRACE=1`).
ifeq ($(VGGNET_DUMP_TRACE),1)
VGGNET_DEFINES += -DDUMP_TRACE
endif

# VGGNET_SGD_SCALAR=1 selects the scalar reference sgd_step instead of the
# vectorized one -- used to A/B the two against each other.
VGGNET_SGD_SCALAR ?= 0
ifeq ($(VGGNET_SGD_SCALAR),1)
VGGNET_DEFINES += -DSGD_SCALAR
endif

bin/vggnet:    $(APPS_DIR)/Makefile
bin/vggnet.%:  $(APPS_DIR)/Makefile
bin/vggnet:    ENV_DEFINES += $(VGGNET_DEFINES)
bin/vggnet.%:  ENV_DEFINES += $(VGGNET_DEFINES)
bin/vggnet:    AUTOVECTORIZE = 0
bin/vggnet.%:  AUTOVECTORIZE = 0
bin/vggnet:    RISCV_CCFLAGS       += -Wa,-I$(CURDIR)/vggnet
bin/vggnet.%:  RISCV_CCFLAGS       += -Wa,-I$(CURDIR)/vggnet
bin/vggnet:    RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/vggnet
bin/vggnet.%:  RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/vggnet
# kernel/activation_layer.c pulls __exp_2xf32 from vggnet/lib/exp.h
# (a symlink to apps/exp/kernel/exp.h, so vggnet stays self-contained).
bin/vggnet:    RISCV_CCFLAGS       += -I$(CURDIR)/vggnet/lib
bin/vggnet.%:  RISCV_CCFLAGS       += -I$(CURDIR)/vggnet/lib
bin/vggnet:    RISCV_CCFLAGS_SPIKE += -I$(CURDIR)/vggnet/lib
bin/vggnet.%:  RISCV_CCFLAGS_SPIKE += -I$(CURDIR)/vggnet/lib
bin/vggnet:    RISCV_CCFLAGS_FPGA  += -I$(CURDIR)/vggnet/lib
bin/vggnet.%:  RISCV_CCFLAGS_FPGA  += -I$(CURDIR)/vggnet/lib
bin/vggnet:    RISCV_CCFLAGS_FPGA  += -Wa,-I$(CURDIR)/vggnet
bin/vggnet.%:  RISCV_CCFLAGS_FPGA  += -Wa,-I$(CURDIR)/vggnet

# train.c holds the scalar SGD update; main.c is scalar forward orchestration.
# Build both at -O0 without the LLVM autovectorizer so it never emits scalable
# vector ops that clash with the hand-written fixed-vl kernels.
vggnet/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
vggnet/main.c.o.spike:         RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

# Every VGGNET_* knob lands in ENV_DEFINES, which only affects the *compile*
# line. Declaring the dependency on the link target alone (`bin/vggnet%:
# Makefile`) relinks stale objects instead of rebuilding them, so a changed
# knob silently has no effect -- e.g. main.c kept a -DFINETUNE_MODE object
# across runs that passed VGGNET_FINETUNE=0, which cost three debugging
# detours. List the objects explicitly, as fc_layer32 already does.
# Depending on the Makefile alone is NOT enough: make cannot see a change to a
# command-line variable, because the Makefile's own mtime is unchanged. So
# record the resolved define set in a stamp and rewrite it only when it
# differs -- the objects then rebuild exactly when a knob actually changes.
# GUARD: vggnet/ is not always present -- a partial checkout, or an scp of just
# the layer benchmarks, leaves it missing. Everything below runs at PARSE time
# ($(shell ...) and $(call find_app_sources,...)), so without this guard EVERY
# make invocation -- including `make clean` -- errors with
#   bash: .../vggnet/.build_flags.tmp: No such file or directory
#   find: 'vggnet': No such file or directory
ifneq ($(wildcard $(APPS_DIR)/vggnet),)

VGGNET_FLAGS_STAMP := $(APPS_DIR)/vggnet/.build_flags
$(shell printf '%s\n' '$(VGGNET_DEFINES)' > $(VGGNET_FLAGS_STAMP).tmp; \
        cmp -s $(VGGNET_FLAGS_STAMP).tmp $(VGGNET_FLAGS_STAMP) \
          || mv $(VGGNET_FLAGS_STAMP).tmp $(VGGNET_FLAGS_STAMP); \
        rm -f $(VGGNET_FLAGS_STAMP).tmp)

VGGNET_SRCS := $(call find_app_sources,vggnet)
$(addsuffix .o,       $(VGGNET_SRCS)): $(APPS_DIR)/Makefile $(VGGNET_FLAGS_STAMP)
$(addsuffix .o.spike, $(VGGNET_SRCS)): $(APPS_DIR)/Makefile $(VGGNET_FLAGS_STAMP)
$(addsuffix .o.fpga,  $(VGGNET_SRCS)): $(APPS_DIR)/Makefile $(VGGNET_FLAGS_STAMP)

# data_cifar100.S .incbin's kernel/cifar100_*.bin; rebuild the object when the
# embedded dataset changes.
vggnet/data_cifar100.S.o vggnet/data_cifar100.S.o.spike vggnet/data_cifar100.S.o.fpga: $(wildcard $(APPS_DIR)/vggnet/kernel/cifar100_*.bin)

endif # vggnet present

# NOTE: `spike-run-vggnet` is auto-generated by the generic run_benchmark_spike
# template below (like every other app), so no explicit rule is needed here.

# ---------------------------------------------------------------------------
# vggnet16: FP16 twin of vggnet. Same MiniVGGNet architecture, but every layer
# runs in _Float16 (see vggnet16/kernel/*). Training uses the FP16 techniques
# from fc_layer16only / conv_layer16only -- dynamic loss scaling, fflags-based
# overflow/underflow detection, skip-step on overflow, momentum as the sole
# accumulator (NO fp32 master weights), RNE rounding, fp32 islands for the
# softmax transcendentals. Built via the generic app template, like vggnet.
#
# weights.c is FP16, generated by scripts/binary2text_weights_fp16.py.
#
# Usage:
#   make spike-run-vggnet16 VGGNET16_FINETUNE=0 VGGNET16_BATCHSIZE=4 VGGNET16_EPOCHS=4
# ---------------------------------------------------------------------------
VGGNET16_BATCHSIZE        ?= 32
VGGNET16_EPOCHS           ?= 1
VGGNET16_STATIC_MAX_BATCH ?= $(VGGNET16_BATCHSIZE)
VGGNET16_MAX_STEPS        ?= 2
VGGNET16_VARIANT          ?= alpha
VGGNET16_DROP1            ?= 0.25
VGGNET16_DROP2            ?= 0.25
VGGNET16_DROPFC          ?= 0.5
VGGNET16_AUGMENT         ?= 1
VGGNET16_FINETUNE        ?= 1

ifeq ($(VGGNET16_VARIANT),full)
VGGNET16_VARIANT_DEFINES := -DC1_CHANNELS=32 -DC2_CHANNELS=64 -DFC1_LAYER=512
else
VGGNET16_VARIANT_DEFINES := -DC1_CHANNELS=12 -DC2_CHANNELS=24 -DFC1_LAYER=192
endif

VGGNET16_DEFINES := \
    -DALEXNET_MODE_TRAIN \
    -DALEXNET_BATCHSIZE=$(VGGNET16_BATCHSIZE) \
    -DALEXNET_STATIC_MAX_BATCH=$(VGGNET16_STATIC_MAX_BATCH) \
    -DALEXNET_EPOCHS=$(VGGNET16_EPOCHS) \
    -DALEXNET_MAX_STEPS=$(VGGNET16_MAX_STEPS) \
    -DOUT_LAYER=100 \
    -DDROPOUT1=$(VGGNET16_DROP1)f -DDROPOUT2=$(VGGNET16_DROP2)f -DDROPOUT_FC=$(VGGNET16_DROPFC)f \
    -DAUGMENT=$(VGGNET16_AUGMENT) \
    $(VGGNET16_VARIANT_DEFINES)

ifeq ($(VGGNET16_FINETUNE),1)
VGGNET16_DEFINES += -DFINETUNE_MODE
endif
ifeq ($(VGGNET16_SELFTEST),1)
VGGNET16_DEFINES += -DCONV_SELFTEST
endif
ifneq ($(VGGNET16_INIT_GAIN),)
VGGNET16_DEFINES += -DWEIGHT_INIT_GAIN=$(VGGNET16_INIT_GAIN)f
endif
ifneq ($(VGGNET16_INPUT_SCALE),)
VGGNET16_DEFINES += -DINPUT_SCALE=$(VGGNET16_INPUT_SCALE)f
endif
ifeq ($(VGGNET16_DUMP_TRACE),1)
VGGNET16_DEFINES += -DDUMP_TRACE
endif
# Bracket the backward pass with prints of the loss local. cross_entropy_loss
# clamps tp to >=1e-7 before logf, so its return is bounded by 16.118 and can
# never be inf -- if the value goes non-finite between there and the return,
# the slot is being clobbered rather than diverging numerically.
ifeq ($(VGGNET16_LOSS_TRACE),1)
VGGNET16_DEFINES += -DLOSS_TRACE
endif
# Three-way conv1 A/B (vector f16 vs scalar f16 vs scalar f32) + input/weight/
# bias checksums. Was previously built by hand; wired up so the flags stamp
# tracks it like every other knob.
ifeq ($(VGGNET16_ABDEBUG),1)
VGGNET16_DEFINES += -DCONV1_ABDEBUG=1
endif
# One-shot dump of the first BatchNorm's internals, split into the reduction /
# the factor scaling / the normalize arithmetic. bn1 is where the forward pass
# first goes non-finite on hardware while its input is still bit-exact.
ifeq ($(VGGNET16_BN1_PROBE),1)
VGGNET16_DEFINES += -DBN1_PROBE
endif
VGGNET16_SGD_SCALAR ?= 0
ifeq ($(VGGNET16_SGD_SCALAR),1)
VGGNET16_DEFINES += -DSGD_SCALAR
endif

bin/vggnet16:    $(APPS_DIR)/Makefile
bin/vggnet16.%:  $(APPS_DIR)/Makefile
bin/vggnet16:    ENV_DEFINES += $(VGGNET16_DEFINES)
bin/vggnet16.%:  ENV_DEFINES += $(VGGNET16_DEFINES)
bin/vggnet16.fpga:   RISCV_CCFLAGS_FPGA += -DDETERMINISTIC_RNG
bin/vggnet16.fpga16: RISCV_CCFLAGS_FPGA16 += -DDETERMINISTIC_RNG
vggnet16/kernel/data.c.o.spike: RISCV_CCFLAGS_SPIKE += -DBATCH_TRACE
vggnet16/kernel/data.c.o.fpga16: RISCV_CCFLAGS_FPGA16 += -DBATCH_TRACE
vggnet16/kernel/rng_spike.c.o.fpga: RISCV_CCFLAGS_FPGA += -DDETERMINISTIC_RNG
vggnet16/kernel/rng_spike.c.o.fpga16: RISCV_CCFLAGS_FPGA16 += -DDETERMINISTIC_RNG
bin/vggnet16:    AUTOVECTORIZE = 0
bin/vggnet16.%:  AUTOVECTORIZE = 0
bin/vggnet16:    RISCV_CCFLAGS       += -Wa,-I$(CURDIR)/vggnet16
bin/vggnet16.%:  RISCV_CCFLAGS       += -Wa,-I$(CURDIR)/vggnet16
bin/vggnet16:    RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/vggnet16
bin/vggnet16.%:  RISCV_CCFLAGS_SPIKE += -Wa,-I$(CURDIR)/vggnet16
# kernel/activation_layer.c pulls __exp_2xf32 from vggnet16/lib/exp.h
bin/vggnet16:    RISCV_CCFLAGS       += -I$(CURDIR)/vggnet16/lib
bin/vggnet16.%:  RISCV_CCFLAGS       += -I$(CURDIR)/vggnet16/lib
bin/vggnet16:    RISCV_CCFLAGS_SPIKE += -I$(CURDIR)/vggnet16/lib
bin/vggnet16.%:  RISCV_CCFLAGS_SPIKE += -I$(CURDIR)/vggnet16/lib
bin/vggnet16:    RISCV_CCFLAGS_FPGA  += -I$(CURDIR)/vggnet16/lib
bin/vggnet16.%:  RISCV_CCFLAGS_FPGA  += -I$(CURDIR)/vggnet16/lib
bin/vggnet16:    RISCV_CCFLAGS_FPGA  += -Wa,-I$(CURDIR)/vggnet16
bin/vggnet16.%:  RISCV_CCFLAGS_FPGA  += -Wa,-I$(CURDIR)/vggnet16

vggnet16/kernel/train.c.o.spike: RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off
vggnet16/main.c.o.spike:         RISCV_CCFLAGS_SPIKE += -O0 -fno-vectorize -mllvm -scalable-vectorization=off

ifneq ($(wildcard $(APPS_DIR)/vggnet16),)
VGGNET16_FLAGS_STAMP := $(APPS_DIR)/vggnet16/.build_flags
$(shell printf '%s\n' '$(VGGNET16_DEFINES)' > $(VGGNET16_FLAGS_STAMP).tmp; \
        cmp -s $(VGGNET16_FLAGS_STAMP).tmp $(VGGNET16_FLAGS_STAMP) \
          || mv $(VGGNET16_FLAGS_STAMP).tmp $(VGGNET16_FLAGS_STAMP); \
        rm -f $(VGGNET16_FLAGS_STAMP).tmp)

VGGNET16_SRCS := $(call find_app_sources,vggnet16)
$(addsuffix .o,        $(VGGNET16_SRCS)): $(APPS_DIR)/Makefile $(VGGNET16_FLAGS_STAMP)
$(addsuffix .o.spike,  $(VGGNET16_SRCS)): $(APPS_DIR)/Makefile $(VGGNET16_FLAGS_STAMP)
$(addsuffix .o.fpga,   $(VGGNET16_SRCS)): $(APPS_DIR)/Makefile $(VGGNET16_FLAGS_STAMP)
# .o.fpga16 was missing here. Without it a define-only change (VGGNET16_ABDEBUG,
# VGGNET16_DUMP_TRACE, ...) did not rebuild the fpga16 objects: only files whose
# SOURCE also changed picked up the new flags, so a binary could silently carry
# some probes and not others. That is exactly how a build made with
# -DCONV1_ABDEBUG=1 reached the board with no A/B output in it.
$(addsuffix .o.fpga16, $(VGGNET16_SRCS)): $(APPS_DIR)/Makefile $(VGGNET16_FLAGS_STAMP)

vggnet16/data_cifar100.S.o vggnet16/data_cifar100.S.o.spike vggnet16/data_cifar100.S.o.fpga vggnet16/data_cifar100.S.o.fpga16: $(wildcard $(APPS_DIR)/vggnet16/kernel/cifar100_*.bin)
endif # vggnet16 present

all: $(BINARIES)

# Pre-process the linker-script to correclty align the sections
.PHONY: linker_script
linker_script: $(COMMON_DIR)/script/align_sections.sh $(ROOT_DIR)/../../config/$(config).mk
	chmod +x $(COMMON_DIR)/script/align_sections.sh
	rm -f $(COMMON_DIR)/link.ld && cp $(COMMON_DIR)/arch.link.ld $(COMMON_DIR)/link.ld
	$(COMMON_DIR)/script/align_sections.sh $(nr_lanes) $(COMMON_DIR)/link.ld

.PHONY: linker_script_fpga
linker_script_fpga: $(COMMON_DIR)/script/align_sections.sh $(ROOT_DIR)/../../config/$(config).mk
	chmod +x $(COMMON_DIR)/script/align_sections.sh
	rm -f $(COMMON_DIR)/link.fpga.ld && cp $(COMMON_DIR)/arch.link.fpga.ld $(COMMON_DIR)/link.fpga.ld
	$(COMMON_DIR)/script/align_sections.sh $(nr_lanes) $(COMMON_DIR)/link.fpga.ld

# Make all applications
$(APPS): % : bin/% $(APPS_DIR)/Makefile $(shell find common -type f)
.PHONY: $(BINARIES)

# Patch spike crt0 to enable vector extension before execution
.PHONY: patch-spike-crt0
patch-spike-crt0: $(spike_env_dir)/benchmarks/common/crt.S
	sed -i s/"li t0, MSTATUS_FS | MSTATUS_XS$$"/"li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS"/ $<
	git update-index --assume-unchanged $<

# Only some applications need to generate and then compile the data file
# Generate a dummy data file for applications that do not need it
ifeq ($(old_data),1)
else
define app_gen_data_template
.PHONY: $1/data.S
$1/data.S:
	cd $1 && if [ -d script ]; then ${PYTHON} script/gen_data.py $(subst ",,$(def_args_$1)) > data.S ; else touch data.S; fi
endef
$(foreach app,$(APPS),$(eval $(call app_gen_data_template,$(app))))
endif

define vector_trace_template
ideal_dispatcher/vtrace/$1.vtrace: bin/$1.spike $(VTRACE_SCRIPTS)
	mkdir -p ideal_dispatcher/vtrace ideal_dispatcher/log ideal_dispatcher/temp
	echo "run" | $(RISCV_SIM_MOD) $(RISCV_SIM_MOD_OPT) $$< 2> ideal_dispatcher/temp/$1.temp 1> ideal_dispatcher/log/$1.log
	cd ideal_dispatcher && scripts/vtrace.sh temp/$1.temp vtrace/$1.vtrace
endef
$(foreach app,$(APPS),$(eval $(call vector_trace_template,$(app))))

define app_compile_template_ideal
bin/$1.ideal: bin/$1.spike ideal_dispatcher/vtrace/$1.vtrace
	mkdir -p bin/
	cp bin/$1.spike bin/$1.ideal
endef
$(foreach app,$(APPS),$(eval $(call app_compile_template_ideal,$(app))))

define app_compile_template_spike
bin/$1.spike: $1/data.S.o.spike $(addsuffix .o.spike, $(call find_app_sources,$1)) $(RUNTIME_SPIKE) $(COMMON_SPIKE) patch-spike-crt0
	mkdir -p bin/
	$$(RISCV_CC) -Iinclude $$(RISCV_CCFLAGS_SPIKE) -o $$@ $$(addsuffix .o.spike, $$(call find_app_sources,$1)) $(RUNTIME_SPIKE) $$(RISCV_LDFLAGS_SPIKE) -DSPIKE
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
endef
$(foreach app,$(APPS),$(eval $(call app_compile_template_spike,$(app))))

define app_compile_template_fpga
bin/$1.fpga: $1/data.S.o.fpga $(addsuffix .o.fpga, $(call find_app_sources,$1)) $(RUNTIME_FPGA) linker_script_fpga
	mkdir -p bin/
	$$(RISCV_CC) -Iinclude $$(RISCV_CCFLAGS_FPGA) -o $$@ $$(addsuffix .o.fpga, $$(call find_app_sources,$1)) $(RUNTIME_FPGA) $$(RISCV_LDFLAGS_FPGA)
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
	@case " $$(INT64CONV_EXEMPT) " in \
	  *" $1 "*) echo "  [int64conv] $1 exempt (deliberately tests the defect)" ;; \
	  *) $(APPS_DIR)/common/script/check_int64conv.sh $$(RISCV_OBJDUMP) $$@ ;; \
	esac
	$$(RISCV_STRIP) $$@ -S --strip-unneeded
	$$(RISCV_OBJCOPY) -O binary $$@ $$@.bin
endef
$(foreach app,$(APPS),$(eval $(call app_compile_template_fpga,$(app))))

# softfloat branch: FPUSupportHalf-only build for vggnet16/conv_layer16only/
# fc_layer16only (see RISCV_CCFLAGS_FPGA16/RUNTIME_FPGA16 in
# common/runtime.mk). Mirrors app_compile_template_fpga exactly, just on the
# .fpga16 suffix throughout, so it never shares an object path -- and never
# overwrites the normal bin/$1.fpga -- with the full-FPU build.
define app_compile_template_fpga16
bin/$1.fpga16: $1/data.S.o.fpga16 $(addsuffix .o.fpga16, $(call find_app_sources,$1)) $(RUNTIME_FPGA16) linker_script_fpga
	mkdir -p bin/
	$$(RISCV_CC) -Iinclude $$(RISCV_CCFLAGS_FPGA16) -o $$@ $$(addsuffix .o.fpga16, $$(call find_app_sources,$1)) $(RUNTIME_FPGA16) $$(RISCV_LDFLAGS_FPGA16)
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
	@case " $$(INT64CONV_EXEMPT) " in \
	  *" $1 "*) echo "  [int64conv] $1 exempt (deliberately tests the defect)" ;; \
	  *) $(APPS_DIR)/common/script/check_int64conv.sh $$(RISCV_OBJDUMP) $$@ ;; \
	esac
	$$(RISCV_STRIP) $$@ -S --strip-unneeded
	$$(RISCV_OBJCOPY) -O binary $$@ $$@.bin
endef

# fcvt.s.l/.lu and fcvt.l/.lu.s DO NOT READ THEIR OPERAND on the current
# bitstream: with RVD=0 the scalar FLen is 32, so fpnew masks INT64 out of
# IntFmtConfig and the cast unit returns the inactive format's all-ones
# DONT_CARE. fcvt.s.lu yields 2^32 and fcvt.s.l yields -1.0 for ANY input, with
# no trap and no flag. This silently zeroed fc_layer16only's weight gradients on
# every training step (AGENT_NOTES_FP16_4LANE.md 46.8) until the (uint32_t)
# narrowing at fmatmul_tn.c:120. The guard below stops that reappearing.
# Remove it once the hardware is fixed (RVD=1, or the cvfpu INT64 patch).
# Apps listed here contain the instruction ON PURPOSE, to test the defect.
INT64CONV_EXEMPT := vec_test fpubug

FP16ONLY_APPS := vggnet16 conv_layer16only fc_layer16only dma_test vec_test fpubug memlat dma_test64 memmatrix latprobe fconv_cmp vrare vrare2 vrare3 vrare4 sldprobe
$(foreach app,$(FP16ONLY_APPS),$(eval $(call app_compile_template_fpga16,$(app))))

# Each app's own -Wa,-I (and vggnet16's extra -I) override RISCV_CCFLAGS_FPGA
# elsewhere in this file (search "-Wa,-I$(CURDIR)"), which RISCV_CCFLAGS_FPGA16
# no longer inherits (it's an independent flag chain -- see the comment in
# common/runtime.mk on why appending to RISCV_CCFLAGS_FPGA doesn't work for
# -march). Mirror them here for the .fpga16 targets specifically.
bin/vggnet16.fpga16:         RISCV_CCFLAGS_FPGA16 += -I$(CURDIR)/vggnet16/lib
bin/vggnet16.fpga16:         RISCV_CCFLAGS_FPGA16 += -Wa,-I$(CURDIR)/vggnet16
bin/conv_layer16only.fpga16: RISCV_CCFLAGS_FPGA16 += -Wa,-I$(CURDIR)/conv_layer16only
bin/fc_layer16only.fpga16:   RISCV_CCFLAGS_FPGA16 += -Wa,-I$(CURDIR)/fc_layer16only

define app_compile_template
bin/$1$(BIN_SUFFIX): $1/data.S.o $(addsuffix .o, $(call find_app_sources,$1)) $(RUNTIME_LLVM) linker_script
	mkdir -p bin/
	$$(RISCV_CC) -Iinclude $$(RISCV_CCFLAGS) -o $$@ $$(addsuffix .o, $$(call find_app_sources,$1)) $(RUNTIME_LLVM) $$(RISCV_LDFLAGS)
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
	$$(RISCV_STRIP) $$@ -S --strip-unneeded
endef
$(foreach app,$(APPS),$(eval $(call app_compile_template,$(app))))

# Make the RISC-V tests
riscv_tests: $(CVA6_BINARIES) $(ARA_BINARIES)

define rvtest_compile_template
TESTS_$(1) := $(addprefix bin/, $($(addsuffix _ara_tests, $1)))

bin/$(1)-ara-%: $(TESTS_DIR)/$(1)/%.$(2) $(RUNTIME_GCC) linker_script
	mkdir -p bin/
	$$(RISCV_CC_GCC) -Iinclude -I$$(TESTS_DIR)/macros/scalar -I$$(TESTS_DIR)/macros/vector $$(RISCV_CCFLAGS_GCC) $$(RISCV_LDFLAGS_GCC) -o $$@ $$< $(RUNTIME_GCC)
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
	$$(RISCV_STRIP) $$@ -S --strip-unneeded
endef

define rvtest_compile_template_c
TESTS_$(1) := $(addprefix bin/, $($(addsuffix _ara_tests, $1)))

bin/$(1)-ara-%: $(TESTS_DIR)/$(1)/%.$(2) $(RUNTIME_LLVM) linker_script
	mkdir -p bin/
	$$(RISCV_CC) -Iinclude -I$$(TESTS_DIR)/macros/scalar -I$$(TESTS_DIR)/macros/vector $$(RISCV_CCFLAGS) $$(RISCV_LDFLAGS) -o $$@ $$< $(RUNTIME_LLVM)
	$$(RISCV_OBJDUMP) $$(RISCV_OBJDUMP_FLAGS) -D $$@ > $$@.dump
	$$(RISCV_STRIP) $$@ -S --strip-unneeded
endef
# CVA6 tests are written in assembly
$(foreach extension, $(CVA6_EXTENSIONS), $(eval $(call rvtest_compile_template,$(extension),S)))
# Ara tests are written in C
$(foreach extension, $(ARA_EXTENSIONS), $(eval $(call rvtest_compile_template_c,$(extension),c)))

# Run Spike tests
.PHONY: riscv_tests_spike
riscv_tests_spike: patch-spike-crt0
	# Compile the tests
	PATH=$(GCC_INSTALL_DIR)/bin:$(LLVM_INSTALL_DIR)/bin:${PATH} ARA_DEFINES="$(DEFINES)" make -j4 -C riscv-tests/isa
	# Run the tests
	PATH=$(ISA_SIM_INSTALL_DIR)/bin:${PATH} make -C riscv-tests/isa run

# Format the sourcecode
.PHONY: format
format:
	$(LLVM_INSTALL_DIR)/bin/clang-format -style=file -i $$(find . | grep -E "\.[h,c,cpp]$$" | grep -v riscv-tests)
	$(LLVM_INSTALL_DIR)/bin/clang-format -style=file -i $$(find riscv-tests/isa/rv64uv | grep -E "\.[h,c,cpp]$$")

# Run benchmarks on Spike
define run_benchmark_spike
spike-run-$1: bin/$1.spike
	mkdir -p spike_runs
	@echo
	@echo "Simulating $1 with SPIKE:"
	@echo
	$(RISCV_SIM) $(RISCV_SIM_OPT) $$< | tee spike_runs/$$@
endef
$(foreach app,$(APPS),$(eval $(call run_benchmark_spike,$(app))))

# Clean Spike simulation
.PHONY: riscv_tests_spike_clean
riscv_tests_spike_clean:
	make -C riscv-tests/isa clean

.PHONY: benchmarks_clean
benchmarks_clean:
	cd $(APPS_DIR)/benchmarks && \
	rm -vf *.S.* *.c.*        && \
	rm -vf data/*.S.*         && \
	rm -vf kernel/*.c.*

.PHONY: clean
clean: benchmarks_clean
	rm -vf $(BINARIES)
	rm -vf $(CVA6_BINARIES)
	rm -vf $(ARA_BINARIES)
	rm -vf $(addsuffix .dump,$(BINARIES))
	rm -vf $(addsuffix .dump,$(CVA6_BINARIES))
	rm -vf $(addsuffix .dump,$(ARA_BINARIES))
	rm -vf $(addsuffix /main.c.o,$(APPS))
	rm -vf $(RUNTIME_GCC)
	rm -vf $(RUNTIME_LLVM)
	rm -vf $(RUNTIME_SPIKE)
	for app in $(APPS); do cd $(APPS_DIR)/$${app} && rm -f $$(find . -name "*.c.o*" -o -name "*.S.o*") && cd ..; done

.INTERMEDIATE: $(addsuffix /main.c.o,$(APPS))
