# MiniVGGNet CIFAR-100 — GCC build
# Usage:
#   make                          → train alpha=0.375 (324,400 params) on CIFAR-100
#   make VARIANT=full             → train the reference 32/64/512 net (2,215,940 params)
#   make BATCHSIZE=128 EPOCHS=50
#   make run                      → build + run training
#   make finetune                 → freeze conv stack, fine-tune the head from minivgg.weights
#   make infer IDX=0              → single-image inference
#   make debug BATCHSIZE=64       → build with -g -O0 -fsanitize=address,undefined
#   make asan                     → build + run under AddressSanitizer
#   make clean

CC      := gcc
# -O3 -march=native is ~2x faster than -O2 and bit-identical in output (no
# -ffast-math, so no reassociation). It ties the binary to this CPU: for the
# spike/ara cross-builds, override with e.g. `make OPT=-O2`.
OPT     := -O3 -march=native -funroll-loops
# OpenMP parallelises the race-free hot loops (matrix_multiply, relu) across
# cores. Kept separate from OPT so the spike/ara cross-builds (which override
# OPT) stay single-threaded; without -fopenmp the `#pragma omp` lines are simply
# ignored, so serial behaviour is bit-identical. Control threads at runtime with
# OMP_NUM_THREADS=N (defaults to all cores).
OPENMP  := -fopenmp
WARN    := -Wall -Wextra -Wno-unused-parameter
STD     := -std=c11
INCLUDE := -I. -Ikernel

# NOTE: keep these assignments free of trailing comments. Make preserves the
# whitespace before a `#`, so `DATASET ?= cifar100  # ...` yields "cifar100  "
# and every ifeq against "cifar100" silently fails.
BATCHSIZE ?= 64
EPOCHS    ?= 30
IDX       ?= 0
# LR is multiplied by 0.1 every LR_STEP_EPOCHS epochs (so 12 -> decays at 12, 24)
LR_STEP   ?= 12
# DATASET:  cifar10 | cifar100
DATASET   ?= cifar100
# VARIANT:  alpha (0.375, default) | full (1.0)
VARIANT   ?= alpha
# FINETUNE: 1 → freeze the conv stack, train the classifier only
FINETUNE  ?= 0
# Regularisation. AUGMENT: random horizontal flip on training batches (1/0).
# DROP1/DROP2/DROPFC: dropout probs after pool1 / pool2 / FC head (0 disables).
# WD: L2 weight decay on weight matrices only. Set all to 0 for the bare net.
AUGMENT   ?= 1
DROP1     ?= 0.25
DROP2     ?= 0.25
DROPFC    ?= 0.5
WD        ?= 0.0005

ifeq ($(DATASET),cifar100)
    DATA_S     := data_cifar100.S
    DEFS_DATA  := -DOUT_LAYER=100
else
    DATA_S     := data.S
    DEFS_DATA  := -DOUT_LAYER=10
endif

# Uniform width multiplier applied to the reference MiniVGGNet 32/32/64/64/512.
#   full   alpha = 1.0    -> 32 / 64 / 512
#   alpha  alpha = 0.375  -> 12 / 24 / 192
ifeq ($(VARIANT),full)
    DEFS_VARIANT := -DC1_CHANNELS=32 -DC2_CHANNELS=64 -DFC1_LAYER=512
else
    DEFS_VARIANT := -DC1_CHANNELS=12 -DC2_CHANNELS=24 -DFC1_LAYER=192
endif

ifeq ($(FINETUNE),1)
    DEFS_FINETUNE := -DFINETUNE_MODE
else
    DEFS_FINETUNE :=
endif

DEFS_REG := \
    -DAUGMENT=$(AUGMENT) \
    -DDROPOUT1=$(DROP1)f -DDROPOUT2=$(DROP2)f -DDROPOUT_FC=$(DROPFC)f \
    -DWEIGHT_DECAY=$(WD)f

DEFS_TRAIN := \
    -DALEXNET_MODE_TRAIN \
    -DALEXNET_BATCHSIZE=$(BATCHSIZE) \
    -DALEXNET_EPOCHS=$(EPOCHS) \
    -DALEXNET_STATIC_MAX_BATCH=$(BATCHSIZE) \
    -DLR_STEP_EPOCHS=$(LR_STEP) \
    $(DEFS_DATA) $(DEFS_VARIANT) $(DEFS_FINETUNE) $(DEFS_REG)

DEFS_INFER := \
    -DALEXNET_MODE_INFERENCE \
    -DALEXNET_BATCHSIZE=1 \
    -DALEXNET_STATIC_MAX_BATCH=1 \
    -DALEXNET_INFER_IDX=$(IDX) \
    $(DEFS_DATA) $(DEFS_VARIANT)

CFLAGS_TRAIN := $(OPT) $(OPENMP) $(WARN) $(STD) $(INCLUDE) $(DEFS_TRAIN)
CFLAGS_INFER := $(OPT) $(OPENMP) $(WARN) $(STD) $(INCLUDE) $(DEFS_INFER)
CFLAGS_DEBUG := -O0 -g3 $(WARN) $(STD) $(INCLUDE) $(DEFS_TRAIN) \
    -fsanitize=address,undefined -fno-omit-frame-pointer

KERNEL_SRCS := \
    kernel/activation_layer.c \
    kernel/batchnorm_layer.c \
    kernel/convolution_layer.c \
    kernel/data.c \
    kernel/dropout_layer.c \
    kernel/fc_layer.c \
    kernel/image_inference.c \
    kernel/inference.c \
    kernel/matrix.c \
    kernel/maxpooling_layer.c \
    kernel/train.c \
    kernel/weights.c \
    $(DATA_S)

MAIN_SRC := main.c

TARGET_TRAIN := minivgg_train
TARGET_INFER := minivgg_infer
TARGET_DEBUG := minivgg_debug

# ---------- default target: train ----------

all: $(TARGET_TRAIN)

$(TARGET_TRAIN): $(MAIN_SRC) $(KERNEL_SRCS)
	$(CC) $(CFLAGS_TRAIN) -o $@ $^ -lm

# ---------- run ----------

run: $(TARGET_TRAIN)
	./$(TARGET_TRAIN)

# ---------- fine-tune (loads minivgg.weights, freezes the conv stack) ----------

finetune:
	$(MAKE) -B FINETUNE=1 run

# ---------- inference ----------

infer: $(TARGET_INFER)
	./$(TARGET_INFER)

$(TARGET_INFER): $(MAIN_SRC) $(KERNEL_SRCS)
	$(CC) $(CFLAGS_INFER) -o $@ $^ -lm

# ---------- debug (AddressSanitizer + UBSan + gdb) ----------

$(TARGET_DEBUG): $(MAIN_SRC) $(KERNEL_SRCS)
	$(CC) $(CFLAGS_DEBUG) -o $@ $^ -lm

debug: $(TARGET_DEBUG)

asan: $(TARGET_DEBUG)
	./$(TARGET_DEBUG)

gdb: $(TARGET_DEBUG)
	gdb -ex run ./$(TARGET_DEBUG)

# ---------- clean ----------

clean:
	rm -f $(TARGET_TRAIN) $(TARGET_INFER) $(TARGET_DEBUG)

.PHONY: all run finetune infer debug asan gdb clean
