#ifndef WEIGHTS_H
#define WEIGHTS_H

// From weights.c
#include "alexnet.h"   // FC_INPUT_UNITS / FC_OUTPUT_UNITS (single source of truth)

// Bounds derive from the FC dimension knob in alexnet.h so that
// sizeof(fc1_weights_32) (used by the shape check in main.c) tracks the real size.
extern float fc1_weights_32[FC_INPUT_UNITS * FC_OUTPUT_UNITS];
extern float fc1_bias_32[FC_OUTPUT_UNITS];

// From the assembly file (.incbin)
extern const float test_inputs_32[];
extern const float test_targets_32[];
extern const int test_labels_32[];

#endif