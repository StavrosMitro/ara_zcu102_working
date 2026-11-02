//
// File:        dropout_layer.h
// Description: inverted dropout with a stored mask for the backward pass
// Author:      Haris Wang
//
// Inverted dropout: during training each element is kept with probability
// keep = 1 - prob and rescaled by 1/keep, so the expected activation is
// unchanged and evaluation needs no rescaling (it simply skips dropout).
//
// dropout_forward fills `mask` (1/keep for kept units, 0 for dropped) and
// scales x in place. dropout_backward multiplies the incoming gradient by the
// same mask, so it must be called with the mask produced by the matching
// forward pass on the same batch.
//
// n is the total element count (batchsize * units_per_sample).

void dropout_forward(_Float16 *x, _Float16 *mask, float prob, int n);
void dropout_backward(_Float16 *dx, const _Float16 *mask, int n);
