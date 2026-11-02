#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
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

# Data for the fmatmul_tn vs fmatmul_deferred cycle comparison.
#
# Both kernels are made to compute the SAME product, C[N,P] = A^T * B, with
# A=[M,N] and B=[M,P]:
#
#   fmatmul_tn_32(c, a, b, M, N, P)              reads A and B as stored,
#                                                and scales the result by 1/M
#   fmatmul_4x4_deferred_32(c, at, bt, N, M, P)  computes at * bt^T, so it needs
#                                                the operands pre-transposed
#
# arg1, arg2, arg3: M, N, P

import numpy as np
import sys


def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i + 3 - n]
    print("    .word 0x%s" % s)


############
## SCRIPT ##
############

if len(sys.argv) == 4:
  M = int(sys.argv[1])
  N = int(sys.argv[2])
  P = int(sys.argv[3])
else:
  print("Error. Give me three arguments: M, N, P.")
  print("C = A^T B with A=[MxN], B=[MxP], C=[NxP]")
  sys.exit(1)

dtype = np.float32

# A = [M,N], B = [M,P]  ->  C = A^T B = [N,P]
A = np.random.rand(M, N).astype(dtype)
B = np.random.rand(M, P).astype(dtype)

# Pre-transposed operands, for the deferred kernel (which computes a * b^T).
At = np.ascontiguousarray(A.T)  # [N,M]
Bt = np.ascontiguousarray(B.T)  # [P,M]

# Golden results. fmatmul_tn_32 folds a 1/M batch scaling into the kernel;
# fmatmul_4x4_deferred_32 does not, so each gets its own gold.
G = np.matmul(A.T, B).astype(dtype)
G_tn = (G / np.float32(M)).astype(dtype)

C = np.zeros([N, P], dtype=dtype)

print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("P", np.array(P, dtype=np.uint64))

# Big matrices go in .l2 (matches main.c's section(".l2") attribute).
print(".section .l2,\"aw\",@progbits")
emit("a", A, 'NR_LANES*4')
emit("b", B, 'NR_LANES*4')
emit("at", At, 'NR_LANES*4')
emit("bt", Bt, 'NR_LANES*4')
emit("c_tn", C, 'NR_LANES*4')
emit("c_df", C, 'NR_LANES*4')
emit("g_tn", G_tn, 'NR_LANES*4')
emit("g_df", G, 'NR_LANES*4')
