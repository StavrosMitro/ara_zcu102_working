//
// File:        utils.c
// Description: Shared element-wise helpers (Ara / RVV), see utils.h.
//
// Every routine strip-mines with vsetvli, so any length works without padding.
// LMUL=8 (m8) is used throughout for maximum memory bandwidth; each function
// sets its own vtype, so they are safe to call from any context.
//
#include "utils.h"

void memset_vectorized_zero_f32(float *dst, size_t n_elements)
{
    // Broadcast zeros once at max VL, then drain to memory at full bandwidth.
    //
    // The splat MUST run at VLMAX. `vsetvli zero, zero, ...` (rd=x0 AND rs1=x0)
    // means "keep the current vl, change vtype only" -- it does NOT select VLMAX.
    // The splat would then fill only however many elements some earlier,
    // unrelated vector op left in vl, and the store loop below (which does set
    // vl up to VLMAX) would write GARBAGE past that point instead of zeros.
    // Writing a real rd with rs1=x0 requests AVL=~0, i.e. vl = VLMAX.
    size_t vlmax;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax));
    asm volatile("vmv.v.i v16, 0");

    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst) : "memory");
        dst += vl;
        n_elements -= vl;
    }
}

void memset_vectorized_value_f32(float *dst, float value, size_t n_elements)
{
    // Broadcast the scalar once; the splat stays valid across the loop.
    // rd != x0 with rs1 = x0 -> AVL = ~0 -> vl = VLMAX (see the note in
    // memset_vectorized_zero_f32: `vsetvli zero, zero` keeps the stale vl).
    size_t vlmax;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax));
    asm volatile("vfmv.v.f v16, %0" :: "f"(value));

    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vse32.v v16, (%0)" :: "r"(dst) : "memory");
        dst += vl;
        n_elements -= vl;
    }
}

void memcpy_vectorized_f32(float *dst, const float *src, size_t n_elements)
{
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vle32.v v16, (%0)" :: "r"(src) : "memory");
        asm volatile("vse32.v v16, (%0)" :: "r"(dst) : "memory");
        src += vl;
        dst += vl;
        n_elements -= vl;
    }
}

void zero_f32_vec(float *buf, int n)
{
    // rd != x0 with rs1 = x0 -> vl = VLMAX, so the whole v8..v15 group is zeroed
    // (see the note in memset_vectorized_zero_f32).
    size_t vlmax;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vlmax));
    asm volatile("vmv.v.i v8, 0");

    float *ptr = buf;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vse32.v v8, (%0)" :: "r"(ptr) : "memory");
        ptr += vl;
        n -= (int)vl;
    }
}

void vector_scale_f32(float *vec, float scale, int length)
{
    int n = length;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v8, (%0)" :: "r"(vec) : "memory");
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(scale));
        asm volatile("vse32.v v8, (%0)" :: "r"(vec) : "memory");
        vec += vl;
        n -= (int)vl;
    }
}

void vec_add_f32(float *dst, const float *src, size_t n_elements)
{
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vle32.v v8,  (%0)" :: "r"(dst) : "memory");
        asm volatile("vle32.v v16, (%0)" :: "r"(src) : "memory");
        asm volatile("vfadd.vv v8, v8, v16");
        asm volatile("vse32.v v8, (%0)" :: "r"(dst) : "memory");
        dst += vl;
        src += vl;
        n_elements -= vl;
    }
}

void vec_affine_f32(float *dst, float a, float b, size_t n_elements)
{
    while (n_elements > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n_elements));
        asm volatile("vle32.v v8, (%0)" :: "r"(dst) : "memory");
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(a));
        asm volatile("vfadd.vf v8, v8, %0" :: "f"(b));
        asm volatile("vse32.v v8, (%0)" :: "r"(dst) : "memory");
        dst += vl;
        n_elements -= vl;
    }
}
