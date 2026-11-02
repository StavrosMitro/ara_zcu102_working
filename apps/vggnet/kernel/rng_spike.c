//
// File:        rng_spike.c
// Description: Minimal rand()/srand() for the bare-metal Spike build.
//
// The Spike link is -nostdlib, so newlib's rand/srand are unavailable. The
// training path genuinely needs a PRNG (dropout masks, dataset shuffle, weight
// init, augmentation), so provide a small deterministic LCG. Guarded by SPIKE so
// the native (newlib) build keeps the standard-library versions.
//
#ifdef SPIKE

// glibc TYPE_0 minimal-standard-style LCG, returning a 31-bit value in
// [0, RAND_MAX] where RAND_MAX == 0x7fffffff (matches newlib's <stdlib.h>).
static unsigned long long __rng_state = 1ULL;

void srand(unsigned int seed)
{
    __rng_state = seed ? (unsigned long long)seed : 1ULL;
}

int rand(void)
{
    __rng_state = __rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((__rng_state >> 33) & 0x7fffffffULL);
}

#endif // SPIKE
