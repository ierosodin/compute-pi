#include <stdio.h>
#include <stdlib.h>
#include "computepi.h"
#include <math.h>

int main(int argc, char const *argv[])
{

    double real = acos(-1);
    double compute, error;

    if (argc < 2) return -1;

    int N = atoi(argv[1]);

    // Baseline
    compute = compute_pi_baseline(N);
    if(real > compute)
        error = real - compute;
    else
        error = compute - real;

    printf("%lf ", error);

    // OpenMP with 2 threads
    compute = compute_pi_openmp(N, 2);
    if(real > compute)
        error = real - compute;
    else
        error = compute - real;

    printf("%lf ", error);

    // OpenMP with 4 threads
    compute = compute_pi_openmp(N, 4);
    if(real > compute)
        error = real - compute;
    else
        error = compute - real;

    printf("%lf ", error);

    // AVX SIMD
    compute = compute_pi_avx(N);
    if(real > compute)
        error = real - compute;
    else
        error = compute - real;

    printf("%lf ", error);

    // AVX SIMD + Loop unrolling
    compute = compute_pi_avx_unroll(N);
    if(real > compute)
        error = real - compute;
    else
        error = compute - real;

    printf("%lf\n", error);

    return 0;
}
