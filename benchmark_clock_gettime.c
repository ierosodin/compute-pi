#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "computepi.h"

#define CLOCK_ID CLOCK_MONOTONIC_RAW
#define ONE_SEC 1000000000.0

int main(int argc, char const *argv[])
{
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};

    if (argc < 2) {
        printf("\nUsage: ./benchmark_clock_gettime -n 1024 [OPTIONS].\n");
        printf("\nOptions:\n   -c    Argument for OpenCL to determine how many work for each work item.\n");
        return -1;
    }

    int N = 1024, chunks = 1;
    int args;

    while (-1 != (args = getopt(argc, argv, "n:c:"))) {
        switch (args) {
        case 'n':
            N = atoi(optarg);
            break;
        case 'c':
            chunks = atoi(optarg);
            break;
        }
    }

    int i, loop = 100, num;

    double data[loop];
    double sum_d = 0.0, sum_p = 0.0, average = 0.0, s = 0.0;

    // Baseline
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_baseline(N);
        clock_gettime(CLOCK_ID, &end);

        data[i] = (end.tv_sec - start.tv_sec) +
                  (end.tv_nsec - start.tv_nsec)/ONE_SEC;
        sum_d += data[i];
        sum_p += pow(data[i], 2);
    }

    average = sum_d/loop;
    s = sqrt(sum_p/loop - pow(average, 2));
    sum_d = 0.0, num = loop;

    for (i = 0; i < loop; i++) {
        if (data[i] > average - 1.645*s && data[i] < average + 1.645*s)
            sum_d += data[i];
        else
            num--;
    }
    printf("%lf ", (double) sum_d/num);

    // OpenMP with 2 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 2);
        clock_gettime(CLOCK_ID, &end);

        data[i] = (end.tv_sec - start.tv_sec) +
                  (end.tv_nsec - start.tv_nsec)/ONE_SEC;
        sum_d += data[i];
        sum_p += pow(data[i], 2);
    }

    average = sum_d/loop;
    s = sqrt(sum_p/loop - pow(average, 2));
    sum_d = 0.0, num = loop;

    for (i = 0; i < loop; i++) {
        if (data[i] > average - 1.645*s && data[i] < average + 1.645*s)
            sum_d += data[i];
        else
            num--;
    }
    printf("%lf ", (double) sum_d/num);

    // OpenMP with 4 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 4);
        clock_gettime(CLOCK_ID, &end);

        data[i] = (end.tv_sec - start.tv_sec) +
                  (end.tv_nsec - start.tv_nsec)/ONE_SEC;
        sum_d += data[i];
        sum_p += pow(data[i], 2);
    }

    average = sum_d/loop;
    s = sqrt(sum_p/loop - pow(average, 2));
    sum_d = 0.0, num = loop;

    for (i = 0; i < loop; i++) {
        if (data[i] > average - 1.645*s && data[i] < average + 1.645*s)
            sum_d += data[i];
        else
            num--;
    }
    printf("%lf ", (double) sum_d/num);

    // AVX SIMD
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_avx(N);
        clock_gettime(CLOCK_ID, &end);

        data[i] = (end.tv_sec - start.tv_sec) +
                  (end.tv_nsec - start.tv_nsec)/ONE_SEC;
        sum_d += data[i];
        sum_p += pow(data[i], 2);
    }

    average = sum_d/loop;
    s = sqrt(sum_p/loop - pow(average, 2));
    sum_d = 0.0, num = loop;

    for (i = 0; i < loop; i++) {
        if (data[i] > average - 1.645*s && data[i] < average + 1.645*s)
            sum_d += data[i];
        else
            num--;
    }
    printf("%lf ", (double) sum_d/num);

    // AVX SIMD + Loop unrolling
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_avx_unroll(N);
        clock_gettime(CLOCK_ID, &end);

        data[i] = (end.tv_sec - start.tv_sec) +
                  (end.tv_nsec - start.tv_nsec)/ONE_SEC;
        sum_d += data[i];
        sum_p += pow(data[i], 2);
    }

    average = sum_d/loop;
    s = sqrt(sum_p/loop - pow(average, 2));
    sum_d = 0.0, num = loop;

    for (i = 0; i < loop; i++) {
        if (data[i] > average - 1.645*s && data[i] < average + 1.645*s)
            sum_d += data[i];
        else
            num--;
    }
    printf("%lf ", (double) sum_d/num);

    // OpenCL
    init_opencl(N, chunks);
    clock_gettime(CLOCK_ID, &start);
    compute_pi_opencl(N, chunks);
    clock_gettime(CLOCK_ID, &end);
    release_opencl();
    printf("%lf\n", (double) (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec)/ONE_SEC);
    return 0;
}
