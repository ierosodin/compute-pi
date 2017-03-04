#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "computepi.h"

#define CLOCK_ID CLOCK_MONOTONIC_RAW
#define ONE_SEC 1000000000.0

int main(int argc, char const *argv[])
{
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};

    if (argc < 2) return -1;

    int N = atoi(argv[1]);
    int i, loop = 100, num;

    double data[loop];
    double sum_d = 0.0, sum_p = 0.0, average = 0.0, s = 0.0;

    // OpenMP with 1 threads
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 1);
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
    printf("1 %lf\n", (double) sum_d/num);

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
    printf("2 %lf\n", (double) sum_d/num);

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
    printf("4 %lf\n", (double) sum_d/num);

    // OpenMP with 8 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 8);
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
    printf("8 %lf\n", (double) sum_d/num);

    // OpenMP with 12 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 12);
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
    printf("12 %lf\n", (double) sum_d/num);

    // OpenMP with 16 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 16);
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
    printf("16 %lf\n", (double) sum_d/num);

    // OpenMP with 24 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 24);
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
    printf("24 %lf\n", (double) sum_d/num);

    // OpenMP with 48 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 48);
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
    printf("48 %lf\n", (double) sum_d/num);

    // OpenMP with 64 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 64);
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
    printf("64 %lf\n", (double) sum_d/num);

    // OpenMP with 128 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 128);
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
    printf("128 %lf\n", (double) sum_d/num);

    // OpenMP with 256 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 256);
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
    printf("256 %lf\n", (double) sum_d/num);

    // OpenMP with 512 threads
    sum_d = 0.0, sum_p = 0.0;
    for (i = 0; i < loop; i++) {
        clock_gettime(CLOCK_ID, &start);
        compute_pi_openmp(N, 512);
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
    printf("512 %lf\n", (double) sum_d/num);

    return 0;
}
