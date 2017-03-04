#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "computepi.h"

#define CLOCK_ID CLOCK_MONOTONIC_RAW
#define ONE_SEC 1000000000.0

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char const *argv[])
{
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};

    if (argc < 2) return -1;
    int N = atoi(argv[1]);
    int chunks = 1;
    if (argc == 3)
        chunks = atoi(argv[2]);

    int i, loop = 100, num;
    int workGroups;

    size_t workGroupSize;

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
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("pi_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
                          &device_id, &ret_num_devices);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, NULL);
    workGroups = ceil(N/workGroupSize/chunks);
    // Create an OpenCL context
    cl_context context = clCreateContext( 0, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, NULL);

    // Build the program
    ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "Pi", &ret);
    // Create memory buffers on the device for each vector
    cl_mem mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    workGroups * sizeof(float), NULL, &ret);

    // Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj);
    clSetKernelArg(kernel, 1, sizeof(float)*(workGroupSize), NULL);
    clSetKernelArg(kernel, 2, sizeof(uint), &N);
    clSetKernelArg(kernel, 3, sizeof(uint), &chunks);

    clock_gettime(CLOCK_ID, &start);
    // Launch Kernel
    size_t globalWorkSize = N/chunks;
    size_t localWorkSize = workGroupSize;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);

    // Read the memory buffer on the device to the local variable
    float pi = 0;
    float *pi_partical = (float *)malloc(sizeof(float)*workGroups);

    ret = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0,
                              sizeof(float)*workGroups, pi_partical, 0, NULL, NULL);
    for (i = 0; i < workGroups; i++) {
        pi += pi_partical[i];
    }
    pi *= (1.0/(float)N);

    clock_gettime(CLOCK_ID, &end);

    printf("%lf\n", (double) (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec)/ONE_SEC);
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(source_str);
    return 0;
}
