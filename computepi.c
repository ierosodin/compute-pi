#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include <CL/cl.h>
#include "computepi.h"

#define MAX_SOURCE_SIZE (0x100000)

char *source_str;
cl_int ret;
int workGroups;
size_t workGroupSize;
cl_context context;
cl_command_queue command_queue;
cl_kernel kernel;
cl_program program;
cl_mem mem_obj;

double compute_pi_baseline(size_t N)
{
    double pi = 0.0;
    double dt = 1.0 / N;                // dt = (b-a)/N, b = 1, a = 0
    for (size_t i = 0; i < N; i++) {
        double x = (double) i / N;      // x = ti = a+(b-a)*i/N = i/N
        pi += dt / (1.0 + x * x);       // integrate 1/(1+x^2), i = 0....N
    }
    return pi * 4.0;
}

double compute_pi_openmp(size_t N, int threads)
{
    double pi = 0.0;
    double dt = 1.0 / N;
    double x;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for private(x) reduction(+:pi)
        for (size_t i = 0; i < N; i++) {
            x = (double) i / N;
            pi += dt / (1.0 + x * x);
        }
    }
    return pi * 4.0;
}

double compute_pi_avx(size_t N)
{
    double pi = 0.0;
    double dt = 1.0 / N;
    register __m256d ymm0, ymm1, ymm2, ymm3, ymm4;
    ymm0 = _mm256_set1_pd(1.0);
    ymm1 = _mm256_set1_pd(dt);
    ymm2 = _mm256_set_pd(dt * 3, dt * 2, dt * 1, 0.0);
    ymm4 = _mm256_setzero_pd();             // sum of pi

    for (int i = 0; i <= N - 4; i += 4) {
        ymm3 = _mm256_set1_pd(i * dt);      // i*dt, i*dt, i*dt, i*dt
        ymm3 = _mm256_add_pd(ymm3, ymm2);   // x = i*dt+3*dt, i*dt+2*dt, i*dt+dt, i*dt+0.0
        ymm3 = _mm256_mul_pd(ymm3, ymm3);   // x^2 = (i*dt+3*dt)^2, (i*dt+2*dt)^2, ...
        ymm3 = _mm256_add_pd(ymm0, ymm3);   // 1+x^2 = 1+(i*dt+3*dt)^2, 1+(i*dt+2*dt)^2, ...
        ymm3 = _mm256_div_pd(ymm1, ymm3);   // dt/(1+x^2)
        ymm4 = _mm256_add_pd(ymm4, ymm3);   // pi += dt/(1+x^2)
    }
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, ymm4);             // move packed float64 values to  256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    return pi * 4.0;
}

double compute_pi_avx_unroll(size_t N)
{
    double pi = 0.0;
    double dt = 1.0 / N;
    register __m256d ymm0, ymm1, ymm2, ymm3, ymm4,
             ymm5, ymm6, ymm7, ymm8, ymm9,
             ymm10,ymm11, ymm12, ymm13, ymm14;
    ymm0 = _mm256_set1_pd(1.0);
    ymm1 = _mm256_set1_pd(dt);
    ymm2 = _mm256_set_pd(dt * 3, dt * 2, dt * 1, 0.0);
    ymm3 = _mm256_set_pd(dt * 7, dt * 6, dt * 5, dt * 4);
    ymm4 = _mm256_set_pd(dt * 11, dt * 10, dt * 9, dt * 8);
    ymm5 = _mm256_set_pd(dt * 15, dt * 14, dt * 13, dt * 12);
    ymm6 = _mm256_setzero_pd();             // first sum of pi
    ymm7 = _mm256_setzero_pd();             // second sum of pi
    ymm8 = _mm256_setzero_pd();             // third sum of pi
    ymm9 = _mm256_setzero_pd();             // fourth sum of pi

    for (int i = 0; i <= N - 16; i += 16) {
        ymm14 = _mm256_set1_pd(i * dt);

        ymm10 = _mm256_add_pd(ymm14, ymm2);
        ymm11 = _mm256_add_pd(ymm14, ymm3);
        ymm12 = _mm256_add_pd(ymm14, ymm4);
        ymm13 = _mm256_add_pd(ymm14, ymm5);

        ymm10 = _mm256_mul_pd(ymm10, ymm10);
        ymm11 = _mm256_mul_pd(ymm11, ymm11);
        ymm12 = _mm256_mul_pd(ymm12, ymm12);
        ymm13 = _mm256_mul_pd(ymm13, ymm13);

        ymm10 = _mm256_add_pd(ymm0, ymm10);
        ymm11 = _mm256_add_pd(ymm0, ymm11);
        ymm12 = _mm256_add_pd(ymm0, ymm12);
        ymm13 = _mm256_add_pd(ymm0, ymm13);

        ymm10 = _mm256_div_pd(ymm1, ymm10);
        ymm11 = _mm256_div_pd(ymm1, ymm11);
        ymm12 = _mm256_div_pd(ymm1, ymm12);
        ymm13 = _mm256_div_pd(ymm1, ymm13);

        ymm6 = _mm256_add_pd(ymm6, ymm10);
        ymm7 = _mm256_add_pd(ymm7, ymm11);
        ymm8 = _mm256_add_pd(ymm8, ymm12);
        ymm9 = _mm256_add_pd(ymm9, ymm13);
    }

    double tmp1[4] __attribute__((aligned(32)));
    double tmp2[4] __attribute__((aligned(32)));
    double tmp3[4] __attribute__((aligned(32)));
    double tmp4[4] __attribute__((aligned(32)));

    _mm256_store_pd(tmp1, ymm6);
    _mm256_store_pd(tmp2, ymm7);
    _mm256_store_pd(tmp3, ymm8);
    _mm256_store_pd(tmp4, ymm9);

    pi += tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] +
          tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] +
          tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] +
          tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

    ymm6 = _mm256_setzero_pd();
    ymm7 = _mm256_setzero_pd();
    ymm8 = _mm256_setzero_pd();
    ymm9 = _mm256_setzero_pd();

    ymm2 = _mm256_set_pd((N-4) * dt, (N-3) * dt, (N-2) * dt, (N-1) * dt);
    ymm3 = _mm256_set_pd((N-8) * dt, (N-7) * dt, (N-6) * dt, (N-5) * dt);
    ymm4 = _mm256_set_pd((N-12) * dt, (N-11) * dt, (N-10) * dt, (N-9) * dt);
    ymm5 = _mm256_set_pd(0.0, (N-15) * dt, (N-14) * dt, (N-13) * dt);

    ymm10 = _mm256_mul_pd(ymm2, ymm2);
    ymm11 = _mm256_mul_pd(ymm3, ymm3);
    ymm12 = _mm256_mul_pd(ymm4, ymm4);
    ymm13 = _mm256_mul_pd(ymm5, ymm5);

    ymm10 = _mm256_add_pd(ymm0, ymm10);
    ymm11 = _mm256_add_pd(ymm0, ymm11);
    ymm12 = _mm256_add_pd(ymm0, ymm12);
    ymm13 = _mm256_add_pd(ymm0, ymm13);

    ymm10 = _mm256_div_pd(ymm1, ymm10);
    ymm11 = _mm256_div_pd(ymm1, ymm11);
    ymm12 = _mm256_div_pd(ymm1, ymm12);
    ymm13 = _mm256_div_pd(ymm1, ymm13);

    ymm6 = _mm256_add_pd(ymm6, ymm10);
    ymm7 = _mm256_add_pd(ymm7, ymm11);
    ymm8 = _mm256_add_pd(ymm8, ymm12);
    ymm9 = _mm256_add_pd(ymm9, ymm13);

    double tmp7[4] __attribute__((aligned(32)));
    double tmp8[4] __attribute__((aligned(32)));
    double tmp9[4] __attribute__((aligned(32)));
    double tmp10[4] __attribute__((aligned(32)));

    _mm256_store_pd(tmp7, ymm6);
    _mm256_store_pd(tmp8, ymm7);
    _mm256_store_pd(tmp9, ymm8);
    _mm256_store_pd(tmp10, ymm9);

    if (N%16 != 0)
        if (N%16 <= 4)
            for (int k = 0; k < N%16; k++)
                pi += tmp7[k];
        else if (N%16 <= 8) {
            pi += tmp7[0] + tmp7[1] + tmp7[2] + tmp7[3];
            for (int k = 0;  k < N%16 - 4; k++)
                pi += tmp8[k];
        } else if (N%16 <= 12) {
            pi += tmp7[0] + tmp7[1] + tmp7[2] + tmp7[3];
            pi += tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3];
            for (int k = 0;  k < N%16 - 8; k++)
                pi += tmp9[k];
        } else if (N%16 < 16) {
            pi += tmp7[0] + tmp7[1] + tmp7[2] + tmp7[3];
            pi += tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3];
            pi +=+ tmp9[0] + tmp9[1] + tmp9[2] + tmp9[3];
            for (int k = 0;  k < N%16 - 12; k++)
                pi += tmp10[k];
        }
    return pi * 4.0;
}

void init_opencl(size_t N, size_t chunks)
{
    // Load the kernel source code into the array source_str
    FILE *fp;
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
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
                          &device_id, &ret_num_devices);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, NULL);
    workGroups = ceil(N/workGroupSize/chunks);
    // Create an OpenCL context
    context = clCreateContext( 0, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, NULL);

    // Build the program
    ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "Pi", &ret);
    // Create memory buffers on the device for each vector
    mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             workGroups * sizeof(float), NULL, &ret);

    // Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj);
    clSetKernelArg(kernel, 1, sizeof(float)*(workGroupSize), NULL);
    clSetKernelArg(kernel, 2, sizeof(uint), &N);
    clSetKernelArg(kernel, 3, sizeof(uint), &chunks);
}

void release_opencl()
{
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(source_str);
}

double compute_pi_opencl(size_t N, size_t chunks)
{
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
    for (int i = 0; i < workGroups; i++) {
        pi += pi_partical[i];
    }
    pi *= (1.0/(float)N);

    return pi;
}
