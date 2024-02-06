#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M (int)1024
#define N (int)1e8
#define ERROR 1e-5

__global__ void add(float *a, float *b, float *c)
{
       int index = threadIdx.x + blockIdx.x * blockDim.x;
       if (index < N)
              c[index] = a[index] + b[index];
}

// inline void check_cuda_errors(const char *filename, const int line_number)
// {
// #ifdef DEBUG
//        cudaDeviceSynchronize();

//        cudaError_t error = cudaGetLastError();
//        if (error != cudaSuccess)
//        {
//               printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
//               exit(EXIT_FAILURE);
//        }
// #endif
// }

void random_floats(float *a)
{
       for (int i = 0; i < N; i++)
       {

              a[i] = (float)rand() / ((float)RAND_MAX / 10);
       }
}

bool check_results(float *a, float *b, float *c)
{
       for (int i = 0; i < N; i++)
       {
              if (fabs(a[i] + b[i] - c[i]) > ERROR)
              {
                     return false;
              }
       }
       return true;
}

int main()
{
       srand(time(NULL));
       float *a, *b, *c;
       int size = N * sizeof(float);

       cudaMallocManaged((void **)&a, size);
       cudaMallocManaged((void **)&b, size);
       cudaMallocManaged((void **)&c, size);

       random_floats(a);
       random_floats(b);

       cudaEvent_t start, stop;
       float time = 0;
       cudaEventCreate(&start);
       cudaEventCreate(&stop);

       cudaEventRecord(start, 0);
       add<<<(N + M - 1) / M, M>>>(a, b, c);

       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       cudaEventElapsedTime(&time, start, stop);

       printf("Elapsed time CUDA : %.7f ms ,for thread: %d\n", time, N);
       cudaEventDestroy(start);
       cudaEventDestroy(stop);

       cudaDeviceSynchronize();

       printf("check results: %s\n", check_results(a, b, c) ? "true" : "false");
       // printf("check results: %d\n", check_results(a, b, c));

       cudaFree(a);
       cudaFree(b);
       cudaFree(c);

       return 0;
}
