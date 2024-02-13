#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr int N = 256;
constexpr int BS = 32;

constexpr int BLOCK_SIZE = 32;

#include <cstdlib>
#include <iostream>
#include <math.h>
__global__ void mul_shared(float *a, float *b, float *c, int N)
{
    int i, j,
        bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y,
        ix = bx * blockDim.x + tx, iy = by * blockDim.y + ty;
    float s = 0.;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

    for (i = 0; i < N / BLOCK_SIZE; i++)
    {
        as[ty][tx] = a[(by * BLOCK_SIZE + ty) * N + i * BLOCK_SIZE + tx];
        bs[ty][tx] = b[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];

        __syncthreads();

        for (j = 0; j < BLOCK_SIZE; j++)
            s += as[ty][j] * bs[j][tx];

        __syncthreads();
    }
    c[iy * N + ix] = s;
}

__global__ void kernel(float *a, float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.;
    for (int i = 0; i < n; i++)
        s += a[row * n + i] * b[i * n + col];
    c[row * n + col] = s;
}

void random_floats(float *a, int n)
{
    for (long int i = 0; i < n; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    if (N % 32 != 0)
    {
        std::cout << "N is not correct\n";
        exit(1);
    }
    printf("N = %d\n", N);

    srand(0);
    float *d_a, *d_b, *d_c;
    float *a, *b, *c;
    float norma = 0.;
    dim3 threads(BS, BS);
    dim3 blocks(N / BS, N / BS);
    int size = N * N * sizeof(float);

    cudaEvent_t start, stop;
    float time = 0;

    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    random_floats(a, N * N);
    random_floats(b, N * N);


    double sum_time=0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Memory copy time: %.10f ms\n", time);

    sum_time+=time;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mul_shared<<<blocks, threads>>>(d_a, d_b, d_c, N);
    //  kernel <<<blocks, threads>>> (d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Computing time : %.10f ms\n", time);
    sum_time+=time;

    printf("Sum time %f: \n",sum_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "Cuda error: " << cudaGetErrorString(error) << std::endl;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            norma += c[N * i + j] * c[N * i + j];
        }
    }
    std::cout << "norma = " << sqrt(norma) << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
