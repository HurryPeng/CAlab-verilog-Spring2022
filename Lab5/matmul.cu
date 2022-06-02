// matmul.cu
// HurryPeng
// 2022.5.26

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

const int N = 1 << 10;
const int BLOCK_SIZE = 1 << 5;

__host__ void rand_init(float * a)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            a[N * i + j] = rand() % 64;
        }
    }
}

__host__ void print_mat(float * a)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%f\t", a[N * i + j]);
        }
        printf("\n");
    }
}

__host__ void gemm_baseline(const float * a, const float * b, float * c)
{
    memset(c, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < N; ++k)
            {
                c[N * i + j] += a[N * i + k] * b[N * k + j];
            }
        }
    }
}

__host__ bool gemm_verify(const float * c, const float * d)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (fabs(d[N * i + j] - c[N * i + j]) > 1e-5)
            {
                return false;
            }
        }
    }
    return true;
}

__global__ void gemm_gpu_impl(float * a, float * b, float * c)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= N * N) return;
    int i = threadId / N;
    int j = threadId % N;

	c[threadId] = 0.0f;
	for (int k = 0; k < N; ++k)
	{
		c[threadId] += a[N * i + k] * b[N * k + j];
	}
}

__host__ void gemm_gpu(float * a, float * b, float * c)
{
    float * cudaA;
    float * cudaB;
    float * cudaC;
    cudaMalloc((void**)&cudaA, N * N * sizeof(float));
	cudaMalloc((void**)&cudaB, N * N * sizeof(float));
	cudaMalloc((void**)&cudaC, N * N * sizeof(float));

    cudaMemcpy(cudaA, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    gemm_gpu_impl<<<N * N / BLOCK_SIZE, BLOCK_SIZE>>>(cudaA, cudaB, cudaC);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cudaC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);
}

template<int blockL>
__global__ void gemm_gpu_block_impl(float * a, float * b, float * c)
{
    int blockN = N / blockL;

    // x and y cannot be swapped! Huge influence on performance! 
    int blkI = blockIdx.y;
    int blkJ = blockIdx.x;
    int subI = threadIdx.y;
    int subJ = threadIdx.x;
    int i = blockL * blkI + subI;
    int j = blockL * blkJ + subJ;
    int cId = N * i + j;
    if (cId >= N * N) return;

    __shared__ float aSub[blockL][blockL];
    __shared__ float bSub[blockL][blockL];

	float localSum = 0.0f;
    for (int blkK = 0; blkK < blockN; ++blkK)
    {
        int kBase = blockL * blkK;

        aSub[subI][subJ] = a[N * i + (kBase + subJ)];
        bSub[subI][subJ] = b[N * (kBase + subI) + j];
        __syncthreads();

        for (int subK = 0; subK < blockL; ++subK)
        {
            localSum += aSub[subI][subK] * bSub[subK][subJ];
        }
        __syncthreads();
    }

	c[cId] = localSum;
}

template<int blockL>
__host__ void gemm_gpu_block(float * a, float * b, float * c)
{
    int blockN = N / blockL;
	dim3 grid(blockN, blockN);
    dim3 block(blockL, blockL);
    
    float * cudaA;
    float * cudaB;
    float * cudaC;
    cudaMalloc((void**)&cudaA, N * N * sizeof(float));
	cudaMalloc((void**)&cudaB, N * N * sizeof(float));
	cudaMalloc((void**)&cudaC, N * N * sizeof(float));

    cudaMemcpy(cudaA, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    gemm_gpu_block_impl<blockL><<<grid, block>>>(cudaA, cudaB, cudaC);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cudaC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);
}

int main(void)
{
    float * a = (float *) calloc(N * N, sizeof(float));
    float * b = (float *) calloc(N * N, sizeof(float));
    float * c = (float *) calloc(N * N, sizeof(float));
    float * d = (float *) calloc(N * N, sizeof(float));

    rand_init(a);
    rand_init(b);

    {
        clock_t start = clock();
        gemm_baseline(a, b, d);
        clock_t end = clock();
        printf("CPU: %d %f\n", true, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    {
        clock_t start = clock();
        gemm_gpu(a, b, c);
        clock_t end = clock();
        bool correct = gemm_verify(c, d);
        printf("GPU: %d %f\n", correct, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    {
        clock_t start = clock();
        gemm_gpu_block<BLOCK_SIZE>(a, b, c);
        clock_t end = clock();
        bool correct = gemm_verify(c, d);
        printf("GPU-Block: %d %f\n", correct, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
}
