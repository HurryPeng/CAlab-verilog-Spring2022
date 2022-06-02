// matmul.c
// HurryPeng
// 2022.5.26

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <memory.h>
#include <immintrin.h>

const int N = 1 << 10;

void rand_init(float * a)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            a[N * i + j] = rand() % 64;
        }
    }
}

void print_mat(float * a)
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

void gemm_baseline(const float * a, const float * b, float * c)
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

bool gemm_verify(const float * c, const float * d)
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

void gemm_avx(const float * a, const float * b, float * c)
{
    memset(c, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            __m256 vecA = _mm256_set1_ps(a[N * i + j]);
            for (int k = 0; k < N; k += 8)
            {
                __m256 vecB = _mm256_loadu_ps(&b[N * j + k]);
                __m256 vecC = _mm256_loadu_ps(&c[N * i + k]);
                vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                _mm256_storeu_ps(&c[N * i + k], vecC);
            }
        }
    }
}

void gemm_avx_block(const float * a, const float * b, float * c, int blockL)
{
    int blockN = N / blockL;
    memset(c, 0, N * N * sizeof(float));
    for (int blkI = 0; blkI < blockN; ++blkI)
    {
        int blkIBase = blkI * blockL;
        for (int blkJ = 0; blkJ < blockN; ++blkJ)
        {
            int blkJBase = blkJ * blockL;
            for (int blkK = 0; blkK < blockN; ++blkK)
            {
                int blkKBase = blkK * blockL;
                for (int i = blkIBase; i < blkIBase + blockL; ++i)
                {
                    for (int j = blkJBase; j < blkJBase + blockL; ++j)
                    {
                        __m256 vecA = _mm256_set1_ps(a[N * i + j]);
                        for (int k = blkKBase; k < blkKBase + blockL; k += 8)
                        {
                            __m256 vecB = _mm256_loadu_ps(&b[N * j + k]);
                            __m256 vecC = _mm256_loadu_ps(&c[N * i + k]);
                            vecC = _mm256_fmadd_ps(vecA, vecB, vecC);
                            _mm256_storeu_ps(&c[N * i + k], vecC);
                        }
                    }
                }
            }
        }
    }
    // free(bt);
}

int main(void)
{
    float * a = calloc(N * N, sizeof(float));
    float * b = calloc(N * N, sizeof(float));
    float * c = calloc(N * N, sizeof(float));
    float * d = calloc(N * N, sizeof(float));

    rand_init(a);
    rand_init(b);

    {
        clock_t start = clock();
        gemm_baseline(a, b, d);
        clock_t end = clock();
        printf("Baseline: %d %f\n", true, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    {
        clock_t start = clock();
        gemm_avx(a, b, c);
        clock_t end = clock();
        bool correct = gemm_verify(c, d);
        printf("AVX: %d %f\n", correct, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    {
        clock_t start = clock();
        gemm_avx_block(a, b, c, 1 << 6);
        clock_t end = clock();
        bool correct = gemm_verify(c, d);
        printf("AVX-Block: %d %f\n", correct, (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    }

    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
}
