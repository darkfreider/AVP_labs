

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
//#include <x86intrin.h>
#include <intrin.h>

#define SEED (7)
uint32_t g_seed;
uint32_t xorshift32(void)
{
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	uint32_t x = g_seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return (g_seed = x);
}

double get_random_double(void)
{
	double arr[6] = {
		1.0, 2.0,
		3.0, 4.0, 
		5.0, 6.0
	};

	return arr[xorshift32() % 7];
}

const int M = 1 * 1024;
const int N = 1 * 1024;
const int L = 1 * 1024;

const int blockSize = 8;

uint64_t g_out_time;
uint64_t g_out_block_time;
uint64_t g_out_block_simd_time;

double *g_mat_mul_result;
double *g_mat_mul_block_result;
double *g_mat_mul_block_simd_result;

void mat_mul(void)
{
	g_seed = SEED;

    double *mat0 = (double *)malloc(sizeof(double) * M * N);
    double *mat1 = (double *)malloc(sizeof(double) * N * L);
	double *mat_out = (double *)malloc(sizeof(double) * M * L);
    
	for (int i = 0; i < (M * N); i++)
    {
        mat0[i] = (double)(i + 1);
    }
    for (int i = 0; i < (N * L); i++)
    {
        mat1[i] = (double)(i + 1);
    }
	for (int i = 0; i < (M * L); i++)
	{
		mat_out[i] = get_random_double();
	}

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);
    
	for (int i = 0; i < M; i++)
    {
		for (int j = 0; j < L; j++)
        {
			for (int k = 0; k < N; k++)
            {
				mat_out[i * L + j] += mat0[i * N + k] * mat1[k * L + j];
            }
        }
    }

	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	g_mat_mul_result = mat_out;
    g_out_time = end.QuadPart - start.QuadPart;
}

void mat_mul_block(void)
{
	g_seed = SEED;

    assert(M % blockSize == 0);
    assert(N % blockSize == 0);
    assert(L % blockSize == 0);

	double *mat0 = (double *)malloc(sizeof(double) * M * N);
	double *mat1 = (double *)malloc(sizeof(double) * N * L);
	double *mat_out = (double *)malloc(sizeof(double) * M * L);

	for (int i = 0; i < (M * N); i++)
	{
		mat0[i] = (double)(i + 1);
	}
	for (int i = 0; i < (N * L); i++)
	{
		mat1[i] = (double)(i + 1);
	}
	for (int i = 0; i < (M * L); i++)
	{
		mat_out[i] = get_random_double();
	}

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);

    for (int ii = 0; ii < (M / blockSize); ii += 1)
    {
		for (int kk = 0; kk < (N / blockSize); kk += 1)
        {
			for (int jj = 0; jj < (L / blockSize); jj += 1)
            {

                for (int i = ii * blockSize; i < (ii + 1) * blockSize; i++)
                {
					for (int k = kk * blockSize; k < (kk + 1) * blockSize; k++)
                    {
						for (int j = jj * blockSize; j < (jj + 1) * blockSize; j++)
                        {
							mat_out[i * L + j] += mat0[i * N + k] * mat1[k * L + j];
                        }
                    }
                }

            }
        }
    }

	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	g_mat_mul_block_result = mat_out;
	g_out_block_time = end.QuadPart - start.QuadPart;
}

void mat_mul_block_simd(void)
{
	g_seed = SEED;

	assert(M % blockSize == 0);
	assert(N % blockSize == 0);
	assert(L % blockSize == 0);

	double *mat0 = (double *)malloc(sizeof(double) * M * N);
	double *mat1 = (double *)malloc(sizeof(double) * N * L);
	double *mat_out = (double *)malloc(sizeof(double) * M * L);

	for (int i = 0; i < (M * N); i++)
	{
		mat0[i] = (double)(i + 1);
	}
	for (int i = 0; i < (N * L); i++)
	{
		mat1[i] = (double)(i + 1);
	}
	for (int i = 0; i < (M * L); i++)
	{
		mat_out[i] = get_random_double();
	}

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);

	for (int ii = 0; ii < (M / blockSize); ii += 1)
	{
		for (int kk = 0; kk < (N / blockSize); kk += 1)
		{
			for (int jj = 0; jj < (L / blockSize); jj += 1)
			{

				int j = jj * blockSize;

				for (int i = ii * blockSize; i < (ii + 1) * blockSize; i++)
				{
					__m256d sum0 = _mm256_setzero_pd();
					__m256d sum1 = _mm256_setzero_pd();

					for (int k = kk * blockSize; k < (kk + 1) * blockSize; k++)
					{
						__m256d r = _mm256_set1_pd(mat0[i * N + k]);

						__m256d t0 = _mm256_load_pd(&mat1[k * L + j + 0]);
						__m256d t1 = _mm256_load_pd(&mat1[k * L + j + 4]);

						sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(r, t0));
						sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(r, t1));
					}

					__m256d m0 = _mm256_load_pd(&mat_out[i * L + j + 0]);
					__m256d m1 = _mm256_load_pd(&mat_out[i * L + j + 4]);

					m0 = _mm256_add_pd(m0, sum0);
					m1 = _mm256_add_pd(m1, sum1);

					_mm256_store_pd(&mat_out[i * L + j + 0], m0);
					_mm256_store_pd(&mat_out[i * L + j + 4], m1);
				}
			
			}
		}
	}

	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	g_mat_mul_block_simd_result = mat_out;
	g_out_block_simd_time = end.QuadPart - start.QuadPart;
}

int main(void)
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

    mat_mul();
	mat_mul_block();
	mat_mul_block_simd();

	printf("simple: %lf\n", (double)g_out_time / (double)frequency.QuadPart);
	printf("block: %lf\n", (double)g_out_block_time / (double)frequency.QuadPart);
	printf("simd block: %lf\n", (double)g_out_block_simd_time / (double)frequency.QuadPart);

	for (int i = 0; i < (M * L); i++)
	{
		double ratio = (g_mat_mul_block_result[i] / g_mat_mul_block_simd_result[i]);
		
		if (fabs(1.0 - ratio) > 0.01)
		{
			printf("ERROR! :: %lf, %lf\n", g_mat_mul_block_result[i], g_mat_mul_block_simd_result[i]);
			return (-1);
		}
	}

    return (0);
}
