

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
//#include <x86intrin.h>
#include <intrin.h>

const int M = 2 * 1024;
const int N = 2 * 1024;
const int L = 2 * 1024;

const int blockSize = 8;

uint64_t out_time;
uint64_t out_block_time;


void mat_mul(void)
{
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
		mat_out[i] = 0.0;
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

    out_time = end.QuadPart - start.QuadPart;
}

void mat_mul_block(void)
{
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
		mat_out[i] = 0.0;
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

	out_block_time = end.QuadPart - start.QuadPart;
}


int main(void)
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

    //mat_mul();
    mat_mul_block();

    //for (int i = 0; i < M; i++)
    //{
    //    for (int j = 0; j < L; j++)
    //    {
    //        //assert(out[i][j] == out_block[i][j]);
    //        if (out[i][j] != out_block[i][j])
    //        {
    //            printf("%lf\n", fabs(out[i][j] - out_block[i][j]));
    //        }
    //    }
    //}

    //printf("simple: %lf\n", (double)out_time / (double)frequency.QuadPart);
    printf("block: %lf\n", (double)out_block_time / (double)frequency.QuadPart);
    //printf("%lf\n", (double)out_time / (double)out_block_time);


    return (0);
}
