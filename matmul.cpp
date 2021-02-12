

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>

const int M = 32 * 32;
const int N = 32 * 32;
const int L = 32 * 32;

const int blockSize = 16;

double mat0[M][N];
double mat1[N][L];

double out[M][L];
double out_block[M][L];

uint64_t out_time;
uint64_t out_block_time;


void mat_mul(void)
{
    uint64_t start = __rdtsc();

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < L; j++)
        {
            for (int k = 0; k < N; k++)
            {
                out[i][j] += mat0[i][k] * mat1[k][j];
            }
        }
    }

    uint64_t end = __rdtsc();
    out_time = end - start;
    printf("%ld\n", end - start);
}

void mat_mul_block(void)
{
    assert(M % blockSize == 0);
    assert(N % blockSize == 0);
    assert(L % blockSize == 0);

    uint64_t start = __rdtsc();

    for (int ii = 0; ii < (M / blockSize); ii += 1)
    {
        for (int jj = 0; jj < (L / blockSize); jj += 1)
        {
            for (int kk = 0; kk < (N / blockSize); kk += 1)
            {

                for (int i = ii * blockSize; i < (ii + 1) * blockSize; i++)
                {
                    for (int j = jj * blockSize; j < (jj + 1) * blockSize; j++)
                    {
                        for (int k = kk * blockSize; k < (kk + 1) * blockSize; k++)
                        {
                            out_block[i][j] += mat0[i][k] * mat1[k][j];
                        }
                    }
                }

            }
        }
    }

    uint64_t end = __rdtsc();
    out_block_time = end - start;
}


int main(void)
{
    double *tmp = &mat0[0][0];
    for (int i = 0; i < (M * N); i++)
    {
        *tmp++ = (double)(i + 1);
    }

    tmp = &mat1[0][0];
    for (int i = 0; i < (N * L); i++)
    {
        *tmp++ = (double)(i + 1);
    }

    mat_mul();
    mat_mul_block();

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < L; j++)
        {
            assert(out[i][j] == out_block[i][j]);
        }
    }

    printf("simple: %ld\n", out_time);
    printf("block: %ld\n", out_block_time);
    printf("%lf\n", (double)out_time / (double)out_block_time);


    /*for (int i = 0; i < (M * L); i++)
    {
        if (out[i] != out_block[i])
        {
            printf("%lf : %lf\n", out[i], out_block[i]);
        }
        assert(out[i] == out_block[i]);
    }*/

    return (0);
}
