#include <stdlib.h>
#include <openacc.h>
#include <time.h>

const int N = 4096;
double A[N][N];
double B[N][N];
double C[N][N];

int main()
{
    int size = N;
    int q = 10;
    srand(time(NULL));
    // scanf("%d",&q);
    clock_t t1, t2;
    double cgpu;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i][j] = (rand() % 256) / 256.0 - 0.5;
            B[i][j] = (rand() % 256) / 256.0 - 0.5;
        }


    t1 = clock();
#pragma acc enter data copyin(A[0 : size][0 : size], B[0 : size][0 : size])
#pragma acc enter data create(C[0 : size][0 : size])
    for (int l = 0; l < q; l++)
    {
#pragma acc parallel loop
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                C[i][j] = 0;
                for (int k = 0; k < N; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
    }
    t2 = clock();
    double s = 0.0;
#pragma acc parallel loop
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            s += C[i][j] * C[i][j];

    cgpu = t2 - t1;
/////////////////////////////////////
#pragma acc exit data copyout(C[0 : size][0 : size])
    // #pragma acc exit delete(A[0:size][0:size],B[0:size][0:size],C[0:size][0:size])

    printf("%ld iterations completed\n", q);
    printf("%.5f on HOST\n", cgpu/CLOCKS_PER_SEC);
}