#include <time.h>
#include <math.h>
#include <openacc.h>

#define _USE_MATH_DEFINES

const long long int N = 150000000;

int main()
{
    double sum = 0;
    double start_time = clock();
    for (long long int i = 1; i <= N; i++)
    {
        sum += (pow(i, (1.0 / 3)) / ((i + 1) * sqrt(i)));
    }
    double end_time = clock();
    printf("HOST sum: %f, time: %f\n", sum, (end_time - start_time)/CLOCKS_PER_SEC);

    start_time = 0;
    end_time = 0;
    
    start_time = clock();
    sum = 0;
#pragma acc kernels
    for (long long int i = 1; i <= N; i++)
    {
        sum += (pow(i, (1.0 / 3)) / ((i + 1) * sqrt(i)));
    }
    end_time = clock();

    printf("GPU sum: %f, time: %f\n", sum, (end_time - start_time)/CLOCKS_PER_SEC);

    return 0;
}