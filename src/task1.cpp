#include <iostream>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <random>
#include <omp.h>
#define n 256

using namespace std;    

int main()
{
    double *matrix1 = new double[n * n];
    double *matrix2 = new double[n * n];
    double *matrix3 = new double[n * n];

    cout << "N = " << n << endl;

    int q = 1;

    srand(time(NULL));

    for (int i = 0; i < n * n; i++)
    {
        matrix1[i] = (double)rand() / RAND_MAX - 0.5;
        matrix2[i] = (double)rand() / RAND_MAX - 0.5;
    }

    double sum = 0;
    double start_time = omp_get_wtime(); 

    for (int count = 0; count < q; count++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                sum = 0;
  #pragma omp parallel for reduction(+ : sum) num_threads(4)
                for (int k = 0; k < n; k++)
                { 
                    sum += matrix1[i * n + k] * matrix2[j + n * k];
                }
                matrix3[i * n + j] = sum;
            }
        }
    }

    double result = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result = 0;
            result += matrix3[i * n + j] * matrix3[i * n + j];
        }
    }
    double end_time =omp_get_wtime(); 
    cout << "Program execution time = " << (end_time - start_time)*1000 << endl;
    cout << "Norm = " << sqrt(result) << endl;

    free(matrix1);
    free(matrix2);
    free(matrix3);
    return 0;
}