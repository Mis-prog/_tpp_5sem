#include <iostream>
#include <vector>
#include <omp.h>
#include "windows.h"
#include "multiplication_matrix.h"

#define N 100

double FirstMatrix[N][N];
double SecondMatrix[N][N];
double Product[N][N];

int main() {
	SetConsoleOutputCP(CP_UTF8);

    int q = 0;
    std::cout << "Введите количество умножений: ";
    std::cin >> q;
    std::cout << std::endl;
    srand(time(NULL));

    set_rand_matrix();

    double start_time = omp_get_wtime();
    for (int p = 0; p < q; p++) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                Product[i][j] = 0;
                for (int k = 0; k < N; k++)
                    Product[i][j] += FirstMatrix[i][k] * SecondMatrix[k][j];
            }
    }
    double end_time = omp_get_wtime();
    
	double Euclid = 0.0;
    norm(Euclid);
	std::cout << "Вычисление нормы:\n";
    print(Euclid, end_time, start_time);

    start_time = omp_get_wtime();
    for (int p = 0; p < q; p++) {
	#pragma omp parallel for num_threads(4)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                Product[i][j] = 0;
                for (int k = 0; k < N; k++)
                    Product[i][j] += FirstMatrix[i][k] * SecondMatrix[k][j];
            }
    }
    end_time = omp_get_wtime();
    
	Euclid = 0.0;
    norm(Euclid);
    std::cout << "Первый алгоритм" << std::endl;
    print(Euclid,end_time,start_time);


    start_time = omp_get_wtime();
    for (int l = 0; l < q; l++) {
        for (int i = 0; i < N; i++){
    #pragma omp parallel for num_threads(4)
            for (int j = 0; j < N; j++) {
                Product[i][j] = 0;
               	for (int k = 0; k < N; k++) 
                   	Product[i][j] += FirstMatrix[i][k] * SecondMatrix[k][j];
              	}
            }
	}
    end_time = omp_get_wtime();   
    
	Euclid = 0.0;
    norm(Euclid);
    std::cout << "Второй алгоритм" << std::endl;
    print(Euclid,end_time,start_time);

    start_time = omp_get_wtime();
    for (int l = 0; l < q; l++)
    {
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++)
            {
                Product[i][j] = 0;
                double sum = 0.0;
    #pragma omp parallel for reduction(+: sum) num_threads(4)
                    for (int k = 0; k < N; k++){
                        sum += FirstMatrix[i][k] * SecondMatrix[k][j];
					}
                    Product[i][j] = sum;
    	    }
		}
    }
    end_time = omp_get_wtime();

    Euclid = 0.0;
    norm(Euclid);
    std::cout << "Третий алгоритм" << std::endl;
    print(Euclid,end_time,start_time);

	return 0;
}
void set_rand_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            FirstMatrix[i][j] = (rand() % 10) / 10.0 - 0.5;
            SecondMatrix[i][j] = (rand() % 10) / 10.0 - 0.5;
        }
    }
}
void norm(double &Euclid)
{
// #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Euclid += Product[i][j] * Product[i][j];
}
void print(double Euclid, double end_time, double start_time)
{
    std::cout << "Норма:" << Euclid << std::endl;
    std::cout << "Время: " << end_time - start_time << std::endl;
    std::cout << std::endl;
}