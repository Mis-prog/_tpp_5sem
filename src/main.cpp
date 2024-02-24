#include <iostream>
#include <time.h>
#include <math.h>
#include <omp.h>


using namespace std;

const long long int N = 1e10;

int main() {
    double sum = 0;
    double start_time = omp_get_wtime();
    omp_set_num_threads(4);
#pragma omp parallel for reduction(+:sum)
    for (long long int i = 1; i <= N; i++) {
        sum += (pow(i, (1.0 / 3)) / ((i + 1) * sqrt(i)));
    }
    double end_time = omp_get_wtime();
    cout << "time = " << (end_time - start_time) << endl;
    cout << "sum = " << sum << endl;

    return 0;
}