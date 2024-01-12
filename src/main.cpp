#include <iostream>
#include <time.h>
#include <math.h>

#define _USE_MATH_DEFINES

using namespace std;

const long long int N = 1e9;

int main() {
 double sum = 0;
 double start_time = clock();
 for (long long int i = 1; i<=N; i++) {
  sum += (pow(i,(1.0/3)) / ((i + 1) * sqrt(i)));
 }
 double end_time = clock();
 cout << "time = " << (end_time - start_time) / CLOCKS_PER_SEC<< endl;
 cout << "sum = " << sum << endl;

 return 0;
}