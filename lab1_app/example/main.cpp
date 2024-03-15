#include "mpi.h"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
int main(int argc, char *argv[])
{ 	
	int MyID, NumProc;
	double sum = 0.0;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
	
	double recv = 0;
	double a;
	
	if (MyID == 0) {
		long N = 6;
		
		for (int i = 1; i <= NumProc-1; ++i) {
			MPI_Send(&N, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
		}
		
		for (long j = 1; j <= N; j += NumProc) {
			for (int i = 1; i <= NumProc-1; ++i) {
				a = j + i;
				MPI_Send(&a, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
			}
			sum += pow(-1, j) * (j + 1) / ((j + 1) * sqrt(j + 1) - 1);	
		}
		for (int i = 1; i <= NumProc-1; ++i) {
			MPI_Recv(&a, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
			sum += a;
		}
	} else if (MyID > 0) {
		long N;
		MPI_Recv(&N, 1, MPI_LONG, 0, MyID, MPI_COMM_WORLD, &status);
		for (long j = MyID + 1; j <= N; j += NumProc) {
			MPI_Recv(&recv, 1, MPI_DOUBLE, 0, MyID, MPI_COMM_WORLD, &status);
			sum += pow(-1, recv) * (recv + 1) / ((recv + 1) * sqrt(recv + 1) - 1);
			
		}
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, MyID, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("%d: %f\n", MyID, sum);
	if(MyID == 0) {
		printf("%d: %f\n", MyID, sum);
	}
	MPI_Finalize();
	return 0;
}
