#include "mpi.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[])
{
	srand((unsigned int)1);

	int MyID, NumProc, ierror;
	ierror = MPI_Init(&argc, &argv);

	if (ierror != MPI_SUCCESS)
		cout << "MPI Init error" << endl;

	MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

	if (argc != 2)
	{
		cout << "[!] 'N' needed" << endl;
		return -1;
	}

	long long L;
	L = atoi(argv[1]);

	long long N = 10 * L, p = N / NumProc, o = N % NumProc;

	double *A = (double*)calloc(N*L, sizeof(double)),
		*B = (double*)calloc(L*L, sizeof(double)),
		*C = (double*)calloc(N*L, sizeof(double)),
		*Ap = (double*)calloc((p + 1)*L, sizeof(double)),
		*Cp = (double*)calloc((p + 1)*L, sizeof(double)),
		*normid = (double*)calloc(1, sizeof(double)),
		*Norma = (double*)calloc(1, sizeof(double));

	int *count = (int*)calloc(NumProc, sizeof(int)), *displs = (int*)calloc(NumProc, sizeof(int)),
		*end = (int*)calloc(NumProc, sizeof(int)),
		ccount;

	double tstart, tfinish;

	if (MyID == 0)
	{
		for (int i = 0; i < N*L; i++) A[i] = -0.5 + (double)rand() / RAND_MAX;;

		for (int i = 0; i < L*L; i++) B[i] = -0.5 + (double)rand() / RAND_MAX;;

		
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (MyID == 0)
	{
		
		for (int i = 0; i < NumProc; i++)
		{
			if (o > i)
			{
				count[i] = (p + 1)*L;
				displs[i] = i * (p + 1)*L;
			}
			else
			{
				count[i] = p * L;
				displs[i] = o * (p + 1)*L + (i - o)*p*L;
			}

		}
	}
	if (MyID < o)
	{
		ccount = (p + 1)*L;
		end[MyID] = p + 1;
	}
	else
	{
		end[MyID] = p;
		ccount = (p)*L;
	}

	MPI_Scatterv(A, count, displs, MPI_DOUBLE, Ap, ccount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (MyID == 0)
	{	
	tstart = MPI_Wtime();	
	}
	for (int i = 0; i < end[MyID]; i++)
	{
		for (int j = 0; j < L; j++)
		{
			Cp[L * (i)+j] = .0;
			for (int k = 0; k < L; k++)
			{
				Cp[L * (i)+j] += Ap[L * (i)+k] * B[L * (k)+j];
			}

			normid[0] += Cp[L * (i)+j] * Cp[L * (i)+j];
		}

	}

	MPI_Gatherv(Cp, ccount, MPI_DOUBLE, C, count, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Reduce(normid, Norma, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (MyID == 0)
	{
		tfinish = MPI_Wtime() - tstart;
		cout << "> Norm = " << Norma[0] << endl;
		cout << "> Time = " << tfinish << endl;
		cout << "> NumProc = " << NumProc << endl;
	}
	free(A); free(C); free(Ap); free(B); free(Cp); free(normid); free(Norma); free(count); free(displs); free(end);
	MPI_Finalize();
}
