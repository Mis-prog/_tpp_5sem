#include "mpi.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	int MyID, NumProc;
	int iError = MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
	MPI_Status status;
	std::cout.precision(8);
	if (iError != MPI_SUCCESS)
	{
		std::cout << "MPI error!\n";
		exit(1);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	double timerStart = MPI_Wtime();
	
	if (MyID == 0)
	{
		long long n = atoll(argv[1]);
		double sum = 0.;
		double localSum = 0.;
		long long amountOfOperations = n / NumProc;
//		long long remainderOperations = n % NumProc;

		for (int i = 1; i <= NumProc-1; i++)
		{
			MPI_Send(&n, 1, MPI_LONG, i, 1000 + i, MPI_COMM_WORLD);
		}
		
		
		long long from = n - amountOfOperations + 1 + 1;
		long long to = n + 1;
		for (long long i = from; i <= to; i++)
		{
			sum += pow(-1., i - 1)/(pow(i, 2) - i);
		}
		
		
		//std::cout << "Sum that ID: " << MyID << " calculated. S = " << sum << "\n";		
		for (int i = 1; i <= NumProc - 1; i++)
		{
			MPI_Recv(&localSum, 1, MPI_DOUBLE, i, 1000, MPI_COMM_WORLD, &status);
			sum += localSum;
		} 
		double timerEnd = MPI_Wtime();
		std::cout << "Number of steps: " << n << std::endl;
		std::cout << "Partial sum is: " << sum << std::endl;
		std::cout << "Elapsed time: " << timerEnd - timerStart << std::endl;
	}
	else if (MyID > 0)
	{
		double localSum = 0.;
		long long n;
		//std::cout << "Processor ID: " << MyID << std::endl;
		MPI_Recv(&n, 1, MPI_LONG, 0, 1000 + MyID, MPI_COMM_WORLD, &status);
		long long amountOfOperations = n / NumProc;
		long long remainderOperations = n % NumProc;
		long long from;
		long long to;
		if (MyID <= remainderOperations)
		{
			from = (amountOfOperations+1) * (MyID-1) + 1 + 1;
			to = from + amountOfOperations + 1;
		}
		else 
		{
			from = remainderOperations * (amountOfOperations + 1) + (MyID - 1 - remainderOperations) * amountOfOperations + 1;  
			to = from + amountOfOperations-1;
		}
		//std::cout << "from: " << from << " to " << to << "\n";		
		for (long long i = from; i <= to; i++)
		{
			localSum += pow(-1., i - 1)/(pow(i, 2) - i);
		}
		MPI_Send(&localSum, 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD);		
	}
	MPI_Finalize();
	return 0;
}
