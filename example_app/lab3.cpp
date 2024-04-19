#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define Pi 3.1415926535897
#define K 0.00001864472034
#define L 0.01 // пространственная область - квадрат с длиной стороны L


using namespace std;

double u0(double x, double y) { return (200. + 50. * y) * (cos(Pi * x / 2.) + x); }
double mu1(double y, double t) { return 200. + 50. * y + t * exp(5. * y); }
double mu2(double y, double t) { return 200. + 50. * y + t * (550. + 200. * y); }
double mu3(double x, double t) { return 200. * (cos(Pi * x / 2.) + x) + t * (550. * sin(Pi * x / 2.) + 1. - x); }
double mu4(double x, double t) { return 250. * (cos(Pi * x / 2.) + x) + t * (750. * x + (1. - x) * exp(5.)); }

double c(double u) { return 1. / (2.25e-3 - 6.08e-10 * u * u); }
double rho(double u) { return 7860. + 41500. / u; }
double lambda(double u) { return 1.45 + 2.3e-2 * u - 2.e-6 * u * u; }


int main(int argc, char* argv[])
{
	int p, o, s = 0.0;
	double* u, * u1;

	int MyID, NumProc, ierror;
	MPI_Status status;
	ierror = MPI_Init(&argc, &argv);
	if (ierror != MPI_SUCCESS)
	{
		printf("MPI initialization error!");
		exit(1);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

	int* rcounts, * displs;
	if (MyID == 0) { rcounts = (int*)calloc(NumProc, sizeof(int)); displs = (int*)calloc(NumProc, sizeof(int)); }

	int I = atoi(argv[1]);
	double T = atof(argv[2]);

	double dx = L / I;
	double tau = (0.9 * dx * dx) / (4.0 * K);
	p = (I - 2) / NumProc;
	o = (I - 2) % NumProc;

	if (MyID < o) { p++; }
	else { s = o; }

	u = new double[(p + 2) * I];

	if (MyID)
		u1 = (double*)calloc((p + 2) * I, sizeof(double));
	else
	{
		u1 = (double*)calloc(I * I, sizeof(double));
	}

	for (int j = MyID * p + s; j < (MyID + 1) * p + s + 2; j++)
	{
		for (int i = 0; i < I; i++)
		{
			u[i + (j - MyID * p - s) * I] = u0(i / (double)I, j / (double)I);
		}
	}

	if (!MyID)
	{
		displs[0] = 0;
		for (int i = 0; i < NumProc; i++)
		{
			if (i < o)
				rcounts[i] = ((I - 2) / NumProc + 3) * I;
			else
				rcounts[i] = ((I - 2) / NumProc + 2) * I;
			if (i < NumProc - 1)
				displs[i + 1] = displs[i] + rcounts[i] - 2 * I;
		}
	}

	MPI_Gatherv(u, (p + 2) * I, MPI_DOUBLE, u1, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (MyID == 0)
	{
		ofstream out("u0.txt");
		for (int i = 0; i < I; i++)
		{
			for (int j = 0; j < I; j++)
				out << u1[i * I + j] << "\t";
			out << endl;
		}
		out.close();
	}

	double tstart = MPI_Wtime();
	for (double t = tau; t < T; t += tau)
	{
		for (int j = 1; j < p + 1; j++)
			for (int i = 1; i < I - 1; i++)
			{
				int idx = i + j * I;
				double lambda1 = lambda((u[idx + 1] + u[idx]) / 2.);
				double lambda2 = lambda((u[i - 1 + j * I] + u[idx]) / 2.);
				double lambda3 = lambda((u[i + (j + 1) * I] + u[idx]) / 2.);
				double lambda4 = lambda((u[i + (j - 1) * I] + u[idx]) / 2.);
				double cc = c(u[idx]);
				double roc = rho(u[idx]);
				u1[idx] = u[idx] + tau / (cc * roc) *
					(lambda1 * (u[idx + 1] - u[idx]) / (dx * dx)
						- lambda2 * (u[idx] - u[idx - 1]) / (dx * dx)
						+ lambda3 * (u[idx + I] - u[idx]) / (dx * dx)
						- lambda4 * (u[idx] - u[idx - I]) / (dx * dx));
			}

		if (MyID)
			MPI_Sendrecv(&u1[I + 1], I - 2, MPI_DOUBLE, MyID - 1, 1, &u1[1], I - 2, MPI_DOUBLE, MyID - 1, 1, MPI_COMM_WORLD, &status);

		if (NumProc - MyID - 1)
			MPI_Sendrecv(&u1[p * I + 1], I - 2, MPI_DOUBLE, MyID + 1, 1, &u1[(p + 1) * I + 1], I - 2, MPI_DOUBLE, MyID + 1, 1, MPI_COMM_WORLD, &status);

		if (!MyID)
			for (int i = 1; i < I - 1; i++)
			{
				u1[i] = mu3(i / (double)I, t / T);
			}

		if (!(NumProc - MyID - 1))
			for (int i = 1; i < I - 1; i++)
			{
				u1[(p + 1) * I + i] = mu4(i / (double)I, t / T);
			}

		for (int j = 0; j < p + 2; j++)
		{
			u1[j * I] = mu1((j + MyID * p + s) / (double)I, t / T);
			u1[(j + 1) * I - 1] = mu2((j + MyID * p + s) / (double)I, t / T);

		}
		for (int j = 0; j < p + 2; j++)
		{
			for (int i = 0; i < I; i++)
			{
				u[i + j * I] = u1[i + j * I];
			}
		}
	}

	MPI_Gatherv(u, (p + 2) * I, MPI_DOUBLE, u1, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (MyID == 0)
	{
		double time = MPI_Wtime() - tstart;
		cout << "Time: " << time << endl;
    cout << "NumProc: " << NumProc << endl;
    cout << "I: " << I << endl;
    cout << "T: " << T << endl;
		ofstream out("u1.txt");
		for (int i = 0; i < I; i++)
		{
			for (int j = 0; j < I; j++)
				out << u1[i * I + j] << "\t";
			out << endl;
		}
		out.close();
	}

	MPI_Finalize();
	return 0;
}