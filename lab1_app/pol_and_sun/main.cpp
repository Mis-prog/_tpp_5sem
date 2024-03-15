#include "mpi.h"
#include <stdio.h>
#include <iostream>
using namespace std;

void q_sort(int* mas, int size) {

    int i = 0;
    int j = size - 1;
    int mid = mas[size / 2];


    do {

        while (mas[i] < mid) {
            i++;
        }

        while (mas[j] > mid) {
            j--;
        }

        if (i <= j) {
            int tmp = mas[i];
            mas[i] = mas[j];
            mas[j] = tmp;

            i++;
            j--;
        }
    } while (i <= j);



    if (j > 0) {

        q_sort(mas, j + 1);
    }
    if (i < size) {

        q_sort(&mas[i], size - i);
    }
}


int main(int argc, char *argv[])
{

    int MyID, NumProc, ierror;
    double a, b;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    MPI_Status status;


    int size;
    srand(time(0));
    if(MyID==0)
    {
        cout<<"Matrix length: "<<endl;
        cin>>size;

        double *matr=new double[size];
        for (int i=0;i<size;i++)
        {
            matr[i]=(int)(100*((double)(rand())/RAND_MAX))-20;
            cout<<matr[i]<<endl;
        }
    }



    MPI_Finalize();
    return 0;
}