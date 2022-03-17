#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
using namespace std;

void generate_matrix(double *mat,int r,int c)
{
    srand((unsigned)time(0));
    for(int i=0;i<r*c;i++){
        mat[i] = (double)rand() / (double)(RAND_MAX)*10;
    }
}

void GEMM(double* matA, double* matB, double* matC, int m, int n, int k)
{
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            double temp = 0;
            for(int t=0;t<n;t++)
                temp += matA[i*n+t]*matB[t*k+j];
            matC[i*k+j] = temp;
        }
    }
}

void print_matrix(double *mat,int r,int c)
{
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            printf("%f",mat[i*c+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc,char** argv)
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    double begin,end,time;
    double *matA,*matB,*matC;
    double *block_A,*block_C;

    int rank, numprocs;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);   //numprocs表示进程数
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);       //rank表示进程号
	
    int lines = m / numprocs;       //lines表示分块后的行数
    matB = new double[n*k];         //每个进程都发送的矩阵B
    block_A = new double[lines*n];  //矩阵A的分块
    block_C = new double[lines*k];  //计算得到的结果矩阵C

    if(rank == 0)
    {
        matA = new double[m*n];
        matC = new double[m*k];
        generate_matrix(matA,m,n);
        generate_matrix(matB,n,k);

        // printf("matrixA:\n");
        // print_matrix(matA,m,n);
        // printf("matrixB:\n");
        // print_matrix(matB,n,k);
    }

    begin = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(matB, n*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matA, lines*n, MPI_DOUBLE, block_A, lines*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GEMM(block_A, matB, block_C, lines, n, k);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(block_C, lines*k, MPI_DOUBLE, matC, lines*k, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    int last = lines * (numprocs-1);
    if(rank == 0 && last < m)
    {
        int remaining = m - last;
        GEMM(matA+last*n,matB,matC+last*k,remaining,n,k);
    }

        // printf("matrixC:\n");
        // print_matrix(matC,m,k);

    end = MPI_Wtime();
    time = end - begin;

    delete[] block_A;
    delete[] block_C;
    delete[] matB;
         
    if(rank == 0)
    {
        printf("THE TIME OF MPI_GEMM OPTIMIZATION: %f s\n", time);
        delete[] matA;
        delete[] matC;
    }
    

    MPI_Finalize();
    return 0;
}