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
            printf("%f ",mat[i*c+j]);
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
    double *block_A,*block_B,*block_C;

    int rank, numprocs;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);   //numprocs表示进程数
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);       //rank表示进程号
	
    int lines = m / numprocs;       //lines表示分块后的行数
    block_A = new double[lines*n];  //矩阵A的分块
    block_B = new double[n*k];      //矩阵B
    block_C = new double[lines*k];  //计算得到的结果矩阵C

    if(rank == 0)
    {
        matA = new double[m*n];
        matB = new double[n*k];
        matC = new double[m*k];
        generate_matrix(matA,m,n);
        generate_matrix(matB,n,k);

        // printf("matrixA:\n");
        // print_matrix(matA,m,n);
        // printf("matrixB:\n");
        // print_matrix(matB,n,k);

        begin = MPI_Wtime();
        for(int i=1;i<numprocs;i++){
			MPI_Send(matB, n*k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
		for(int i=1;i<numprocs;i++){
			MPI_Send(matA + (i-1)*lines*n, n*lines, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}

        int last = lines * (numprocs-1);
        if(last < m)
        {
            int remaining = m - last;
            GEMM(matA+last*n,matB,matC+last*k,remaining,n,k);
        }

        for(int i=1;i<numprocs;i++){
			MPI_Recv(block_C, lines*k, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int t=0;t<lines;t++){
				for(int j=0;j<k;j++){
					matC[((i-1)*lines+t)*k+j] = block_C[t*k+j];
				}
			}
		}

        end = MPI_Wtime();

        // printf("matrixC:\n");
        // print_matrix(matC,m,k);

        time = end - begin;
        printf("THE TIME OF MPI_GEMM: %f s\n", time);
        delete[] matA;
        delete[] matB;
        delete[] matC;
         
    }else
    {
        MPI_Recv(block_B, n*k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(block_A, n*lines, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        GEMM(block_A, block_B, block_C, lines, n, k);
        MPI_Send(block_C, lines*k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    delete[] block_A;
    delete[] block_B;
    delete[] block_C;

    MPI_Finalize();

    printf("THE TIME OF GEMM: %f s\n",time);

    return 0;
}