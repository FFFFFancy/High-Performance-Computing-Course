#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"matrix_multiply.h"

double** generate_matrix(int r,int c)
{
    double** mat = (double**)malloc(sizeof(double*)*r);
    for(int i=0;i<r;i++){
        mat[i]=(double*)malloc(sizeof(double)*c);
    }

    srand((unsigned)time(0));
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            mat[i][j] = (double)rand() / (double)(RAND_MAX)*10;
            //printf("%d",mat[i][j]);
        }
    }

    return mat;
}

int main()
{
    int m,n,k;

    printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

    double** matA = generate_matrix(m,n);
    double** matB = generate_matrix(n,k);
    double** matC;

    clock_t begin,end;
    begin = clock();
	matC = matrix_multiply(matA,matB,m,n,k);
	end = clock();
    double time=(double)(end-begin)/CLOCKS_PER_SEC;

    printf("THE TIME OF GEMM: %f s\n",time);

    return 0;
}