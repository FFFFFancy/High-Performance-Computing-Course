#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"parallel_for.h"

int m,n,k;
int thread_num;
double **matA,**matB,**matC;

struct for_index {
    int start;
    int end;
    int increment;
};

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

void print_matrix(double *mat,int r,int c)
{
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            printf("%f",mat[i*c+j]);
        }
        printf("/n");
    }
    printf("\n");
}

void GEMM(void *args)
{
    struct for_index * index = (struct for_index *) args;
    for (int i = index->start; i < index->end; i = i + index->increment){
        for (int j = 0; j < k; j++)
		{
			matC[i][j] = 0;
			for (int l = 0; l < n; l++)
			{
				matC[i][j] += matA[i][l] * matB[l][j];
			}
		}
	}
}

int main(int argc,char* argv[])
{
    thread_num = atoi(argv[1]);

	printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

    matA = generate_matrix(m,n);
    matB = generate_matrix(n,k);
    matC = (double**)malloc(sizeof(double*)*m);;
    for(int i=0;i<m;i++){
        matC[i] = (double*)malloc(sizeof(double)*k);
    }

    clock_t begin,end;
    double time;
    begin=clock();

	parallel_for(0, m, 1, GEMM, NULL, thread_num);
    
	end=clock();
	time=(double)(end-begin)/CLOCKS_PER_SEC;
    
    printf("THE TIME OF OPENMP_GEMM: %f s\n",time);

    return 0;
}