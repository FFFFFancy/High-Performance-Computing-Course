#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int m,n,k;
double **matA,**matB,**matC;

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

int main(int argc,char** argv)
{
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
    #pragma omp parallel for
	for (int i = 0; i < m;i++){
    #pragma omp privite(sum,t) parallel for
		for (int j = 0; j < n;j++){
			double sum = 0;
			for (int t = 0; t < k; t++)
				sum += matA[i][t] * matB[t][j];
			matC[i][j] = sum;
		}
	}
	end=clock();
	time=(double)(end-begin)/CLOCKS_PER_SEC;
    printf("THE TIME OF DEFAULT: %f s\n",time);

    begin=clock();
    #pragma omp parallel for schedule(static,1)
	for (int i = 0; i < m;i++){
    #pragma omp privite(sum,t) parallel for schedule(static,1)
		for (int j = 0; j < n;j++){
			double sum = 0;
			for (int t = 0; t < k; t++)
				sum += matA[i][t] * matB[t][j];
			matC[i][j] = sum;
		}
	}
	end=clock();
	time=(double)(end-begin)/CLOCKS_PER_SEC;
    printf("THE TIME OF STATIC: %f s\n",time);

    begin=clock();
    #pragma omp parallel for schedule(dynamic,1)
	for (int i = 0; i < m;i++){
    #pragma omp privite(sum,t) parallel for schedule(dynamic,1)
		for (int j = 0; j < n;j++){
			double sum = 0;
			for (int t = 0; t < k; t++)
				sum += matA[i][t] * matB[t][j];
			matC[i][j] = sum;
		}
	}
	end=clock();
	time=(double)(end-begin)/CLOCKS_PER_SEC;
    printf("THE TIME OF DYNAMIC: %f s\n",time);

    return 0;
}