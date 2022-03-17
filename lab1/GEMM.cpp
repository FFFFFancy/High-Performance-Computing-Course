#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int m,n,k;

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

double** GEMM(double** matA, double** matB)
{
    double** ans = (double**)malloc(sizeof(double*)*m);
    for(int i=0;i<m;i++){
        ans[i]=(double*)malloc(sizeof(double)*k);
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            ans[i][j] = 0;
            for(int t=0;t<n;t++){
                ans[i][j] += matA[i][t]*matB[t][j];
            }
        }
    }
    return ans;
}

int main()
{
    printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

    double** matA = generate_matrix(m,n);
    double** matB = generate_matrix(n,k);
    double** matC;

    clock_t begin,end;
    begin = clock();
	matC = GEMM(matA,matB);
	end = clock();
    double time=(double)(end-begin)/CLOCKS_PER_SEC;

    printf("matrixA(%d×%d)：\n",m,n);
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            printf("%f ",matA[i][j]);
        }
        printf("\n");
    }

	printf("matrixB(%d×%d)：\n",n,k);
    for (int i=0;i<n;i++){
        for (int j=0;j<k;j++){
            printf("%f ",matB[i][j]);
        }
        printf("\n");
    }

    printf("matrixC(%d×%d)=matrixA*matrixB：\n",m,k);
    for (int i=0;i<m;i++){
        for (int j=0;j<k;j++){
            printf("%f ",matC[i][j]);
        }
        printf("\n");
    }

    printf("THE TIME OF GEMM: %f s\n",time);

    return 0;
}