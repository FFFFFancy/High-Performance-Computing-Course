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

double** divide_dimensionK(double** matA, double** matB)
{
    double** ans = (double**)malloc(sizeof(double*)*m);
    for(int i=0;i<m;i++){
        ans[i]=(double*)malloc(sizeof(double)*k);
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j+=4){
            ans[i][j+0] = 0;
            ans[i][j+1] = 0;
            ans[i][j+2] = 0;
            ans[i][j+3] = 0;
            for(int t=0;t<n;t++){
                ans[i][j+0] += matA[i][t]*matB[t][j+0];
                ans[i][j+1] += matA[i][t]*matB[t][j+1];
                ans[i][j+2] += matA[i][t]*matB[t][j+2];
                ans[i][j+3] += matA[i][t]*matB[t][j+3];
            }
        }
    }
    return ans;
}

double** divide_dimensionM(double** matA, double** matB)
{
    double** ans = (double**)malloc(sizeof(double*)*m);
    for(int i=0;i<m;i++){
        ans[i]=(double*)malloc(sizeof(double)*k);
    }
    for(int i=0;i<m;i+=4){
        for(int j=0;j<k;j+=4){
            ans[i+0][j+0] = 0;
            ans[i+0][j+1] = 0;
            ans[i+0][j+2] = 0;
            ans[i+0][j+3] = 0;
            ans[i+1][j+0] = 0;
            ans[i+1][j+1] = 0;
            ans[i+1][j+2] = 0;
            ans[i+1][j+3] = 0;
            ans[i+2][j+0] = 0;
            ans[i+2][j+1] = 0;
            ans[i+2][j+2] = 0;
            ans[i+2][j+3] = 0;
            ans[i+3][j+0] = 0;
            ans[i+3][j+1] = 0;
            ans[i+3][j+2] = 0;
            ans[i+3][j+3] = 0;
            for(int t=0;t<n;t++){
                ans[i+0][j+0] += matA[i+0][t]*matB[t][j+0];
                ans[i+0][j+1] += matA[i+0][t]*matB[t][j+1];
                ans[i+0][j+2] += matA[i+0][t]*matB[t][j+2];
                ans[i+0][j+3] += matA[i+0][t]*matB[t][j+3];
                ans[i+1][j+0] += matA[i+1][t]*matB[t][j+0];
                ans[i+1][j+1] += matA[i+1][t]*matB[t][j+1];
                ans[i+1][j+2] += matA[i+1][t]*matB[t][j+2];
                ans[i+1][j+3] += matA[i+1][t]*matB[t][j+3];
                ans[i+2][j+0] += matA[i+2][t]*matB[t][j+0];
                ans[i+2][j+1] += matA[i+2][t]*matB[t][j+1];
                ans[i+2][j+2] += matA[i+2][t]*matB[t][j+2];
                ans[i+2][j+3] += matA[i+2][t]*matB[t][j+3];
                ans[i+3][j+0] += matA[i+3][t]*matB[t][j+0];
                ans[i+3][j+1] += matA[i+3][t]*matB[t][j+1];
                ans[i+3][j+2] += matA[i+3][t]*matB[t][j+2];
                ans[i+3][j+3] += matA[i+3][t]*matB[t][j+3];
            }
        }
    }
    return ans;
}

double** divide_dimensionN(double** matA, double** matB)
{
    double** ans = (double**)malloc(sizeof(double*)*m);
    for(int i=0;i<m;i++){
        ans[i]=(double*)malloc(sizeof(double)*k);
    }
    for(int i=0;i<m;i+=4){
        for(int j=0;j<k;j+=4){
            ans[i+0][j+0] = 0;
            ans[i+0][j+1] = 0;
            ans[i+0][j+2] = 0;
            ans[i+0][j+3] = 0;
            ans[i+1][j+0] = 0;
            ans[i+1][j+1] = 0;
            ans[i+1][j+2] = 0;
            ans[i+1][j+3] = 0;
            ans[i+2][j+0] = 0;
            ans[i+2][j+1] = 0;
            ans[i+2][j+2] = 0;
            ans[i+2][j+3] = 0;
            ans[i+3][j+0] = 0;
            ans[i+3][j+1] = 0;
            ans[i+3][j+2] = 0;
            ans[i+3][j+3] = 0;
            for(int t=0;t<n;t+=4){
                ans[i+0][j+0] += matA[i+0][t+0]*matB[t+0][j+0];
                ans[i+0][j+1] += matA[i+0][t+0]*matB[t+0][j+1];
                ans[i+0][j+2] += matA[i+0][t+0]*matB[t+0][j+2];
                ans[i+0][j+3] += matA[i+0][t+0]*matB[t+0][j+3];
                ans[i+1][j+0] += matA[i+1][t+0]*matB[t+0][j+0];
                ans[i+1][j+1] += matA[i+1][t+0]*matB[t+0][j+1];
                ans[i+1][j+2] += matA[i+1][t+0]*matB[t+0][j+2];
                ans[i+1][j+3] += matA[i+1][t+0]*matB[t+0][j+3];
                ans[i+2][j+0] += matA[i+2][t+0]*matB[t+0][j+0];
                ans[i+2][j+1] += matA[i+2][t+0]*matB[t+0][j+1];
                ans[i+2][j+2] += matA[i+2][t+0]*matB[t+0][j+2];
                ans[i+2][j+3] += matA[i+2][t+0]*matB[t+0][j+3];
                ans[i+3][j+0] += matA[i+3][t+0]*matB[t+0][j+0];
                ans[i+3][j+1] += matA[i+3][t+0]*matB[t+0][j+1];
                ans[i+3][j+2] += matA[i+3][t+0]*matB[t+0][j+2];
                ans[i+3][j+3] += matA[i+3][t+0]*matB[t+0][j+3];

                ans[i+0][j+0] += matA[i+0][t+1]*matB[t+1][j+0];
                ans[i+0][j+1] += matA[i+0][t+1]*matB[t+1][j+1];
                ans[i+0][j+2] += matA[i+0][t+1]*matB[t+1][j+2];
                ans[i+0][j+3] += matA[i+0][t+1]*matB[t+1][j+3];
                ans[i+1][j+0] += matA[i+1][t+1]*matB[t+1][j+0];
                ans[i+1][j+1] += matA[i+1][t+1]*matB[t+1][j+1];
                ans[i+1][j+2] += matA[i+1][t+1]*matB[t+1][j+2];
                ans[i+1][j+3] += matA[i+1][t+1]*matB[t+1][j+3];
                ans[i+2][j+0] += matA[i+2][t+1]*matB[t+1][j+0];
                ans[i+2][j+1] += matA[i+2][t+1]*matB[t+1][j+1];
                ans[i+2][j+2] += matA[i+2][t+1]*matB[t+1][j+2];
                ans[i+2][j+3] += matA[i+2][t+1]*matB[t+1][j+3];
                ans[i+3][j+0] += matA[i+3][t+1]*matB[t+1][j+0];
                ans[i+3][j+1] += matA[i+3][t+1]*matB[t+1][j+1];
                ans[i+3][j+2] += matA[i+3][t+1]*matB[t+1][j+2];
                ans[i+3][j+3] += matA[i+3][t+1]*matB[t+1][j+3];

                ans[i+0][j+0] += matA[i+0][t+2]*matB[t+2][j+0];
                ans[i+0][j+1] += matA[i+0][t+2]*matB[t+2][j+1];
                ans[i+0][j+2] += matA[i+0][t+2]*matB[t+2][j+2];
                ans[i+0][j+3] += matA[i+0][t+2]*matB[t+2][j+3];
                ans[i+1][j+0] += matA[i+1][t+2]*matB[t+2][j+0];
                ans[i+1][j+1] += matA[i+1][t+2]*matB[t+2][j+1];
                ans[i+1][j+2] += matA[i+1][t+2]*matB[t+2][j+2];
                ans[i+1][j+3] += matA[i+1][t+2]*matB[t+2][j+3];
                ans[i+2][j+0] += matA[i+2][t+2]*matB[t+2][j+0];
                ans[i+2][j+1] += matA[i+2][t+2]*matB[t+2][j+1];
                ans[i+2][j+2] += matA[i+2][t+2]*matB[t+2][j+2];
                ans[i+2][j+3] += matA[i+2][t+2]*matB[t+2][j+3];
                ans[i+3][j+0] += matA[i+3][t+2]*matB[t+2][j+0];
                ans[i+3][j+1] += matA[i+3][t+2]*matB[t+2][j+1];
                ans[i+3][j+2] += matA[i+3][t+2]*matB[t+2][j+2];
                ans[i+3][j+3] += matA[i+3][t+2]*matB[t+2][j+3];

                ans[i+0][j+0] += matA[i+0][t+3]*matB[t+3][j+0];
                ans[i+0][j+1] += matA[i+0][t+3]*matB[t+3][j+1];
                ans[i+0][j+2] += matA[i+0][t+3]*matB[t+3][j+2];
                ans[i+0][j+3] += matA[i+0][t+3]*matB[t+3][j+3];
                ans[i+1][j+0] += matA[i+1][t+3]*matB[t+3][j+0];
                ans[i+1][j+1] += matA[i+1][t+3]*matB[t+3][j+1];
                ans[i+1][j+2] += matA[i+1][t+3]*matB[t+3][j+2];
                ans[i+1][j+3] += matA[i+1][t+3]*matB[t+3][j+3];
                ans[i+2][j+0] += matA[i+2][t+3]*matB[t+3][j+0];
                ans[i+2][j+1] += matA[i+2][t+3]*matB[t+3][j+1];
                ans[i+2][j+2] += matA[i+2][t+3]*matB[t+3][j+2];
                ans[i+2][j+3] += matA[i+2][t+3]*matB[t+3][j+3];
                ans[i+3][j+0] += matA[i+3][t+3]*matB[t+3][j+0];
                ans[i+3][j+1] += matA[i+3][t+3]*matB[t+3][j+1];
                ans[i+3][j+2] += matA[i+3][t+3]*matB[t+3][j+2];
                ans[i+3][j+3] += matA[i+3][t+3]*matB[t+3][j+3];
            }
        }
    }
    return ans;
}

int main()
{
    printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

    if(m%4||n%4||k%4){
		printf("INPUT ERROR: The input must be a multiple of 4.\n");
		return 0;
	} 

    double** matA = generate_matrix(m,n);
    double** matB = generate_matrix(n,k);

    clock_t begin,end;
    begin = clock();
	double **matC = GEMM(matA,matB);
	end = clock();
    double time1=(double)(end-begin)/CLOCKS_PER_SEC;

    begin = clock();
	double **matD = divide_dimensionN(matA,matB);
	end = clock();
    double time2=(double)(end-begin)/CLOCKS_PER_SEC;

	printf("THE TIME OF GEMM:     %f s\n",time1);
	printf("THE TIME OF divide_dimension: %f s\n",time2);

    return 0;
}