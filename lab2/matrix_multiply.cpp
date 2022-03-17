#include<stdlib.h>
#include "matrix_multiply.h"
double** matrix_multiply(double** matA, double** matB, int m, int n, int k)
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