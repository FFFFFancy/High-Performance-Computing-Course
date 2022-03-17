#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int m,n,k;

int extend_matrix(int m, int n, int k)
{
    int ext = m;
    if(ext<n)   ext = n;
    if(ext<k)   ext = k;

    while(ext & (ext-1) != 0)
        ext++;
    
    return ext;
}

double** generate_matrix(int r, int c, int extend)
{
    double** mat = (double**)malloc(sizeof(double*)*extend);
    for(int i=0;i<extend;i++){
        mat[i]=(double*)malloc(sizeof(double)*extend);
    }

    srand((unsigned)time(0));
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            mat[i][j] = (double)rand() / (double)(RAND_MAX)*10;
        }
        for(int j=c;j<extend;j++){
            mat[i][j] = 0;
        }
    }

    for(int i=r;i<extend;i++){
        for(int j=0;j<extend;j++){
            mat[i][j] = 0;
        }
    }

    return mat;
}

double** divide_matrix(double** mat, int size, int div_pos)       //矩阵分块
{
    double **matrix = (double**)malloc(sizeof(double*)*(size/2));
	if(div_pos == 1){
		for (int i=0;i<size/2;i++){
			matrix[i] = (double *)malloc(sizeof(double)*(size/2));
			for (int j=0;j<size/2;j++){
				matrix[i][j] = mat[i][j];
			}
		}
	}
	else if(div_pos == 2){
		for (int i=0;i<size/2;i++){
			matrix[i]=(double *)malloc(sizeof(double)*(size/2));
			for (int j=0;j<size/2;j++){
				matrix[i][j] = mat[i][size/2+j];
			}
		}
	}
	else if(div_pos == 3){
		for (int i=0;i<size/2;i++) {
			matrix[i]=(double *)malloc(sizeof(double)*(size/2));
			for (int j=0;j<size/2;j++) {
				matrix[i][j] = mat[i+size/2][j];
			}
		}
	}
	else if(div_pos == 4){
		for (int i=0;i<size/2;i++){
			matrix[i]=(double *)malloc(sizeof(double)*(size/2));
			for (int j=0;j<size/2;j++){
				matrix[i][j] = mat[i+size/2][j+size/2];
			}
		}
	}
	return matrix;
}

double** merge_matrix(double** mat1, double** mat2, double** mat3, double** mat4, int size)     //矩阵合并
{
	double **matrix=(double**)malloc(sizeof(double*)*size);
	for (int i=0;i<size;i++)
			matrix[i]=(double *)malloc(sizeof(double)*size);

    for(int i=0;i<size/2;i++){
        for(int j=0;j<size/2;j++){
            matrix[i][j] = mat1[i][j];
        }
        for(int j=size/2;j<size;j++){
            matrix[i][j] = mat2[i][j-size/2];
        }
    }

    for(int i=size/2;i<size;i++){
        for(int j=0;j<size/2;j++){
            matrix[i][j] = mat3[i-size/2][j];
        }
        for(int j=size/2;j<size;j++){
            matrix[i][j] = mat2[i-size/2][j-size/2];
        }
    }
	return matrix;
}

double** add_matrix(double** matA, double** matB, int size)     //矩阵加法
{
	double **ans=(double**)malloc(sizeof(double*)*size);
	for (int i=0;i<size;i++){
		ans[i]=(double *)malloc(sizeof(double)*size);
		for (int j=0;j<size;j++){
			ans[i][j] = matA[i][j] + matB[i][j];
		}
	}
	return ans;
}

double** sub_matrix(double** matA, double** matB, int size)     //矩阵减法
{
	double **ans=(double**)malloc(sizeof(double*)*size);
	for (int i=0;i<size;i++){
		ans[i]=(double *)malloc(sizeof(double)*size);
		for (int j=0;j<size;j++){
			ans[i][j] = matA[i][j] - matB[i][j];
		}
	}
	return ans;
}

double** GEMM(double** matA, double** matB, int size)
{
    double** ans = (double**)malloc(sizeof(double*)*size);
    for(int i=0;i<size;i++){
        ans[i]=(double*)malloc(sizeof(double)*size);
    }
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            ans[i][j] = 0;
            for(int t=0;t<size;t++){
                ans[i][j] += matA[i][t]*matB[t][j];
            }
        }
    }
    return ans;
}

double** Strassen(double** matA, double** matB, int size)
{
    if(size <= 64){
		return GEMM(matA, matB, size);
	}
    else{
        double** a11 = divide_matrix(matA,size,1);
        double** a12 = divide_matrix(matA,size,2);
        double** a21 = divide_matrix(matA,size,3);
        double** a22 = divide_matrix(matA,size,4);
        double** b11 = divide_matrix(matB,size,1);
        double** b12 = divide_matrix(matB,size,2);
        double** b21 = divide_matrix(matB,size,3);
        double** b22 = divide_matrix(matB,size,4);

        double** p1 = Strassen(a11, sub_matrix(b12, b22, size/2), size/2);
		double** p2 = Strassen(add_matrix(a11, a12, size/2), b22, size/2);
		double** p3 = Strassen(add_matrix(a21, a22, size/2), b11, size/2);
		double** p4 = Strassen(a22, sub_matrix(b21, b11, size/2), size/2);
		double** p5 = Strassen(add_matrix(a11, a22, size/2), add_matrix(b11, b22, size/2), size/2);
		double** p6 = Strassen(sub_matrix(a12, a22, size/2), add_matrix(b21, b22, size/2), size/2);
		double** p7 = Strassen(sub_matrix(a11, a21, size/2), add_matrix(b11, b12, size/2), size/2);
		double** C11 = add_matrix(add_matrix(p4, p5, size/2), sub_matrix(p6, p2, size/2), size/2);

		double** C12 = add_matrix(p1, p2, size/2);
		double** C21 = add_matrix(p3, p4, size/2);
		double** C22 = add_matrix(sub_matrix(p1, p3, size/2), sub_matrix(p5, p7, size/2), size/2);

        return merge_matrix(C11,C12,C21,C22,size);
    }
}

int main()
{
    printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

    int extend_size = extend_matrix(m,n,k);
    double** matA = generate_matrix(m,n,extend_size);
    double** matB = generate_matrix(n,k,extend_size);
    double** matC;
    double** matD;

    clock_t begin,end;
    begin = clock();
	matC = GEMM(matA,matB,extend_size);
	end = clock();
    double time1=(double)(end-begin)/CLOCKS_PER_SEC;

    begin = clock();
	matD = Strassen(matA,matB,extend_size);
	end = clock();
    double time2=(double)(end-begin)/CLOCKS_PER_SEC;

	printf("THE TIME OF GEMM:     %f s\n",time1);
	printf("THE TIME OF Strassen: %f s\n",time2);

    return 0;
}