#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

MKL_INT m, n, k;

int main()
{

	MKL_INT         lda, ldb, ldc;
	MKL_Complex8    alpha, beta;
	MKL_Complex8   *a, *b, *c;
	CBLAS_LAYOUT    layout = CblasRowMajor;
	CBLAS_TRANSPOSE transA = CblasNoTrans;
	CBLAS_TRANSPOSE transB = CblasNoTrans;
	MKL_INT         ma, na, mb, nb;

	/***************** 参数初始化 *****************/
	printf("ENTER 3 INTERGERS (512~2048) :");
    scanf("%d%d%d",&m,&n,&k);

	alpha.real = 1;
	alpha.imag = 0;
	beta.real = beta.imag = 0;

	if (transA == CblasNoTrans) {
		ma = m;
		na = k;
	} else {
		ma = k;
		na = m;
	}
	if (transB == CblasNoTrans) {
		mb = k;
		nb = n;
	} else {
		mb = n;
		nb = k;
	}

	a = (MKL_Complex8 *)mkl_calloc(ma*na, sizeof(MKL_Complex8), 64);
	b = (MKL_Complex8 *)mkl_calloc(mb*nb, sizeof(MKL_Complex8), 64);
	c = (MKL_Complex8 *)mkl_calloc(ma*nb, sizeof(MKL_Complex8), 64);

	/************** 矩阵与向量赋值 *******************/
	for (int i = 0; i < ma*na; i++) {
		a[i].real = (float)(i + 1);
		a[i].imag = (float)i;
	}
	for (int i = 0; i < mb*nb; i++) {
		b[i].real = (float)i;
		b[i].imag = (float)(i + 1);
	}
	
	if (layout == CblasRowMajor) {
		lda = na;
		ldb = nb;
		ldc = nb;
	} else {
		lda = ma;
		ldb = mb;
		ldc = ma;
	}

    clock_t begin,end;
    begin = clock();
	/*      Call CGEMM subroutine ( C Interface )                  */
	cblas_cgemm(layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb,
		&beta, c, ldc);
    end = clock();
    double time=(double)(end-begin)/CLOCKS_PER_SEC;

    //print time
    printf("THE TIME OF mkl: %f s\n",time);

	mkl_free(a);
	mkl_free(b);
	mkl_free(c);

	return 0;
}
