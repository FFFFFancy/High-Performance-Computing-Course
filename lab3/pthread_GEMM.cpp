#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
using namespace std;

int m,n,k;
double **matA,**matB,**matC;

struct parameter {
	int row,column;
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

void print_matrix(double **mat,int r,int c)
{
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            printf("%f",mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void *thread_compute(void *x) {
	struct parameter *data = (parameter*)x; 
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += matA[data->row][i] * matB[i][data->column];
	}
	matC[data->row][data->column] = sum;
	pthread_exit(0);
}

int main(int argc,char** argv)
{
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	k=atoi(argv[3]);

    matA = generate_matrix(m,n);
    matB = generate_matrix(n,k);
    matC = (double**)malloc(sizeof(double*)*m);;
    for(int i=0;i<m;i++){
        matC[i] = (double*)malloc(sizeof(double)*k);
    }

    print_matrix(matA,m,n);
    print_matrix(matB,n,k);

    clock_t begin,end;
    double time;

    begin=clock();
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < k; j++) {
			struct parameter *x = (struct parameter *) malloc(sizeof(struct parameter));
			x->row = i;
			x->column = j;
			pthread_t t;
			pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_create(&t,&attr,thread_compute,x);
			pthread_join(t, NULL);
		}
	}
	end=clock();
	time=(double)(end-begin)/CLOCKS_PER_SEC;
    
    printf("THE TIME OF PTHREAD_GEMM: %f s\n",time);

    return 0;
}