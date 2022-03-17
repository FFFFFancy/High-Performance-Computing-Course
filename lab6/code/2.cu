#include <iostream>
#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"
using namespace std;

int main(int argc, char const *argv[])
{
    int M=atoi(argv[1]);
    int N=atoi(argv[2]);
    int K=atoi(argv[3]);

    // 分配 host 的内存空间
    double *host_A = (double*)malloc (N*M*sizeof(double));
    double *host_B = (double*)malloc (N*M*sizeof(double));
    double *host_C = (double*)malloc (M*M*sizeof(double));
    
    srand((unsigned)time(0));
    for (int i=0; i<N*M; i++) {
        host_A[i] = (double)rand() / (double)(RAND_MAX)*1e4;
        host_B[i] = (double)rand() / (double)(RAND_MAX)*1e4;
    }

    // 创建并初始化 CUBLAS 库对象
    cublasHandle_t handle;
    cublasCreate(&handle);

    double *device_A, *device_B, *device_C;
    // 分配 device 内存空间
    cudaMalloc ((void**)&device_A, N * M * sizeof(double));
    cudaMalloc ((void**)&device_B, N * M * sizeof(double));
    cudaMalloc ((void**)&device_C, M * M * sizeof(double));

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 将矩阵复制到显存中
    cublasSetVector(N * M, sizeof(double), host_A, 1, device_A, 1);
    cublasSetVector (N * M, sizeof(double), host_B, 1, device_B, 1);
    cudaDeviceSynchronize();
    // 赋值alpha和beta，计算矩阵乘法
    double alpha=1;
    double beta=0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, M, N, &alpha, device_A, N, device_B, M, &beta, device_C, M);
    cudaDeviceSynchronize();
    // 将结果复制回内存
    cublasGetVector(M * M, sizeof(double), device_C, 1, host_C, 1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("matrixA: %dx%d  matrixB: %dx%d\n", M, N, N, K);
    printf("The time of CUBLAS: %f ms.\n", time);

    // 清理内存
    free (host_A);
    free (host_B);
    free (host_C);
    cudaFree (device_A);
    cudaFree (device_B);
    cudaFree (device_C);
    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
    
    return 0;
}