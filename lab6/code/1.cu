#include <stdio.h>
#include <stdlib.h>

 __global__ void GEMM_cuda(double *matA,double *matB, double *matC, int m, int n, int k)
 { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0;
    if( col < k && row < m) {
        for(int i = 0; i < n; i++) {
            temp += matA[row * n + i] * matB[i * k + col];
        }
        matC[row * k + col] = temp;
    }
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

int main(int argc, char const *argv[])
{
    int block_size=atoi(argv[1]);
    int m=atoi(argv[2]);
    int n=atoi(argv[3]);
    int k=atoi(argv[4]);

    double *host_a, *host_b, *host_c;
    double *device_a, *device_b, *device_c;
    float cuda_time;

    cudaMallocHost((void **) &host_a, sizeof(double)*m*n);
    cudaMallocHost((void **) &host_b, sizeof(double)*n*k);
    cudaMallocHost((void **) &host_c, sizeof(double)*m*k);

    srand((unsigned)time(0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            host_a[i * n + j] = (double)rand() / (double)(RAND_MAX)*1e4;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            host_b[i * k + j] = (double)rand() / (double)(RAND_MAX)*1e4;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMalloc((void **) &device_a, sizeof(double)*m*n);
    cudaMalloc((void **) &device_b, sizeof(double)*n*k);
    cudaMalloc((void **) &device_c, sizeof(double)*m*k);
    cudaMemcpy(device_a, host_a, sizeof(double)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(double)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + block_size - 1) / block_size;
    unsigned int grid_cols = (k + block_size - 1) / block_size;
    dim3 gridSize(grid_cols, grid_rows);
    dim3 blockSize(block_size, block_size);
    GEMM_cuda<<<gridSize, blockSize>>>(device_a, device_b, device_c, m, n, k);

    cudaMemcpy(host_c, device_c, sizeof(double)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("matrixA: %dx%d  matrixB: %dx%d  Block_size = %d\n", m, n, n, k, block_size);
    printf("The time of CUDA_GEMM: %f ms.\n", cuda_time);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    return 0;
}