#include <stdio.h>
#include <sys/time.h>

#define mat_height 6
#define mat_width 6
#define filter_height 3
#define filter_width 3
#define stride 1

#define block_size_x 3
#define block_size_y 3

// padding
#define padding_height ((filter_height / 2) * 2)
#define padding_width ((filter_width / 2) * 2)
#define input_height (mat_height + padding_height)
#define input_width (mat_width + padding_width)

__global__ void convolution_kernel(float *output, float *input, float *filter)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (y % stride == 0 && x % stride == 0)
    {
        for (int i = 0; i < filter_height; i++)
        {
            for (int j = 0; j < filter_width; j++)
            {
                sum += input[(y + i) * input_width + x + j] * filter[i * filter_width + j];
            }
        }
        output[y / stride * mat_width + x / stride] = sum;
    }
}

__global__ void cuda_add(float *arr1, float *arr2, float *arr3, float *result){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y % stride == 0 && x % stride == 0)
        result[y / stride * mat_width + x / stride] = arr1[y / stride * mat_width + x / stride] + arr2[y / stride * mat_width + x / stride] + arr3[y / stride * mat_width + x / stride];
}

void print_mat(float *arr, int w, int h)
{
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            printf("%f\t", arr[i * w + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    // 动态分配内存
    // input1-3，经过 padding ，作为输入的矩阵，共三层
    float *input1 = (float *)malloc(input_height * input_width * sizeof(float));
    float *input2 = (float *)malloc(input_height * input_width * sizeof(float));
    float *input3 = (float *)malloc(input_height * input_width * sizeof(float));

    //   经过cuda运算后，三层结果相加的最终结果
    float *conv_result = (float *)malloc(mat_height * mat_width * sizeof(float));

    // 三个不同的卷积核
    float *filter1 = (float *)malloc(filter_height * filter_width * sizeof(float));
    float *filter2 = (float *)malloc(filter_height * filter_width * sizeof(float));
    float *filter3 = (float *)malloc(filter_height * filter_width * sizeof(float));

    // 初始化 input
    for (int i = 0; i < input_height * input_width; i++)
    {
        input1[i] = (float)(rand() % 50)/100;
        input2[i] = (float)(rand() % 50)/100;
        input3[i] = (float)(rand() % 50)/100;
    }

    // 初始化 filter
    for (int i = 0; i < filter_height * filter_width; i++)
    {
        filter1[i] = (float)(rand() % 50)/100;
        filter2[i] = (float)(rand() % 50)/100;
        filter3[i] = (float)(rand() % 50)/100;
    }

    // 分配CUDA空间
    float *cuda_input1;
    float *cuda_output1;
    float *cuda_filter1;
    cudaMalloc((void **)&cuda_input1, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output1, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter1, filter_height * filter_width * sizeof(float));
    float *cuda_input2;
    float *cuda_output2;
    float *cuda_filter2;
    cudaMalloc((void **)&cuda_input2, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output2, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter2, filter_height * filter_width * sizeof(float));
    float *cuda_input3;
    float *cuda_output3;
    float *cuda_filter3;
    cudaMalloc((void **)&cuda_input3, input_height * input_width * sizeof(float));
    cudaMalloc((void **)&cuda_output3, mat_height * mat_width * sizeof(float));
    cudaMalloc((void **)&cuda_filter3, filter_height * filter_width * sizeof(float));

    // CUDA 中，经过卷积运算后将三层结果相加后得到的最终结果
    float *cuda_result;
    cudaMalloc((void **)&cuda_result, mat_height * mat_width * sizeof(float));
    
    // 将host数据（矩阵和 filter）拷贝到 device中
    cudaMemcpy(cuda_input1, input1, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter1, filter1, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input2, input2, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter2, filter2, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_input3, input3, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter3, filter3, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);

    //结果初始化为0
    cudaMemset(cuda_output1, 0, mat_height * mat_width * sizeof(float));
    cudaMemset(cuda_output2, 0, mat_height * mat_width * sizeof(float));
    cudaMemset(cuda_output3, 0, mat_height * mat_width * sizeof(float));

    //设置 block 和 grid
    dim3 blockSize(filter_height, filter_width);
    dim3 gridSize(int(ceilf(mat_width / (float)blockSize.x)), int(ceilf(mat_height / (float)blockSize.y)));

    //CUDA并行计算并计时
    cudaDeviceSynchronize();
    float time_gpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    convolution_kernel<<<gridSize, blockSize>>>(cuda_output1, cuda_input1, cuda_filter1);
    convolution_kernel<<<gridSize, blockSize>>>(cuda_output2, cuda_input2, cuda_filter2);
    convolution_kernel<<<gridSize, blockSize>>>(cuda_output3, cuda_input3, cuda_filter3);
    cuda_add<<<gridSize, blockSize>>>(cuda_output1, cuda_output2, cuda_output3, cuda_result);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);
    printf("CUDA CONVOLUTION TIME:        %f ms\n", time_gpu);

    // 将结果拷贝到内存空间
    cudaMemcpy(conv_result, cuda_result, mat_height * mat_width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("RESULT:\n");
    print_mat(conv_result, mat_height, mat_width);

    // 释放空间
    free(filter1);
    free(input1);
    free(filter2);
    free(input2);
    free(filter3);
    free(input3);
    free(conv_result);
    
    cudaFree(cuda_output1);
    cudaFree(cuda_input1);
    cudaFree(cuda_filter1);
    cudaFree(cuda_output2);
    cudaFree(cuda_input2);
    cudaFree(cuda_filter2);
    cudaFree(cuda_output3);
    cudaFree(cuda_input3);
    cudaFree(cuda_filter3);
    cudaFree(cuda_result);

    return 0;
}