#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA kernel to perform softmax on each row of a 2D array
__global__ void softmax_v0(float* input, float* output, int rows, int cols) {
    __shared__ float s_max;
    __shared__ float s_sum;

    int row = blockIdx.x;
    if (row < rows) {
        // Find the maximum value in the current row
        float max_val = -INFINITY;
        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
            int index = row * cols + tid;
            if (input[index] > max_val) {
                max_val = input[index];
            }
        }
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        // Compute the sum of exp(x - max_val) for each element in the current row
        float local_sum = 0.0f;
        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
            int index = row * cols + tid;
            output[index] = expf(input[index] - s_max);
            local_sum += output[index];
        }
        __syncthreads();

        // Use reduction to calculate the total sum in the block
        __shared__ float partial_sums[256];
        partial_sums[threadIdx.x] = local_sum;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (threadIdx.x < s) {
                partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
            }
        }
        if (threadIdx.x == 0) {
            s_sum = partial_sums[0];
        }
        __syncthreads();

        // Divide each element by the sum to get the softmax values
        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
            int index = row * cols + tid;
            output[index] /= s_sum;
        }
    }
}

__device__ float warp_reduce_sum(float val){
    for(int stride=32/2; stride>=1; stride>>=1){
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
    return val;
}

__device__ float warp_reduce_max(float val){
    for(int stride=32/2; stride>=1; stride>>=1){
        val = max(val, __shfl_down_sync(0xffffffff, val, stride));
    }
    return val;
}

// __device__ float block_reduce_sum(float val){
//     int tid = threadIdx.x;
//     int lane_id = tid % 32;
//     int warp_id = tid / 32;

// }
// input, output: [rows, cols]
__global__ void softmax_v1(float* input, float* output, int rows, int cols) {
    int tid = threadIdx.x;
    int src_id = blockDim.x * blockIdx.x + tid;
    int dst_id = src_id;

    __shared__ float row_max;
    __shared__ float row_sum;

    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // 1st reduce get max, sum in each warp
    float max = input[src_id];
    max = warp_reduce_max(max);
    // 最先进的gpu blockDim=2048
    // 两轮reduce能够覆盖32*32范围的数据
    // 因此一般情况下两轮绝对够用
    __shared__ float warp_max[512/32];
    __shared__ float warp_sum[512/32];
   
    if(lane_id==0){
        warp_max[warp_id] = max;
        // warp_sum[warp_id] = sum;
    }
    __syncthreads();

    if(warp_id==0){
        max = warp_max[lane_id];
        // sum = warp_sum[lane_id];
        max = warp_reduce_max(max);
        // sum = (lane_id<32)?warp_reduce_sum(sum):(float)0;
    }

    if(lane_id==0 && warp_id==0){
        row_max = max;
    }
    __syncthreads();
    // 可以broadcast
    float sum = expf(input[src_id] - row_max);
    sum = warp_reduce_max(sum);
    if(lane_id==0){
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    if(warp_id==0){
        sum = warp_sum[lane_id];
        sum = warp_reduce_sum(sum);
    }

    if(lane_id==0 && warp_id==0){
        row_sum = sum;
    }
    __syncthreads();
    output[dst_id] = expf(input[src_id] - row_max) / row_sum;
}

// Function to perform softmax on a 2D array using CUDA
void softmax_cuda(float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    // Allocate device memory
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = cols;
    int gridSize = rows;//[rows, cols]

    // // Launch the CUDA kernel
    // softmax_v0<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    // // Copy output data from device to host
    // cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout<<"v0"<<std::endl;
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         std::cout << output[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    softmax_v1<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    // Copy output data from device to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"v1"<<std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << output[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int rows = 8;
    int cols = 16;
    float* input = new float[rows * cols];
    float* output = new float[rows * cols];

    // Initialize input data
    for (int i = 0; i < rows * cols; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform softmax using CUDA
    softmax_cuda(input, output, rows, cols);

    // Print the output
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         std::cout << output[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Free host memory
    delete[] input;
    delete[] output;

    return 0;
}