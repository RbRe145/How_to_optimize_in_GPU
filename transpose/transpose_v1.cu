#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#include <iostream>
#include <iomanip>

// use 2-D threadblock to transpose
// in(M*N) out(N*M)
template<typename T, int M, int N>
__global__ void transpose(T* in, T*out){
    int src_row = blockDim.y * blockIdx.y + threadIdx.y;
    int src_col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int s_data[32][32];
    if(src_row<M && src_col<N){
        s_data[threadIdx.y][threadIdx.x] = in[src_row*N + src_col];
        __syncthreads();
        int dst_row = src_col;
        int dst_col = src_row;
        out[dst_row*M + dst_col] = s_data[threadIdx.y][threadIdx.x];
    }
}

template<typename T, int M, int N>
void transpose_cpu(T* in, T*out){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            out[j*M+i] = in[i*N+j];
        }
    }
}

template<typename T, int M, int N>
bool check(T* cpu_ans, T* gpu_ans){
    // both are N*M
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            if(abs(cpu_ans[i*M+j]-gpu_ans[i*M+j])>1e-6){
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void printMatrix(const T* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // 使用 std::setw 控制宽度，保证对齐效果
            std::cout << std::setw(3) << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}
int main(){
    const int M=1024, N=1024;
    dim3 block_size(32, 32);
    dim3 grid_size(M/32, N/32);
    
    float* h_in = (float*)malloc(M*N*sizeof(float));
    float* h_out = (float*)malloc(N*M*sizeof(float));
    float* d_out_cpu = (float*)malloc(N*M*sizeof(float));

    float* d_in, *d_out;
    cudaMalloc((void**)&d_in, M*N*sizeof(float));
    cudaMalloc((void**)&d_out, M*N*sizeof(float));

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            h_in[i*N+j] = (float)j;
        }
    }
    cudaMemcpy(d_in, h_in, M*N*sizeof(float), cudaMemcpyHostToDevice);
    transpose<float, M, N><<<grid_size, block_size>>>(d_in, d_out);
    cudaMemcpy(d_out_cpu, d_out, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    transpose_cpu<float, M, N>(h_in, h_out);
    if(check<float, M, N>(d_out_cpu, h_out)){
        printf("Pass \n");
    }else{
        printf("Wrong \n");
        printf("------------------------ \n");
        printMatrix<float>(h_in, M, N);
        printf("------------------------ \n");
        printMatrix<float>(d_out_cpu, N, M);
        printf("------------------------ \n");
        printMatrix<float>(h_out, N, M);

    }
}