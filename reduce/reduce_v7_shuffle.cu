#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

// 对warp内32个线程做规约，规约为一个值
// 规约log32次, 返回结果为warp内的
template<typename T>
__device__ T warp_reduce_sum(T sum){
    for(int stride=WARP_SIZE/2; stride>=1; stride>>=1){
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    return sum;
}
template<typename T, int warp_num>
__global__ void reduce7(T*input, T*output){
    int tid = threadIdx.x;
    int src_i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ T shmem[THREAD_PER_BLOCK];
    shmem[tid] = input[src_i];
    __syncthreads();
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    T warp_sum = shmem[tid];
    warp_sum = warp_reduce_sum<T>(warp_sum);
    __shared__ T shmem2[warp_num];
    // 每个warp内第0号线程的结果
    if(lane_id==0){
        shmem2[warp_id] = warp_sum;
    }
    __syncthreads();
    // 只有第一个warp进行reduce
    if(warp_id==0){
        warp_sum = (lane_id<warp_num)? shmem2[lane_id]:static_cast<T>(0);
        warp_sum = warp_reduce_sum<T>(warp_sum);
    }
    if(tid==0){
        output[blockIdx.x] = warp_sum;
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024; // 32M data
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    const int block_num = 1024;
    const int NUM_PER_BLOCK = N / block_num;
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
    float *out=(float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,block_num*sizeof(float));
    float *res=(float *)malloc(block_num*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=i%456;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            if(i * NUM_PER_BLOCK + j < N){
                cur+=a[i * NUM_PER_BLOCK + j];
            }
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( block_num, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);

    reduce7<float, THREAD_PER_BLOCK/WARP_SIZE><<<Grid,Block>>>(d_a, d_out);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf %lf \n",out[i], res[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
