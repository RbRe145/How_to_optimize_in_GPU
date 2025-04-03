#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <random>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
#define WARP_NUM (THREAD_PER_BLOCK/WARP_SIZE)

// 对warp内32个线程做规约，规约为一个值
// 规约log32次, 返回结果为warp内的
template<typename T>
__device__ T warp_reduce_sum(T sum){
    for(int stride=WARP_SIZE/2; stride>=1; stride>>=1){
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    return sum;
}
template<typename T>
__global__ void reduce7(T*input, T*output){
    int tid = threadIdx.x;
    int src_i = blockDim.x * blockIdx.x + threadIdx.x;

    T warp_sum = input[src_i];
    warp_sum = warp_reduce_sum<T>(warp_sum);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ T shmem2[WARP_NUM];
    // 每个warp内第0号线程的结果
    if(lane_id==0){
        shmem2[warp_id] = warp_sum;
    }
    __syncthreads();
    // 只有第一个warp进行reduce
    if(warp_id==0){
        warp_sum = (lane_id<WARP_NUM)? shmem2[lane_id]:static_cast<T>(0);
        warp_sum = warp_reduce_sum<T>(warp_sum);
    }
    if(tid==0){
        output[blockIdx.x] = warp_sum;
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if((out[i]-res[i])>1e-3)
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    std::random_device rd;          // 用于获得随机种子
    std::mt19937 gen(rd());         // Mersenne Twister 生成器

    // 定义均匀分布，生成 [0.0, 1.0) 之间的随机浮点数
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i=0;i<N;i++){
        a[i]=dis(gen);;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);

    reduce7<float><<<Grid,Block>>>(d_a,d_out);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            if(i>10) break;
            printf("%lf %lf \n",out[i], res[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
