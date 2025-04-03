#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// bank conflict
template<typename T>
__global__ void reduce1(T* input, T* output){
    __shared__ T shmem[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    int src_i = blockDim.x*blockIdx.x + threadIdx.x;
    shmem[tid] = input[src_i];
    int warp_size = 128 / sizeof(T);
    __syncthreads();
    for(int i=blockDim.x/2; i>=1; i>>=1){
        if(tid < i){
            shmem[tid] += shmem[tid+i];
        }
        // i<=warp_size的时候，只有warp0在工作，不需要同步其他warp内的线程
        if(i>warp_size){
            __syncthreads();
        }
    }
    if(tid==0) output[blockIdx.x] = shmem[tid];
}
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
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

    for(int i=0;i<N;i++){
        a[i]=1;
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

    reduce1<float><<<Grid,Block>>>(d_a,d_out);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
