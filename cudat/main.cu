#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

#ifndef BLOCK
#define BLOCK 512
#endif

dim3 cudaGridSize(uint n)
{
    uint k = (n - 1) /BLOCK + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}


__global__
void add(int n, float *x, float *y) {
    for (int i=0; i<n;i++) {
        y[i] = x[i] + y[i];
    }
}


__global__
void sigmoid(int n, float *in, float* out) {
    for (int i=0; i<n; i++) {
        out[i] = 1. / (1+exp(-in[i]));
    }
}

__global__
void testKernel(int *times, int *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    out++;
    printf("%d ", idx);
}


__global__ void printThreadIndex() {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*blockDim.x * gridDim.x + ix; 
    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d, %d), global index %2d \n", 
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
}

__device__ float Logist(float data){ return 1./(1. + exp(-data)); }


__global__
void threshold(int n, float* in, float *out) {
    for(int i=0; i<n;i++) {
        out[i] = in[i] > 0.5 ? 1: 0;
    }
}


__global__
void matrixKernel(float *matrix, float *out, int c, int h, int w) {
    // doing a quick matrix multiply on a 80x244x244
    // doing matrix sigmoid on all matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h*w ) return;

    for(int cls=0; cls<c;cls++) {
        float mm = matrix[cls*w*h + idx];
        float a = Logist(mm);
        out[cls*w*h + idx] = a;
        // printf("%f %f\n", mm, a);
    }
    
}


void test1() {

    int N = 1<<20;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i=0; i<N; i++) {
        x[i] = 1.f;
        y[i] = 2.f;
    }

    cout << x[0] << " " << y[0] << endl;

    add<<<1, 1>>>(N, x, y);

    cout << y[0] << " " << y[1] << endl;

    cudaDeviceSynchronize();

    cout << y[0] << " " << y[1] << endl;

    cudaFree(x);
    cudaFree(y);

    float *logits;
    cudaMallocManaged(&logits, N*sizeof(float));
    N = 80*64*32;
    for (int i=0; i<N;i++) {
        logits[i] = 4.f;
    }

    sigmoid<<<1, 1>>>(N, logits, logits);

    cudaDeviceSynchronize();

    cout << logits[0] << endl;
}

void test2() {
    float *m_fea;
    int c=80, w=224, h =244;
    int N = c*w*h;
    printf("%d\n", N);
    cudaMallocManaged(&m_fea, N*sizeof(float));
    for(int i=0;i<N;i++) {
        m_fea[i] = 1.2f;
    }
    printf("%f %f %f \n", m_fea[0], m_fea[1], m_fea[2]);
    matrixKernel<<< w, h>>>(m_fea, m_fea, c, h, w);
    cudaDeviceSynchronize();

    printf("%f %f %f \n", m_fea[0], m_fea[1], m_fea[2]);
}


void test3() {
    dim3 grid(3, 2, 1);
    dim3 block(5, 3, 1);
    int *tt;
    cudaMallocManaged(&tt, 1*sizeof(int));
    tt = 0;
    // testKernel<<<grid, block>>>(tt, tt);
    printThreadIndex<<<grid, block>>>();
    // testKernel<<<2, 2>>>(tt, tt);
    cudaDeviceSynchronize();
    // why got 0 when access data from it?? core dumped acutally
    printf("\n%d\n", tt[0]);
}


__global__
void selectCandidates(float* in, float* out, float thresh, int shift, int* count) {
    // in is cx,cy,w,h,l,c,
    int idx = threadIdx.x + blockIdx.x;
    float v = in[idx];
    if (v > thresh) {
        // why count value doesn't changed
        // unsigned int my_d_count = atomicAdd(count, (int)1);
        atomicAdd(count, (int)1);
        // printf("add 1 now count: %d \n", count);
        // printf("add 1 now count:  \n");
        printf("d_count before atomic increment: my_d_count: %d, now: %d \n", 1, count); 
        out[*count] = v;
    }
}

void test4() {
    // 
    float* b;
    int s = 10890;
    cudaMallocManaged(&b, s*sizeof(float));
    std::cout << "print...\n";
    for (int i=0; i< s;i++) {
        b[i] = rand() % 1000 / (float)(1001);
        // b[i] = 0.6f;
        // printf("%f", b[i]);
    }
    // a里面是s个随机的值
    for(int i=0;i<30;i++) {
        printf("%f ", b[i]);
    }
    // a is a memory block of random floats
    // 我现在想把a里面大于threshold的值拿出来
    // 用CUDA来做，怎么把值拿到？
    dim3 blockSize(512);
    dim3 threadSize(s/512);
    std::cout << blockSize.x;
    std::cout << threadSize.x;
    // 执行kernel
    float* y;
    cudaMallocManaged(&y, s*sizeof(float));
    int* count;
    int t = 0;
    count = &t;

    selectCandidates << < threadSize, blockSize >> >(b, y, 0.5, s, count);

    // select out these bigger than 0.1
    cudaDeviceSynchronize();

    std::cout << "\nresult: \n";
    std::cout << "count: " << *count << std::endl;
}


int main() {
    test4();
    // test3();
    // test2();
}