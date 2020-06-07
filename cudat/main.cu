#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

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
void threshold(int n, float* in, float *out) {
    
}


int main() {

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