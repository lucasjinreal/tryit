#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"

using namespace std;


__global__
void helloFromGPU() {
    printf("hello from GPU\n");
}


int main(void)
{

    float *dx;
    cudaMalloc(&dx, 1<<4*sizeof(float));
    printf("hello from cpu.\n");
    helloFromGPU<<<1, 10>>>();

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    size_t mem_avail;
    size_t mem_total;
    cudaMemGetInfo(&mem_avail, &mem_total);
    cout << "device: " << devProp.name << endl;
    cout << "memoery: " << mem_total/1024./1024./1000.<< "G" << endl;
    cout << "memoery avail: " << mem_avail/1024./1024./1000. << "G" << endl;
    cout << "SM: " << devProp.multiProcessorCount << endl;
    cout << "shared memory: " << devProp.sharedMemPerBlock << endl;
    cout << "max threads: " << devProp.maxThreadsPerBlock << endl;
    cout << "";
    cout << "cuda query done.\n";
}
