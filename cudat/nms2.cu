// ------------------------------------------------------------------
// Copyright (c) AutoX



#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <math.h>

#include "thrust/sort.h"
#include <thrust/device_vector.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>



// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000
#define DIVUP(m,n) (((m)+(n)-1) / (n))
#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

int const threadsPerBlock = sizeof(unsigned long long) * 8;


template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}


template <typename T>
__device__ inline float devIoU(T const* const a, T const* const b) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  T interS = width * height;
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}


template <typename T>
__global__ void nms_kernel(
    const int n_boxes,
    const float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ATenCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

/**
dets is [5, 4]
scores is [5]
*/
// void nms_cuda(const float* dets,
//     const float* scores,
//     float iou_threshold,
//     int* out) {
  
//   at::cuda::CUDAGuard device_guard(dets.device());

//   // sort score first
//   auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
//   // sort dets with score
//   auto dets_sorted = dets.index_select(0, order_t);

//   int dets_num = dets.size(0);
//   const int col_blocks = ATenCeilDiv(static_cast<int64_t>(dets_num), static_cast<int64_t>(threadsPerBlock));

//   at::Tensor mask =
//       at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));
//   // make mask gpu located memory
  

//   dim3 blocks(col_blocks, col_blocks);
//   dim3 threads(threadsPerBlock);
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
//   auto scalar_t = dets_sorted.type();

//   // not support f16 for now
//   nms_kernel<float><<<blocks, threads, 0, stream>>>(
//     dets_num,
//     iou_threshold,
//     dets_sorted.data_ptr<float>(),
//     (unsigned long long*)mask.data_ptr<int64_t>());
  
//   // mask: which box keep
//   at::Tensor mask_cpu = mask.to(at::kCPU);
//   unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

//   std::vector<unsigned long long> remv(col_blocks);
//   memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

//   at::Tensor keep =
//       at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
//   int64_t* keep_out = keep.data_ptr<int64_t>();

//   int num_to_keep = 0;
//   for (int i = 0; i < dets_num; i++) {
//     int nblock = i / threadsPerBlock;
//     int inblock = i % threadsPerBlock;
//     if (!(remv[nblock] & (1ULL << inblock))) {
//       keep_out[num_to_keep++] = i;
//       unsigned long long* p = mask_host + i * col_blocks;
//       for (int j = nblock; j < col_blocks; j++) {
//         remv[j] |= p[j];
//       }
//     }
//   }
//   AT_CUDA_CHECK(cudaGetLastError());
//   return order_t.index(
//       {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
//            .to(order_t.device(), keep.scalar_type())});
// }


int main(void)
{   
  std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
  
  float dets[6][4] = {
    {23, 34, 56, 76},
    {11, 23, 45, 45},
    {12, 22, 47, 47},
    {9, 45, 56, 65},
    {20, 37, 55, 75},
  };
  float scores[6] = {0.7, 0.6, 0.8, 0.4, 0.2, 0.6};
  float iou_threshold = 0.2;
  // copy data to gpu
  std::cout << sizeof(dets) << std::endl;
  std::cout << sizeof(scores) << std::endl;

  float *dev_dets, *dev_scores;
  int *dev_indices;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&dev_scores, sizeof(scores));
  err = cudaMalloc((void **)&dev_dets, sizeof(dets));
  err = cudaMalloc((void **)&dev_indices, (sizeof(scores)/sizeof(float))*sizeof(int));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed!");
    return 1;
  }
  cudaMemcpy(dev_dets, dets, sizeof(dets), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_scores, scores, sizeof(scores), cudaMemcpyHostToDevice);
  std::cout << "copied data to GPU.\n";

  // DEBUG: get back copied cuda data
  float host_dets[sizeof(dets)/sizeof(float)];
  float host_scores[6];
  cudaMemcpy(&host_dets, dev_dets, sizeof(dets), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_scores, dev_scores, sizeof(scores), cudaMemcpyDeviceToHost);
  std::cout << "copied from cuda back to host.\n";
  std::cout << "host_dets size: " << sizeof(host_dets) << std::endl;
  for (int i=0;i<sizeof(dets)/sizeof(float);i++) {
    std::cout << host_dets[i] << " ";
  }
  std::cout << std::endl;
  for (int i=0;i<sizeof(scores)/sizeof(float);i++) {
    std::cout << static_cast<float>(host_scores[i]) << " ";
  }
  std::cout << std::endl;

  // let's try sort scores using thrust
  // ----------------------- Method 1 ---------------------------------
  thrust::device_vector<int> sorted_indices(sizeof(scores)/sizeof(float));
  thrust::sequence(sorted_indices.begin(), sorted_indices.end(), 0);
  thrust::sort_by_key(thrust::device, dev_scores, dev_scores+sizeof(scores)/sizeof(float), sorted_indices.begin());
  printf("sorted done.\n");
  cudaMemcpy(&host_scores, dev_scores, sizeof(scores), cudaMemcpyDeviceToHost);
  for (int i=0;i<sizeof(scores)/sizeof(float);i++) {
    std::cout << static_cast<float>(host_scores[i]) << " ";
  }
  std::cout << std::endl;
  for(auto index: sorted_indices) {
    std::cout << index << " ";
  }
  std::cout << std::endl;
  
  // Determine temporary device storage requirements
  // void     *d_temp_storage = NULL;
  // size_t   temp_storage_bytes = 0;
  // thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
  // )
  

  // nms_cuda()

  cudaFree(dev_dets);
  cudaFree(dev_scores);

  std::cout << "done.\n";

}
