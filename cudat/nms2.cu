// ------------------------------------------------------------------
// Copyright (c) AutoX



#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <math.h>



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

  float *dev_dets, *dev_scores;
  cudaMalloc((void **) &dev_dets, sizeof(dets));
  std::cout << "cuda malloced.\n";

  // get back copied cuda data
  float *host_dets;
  cudaMemcpy(host_dets, dev_dets, sizeof(dets), cudaMemcpyDeviceToHost);
  std::cout << "copied from cuda back to host.\n";
  for (int i=0;i<sizeof(dets)/sizeof(float);i++) {
    std::cout << host_dets[i] << " ";
  }


  // auto indices = nms_cuda(dets, scores, iou_threshold);
  // std::cout << indices << std::endl;
}
