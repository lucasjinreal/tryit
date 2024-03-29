#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



int main() {

    // cv::Mat image = cv::imread("../images/lanes/town01191.jpg");
    // // cv::imshow("aaa", image);
    // // cv::waitKey(0);

    // cv::Mat img_float(image.rows, image.cols, CV_32FC3);
    // image.convertTo(img_float, CV_32FC3);
    // float* imgGpuPtr;
    // cudaMalloc(&imgGpuPtr, image.rows*image.cols*3*sizeof(float));

    // cv::cuda::GpuMat imgGpuSrc;
    // cv::cuda::GpuMat imgGpu(image.rows, image.cols, CV_32FC3, 
    //     imgGpuPtr, image.cols*3*sizeof(float));
    // imgGpuSrc.upload(img_float);
    // imgGpuSrc.convertTo(imgGpuSrc, CV_32FC3);
    // imgGpuSrc.copyTo(imgGpu);


    // void *dTmp;
    // cudaMalloc(&dTmp, imgGpu.rows * imgGpu.step);
    // std::cout << "step: " << imgGpu.step  << " rows: " << imgGpu.rows << std::endl;
    // std::cout << "cols: " << image.cols * 3 << " rows: " << image.rows << std::endl;
    // // cudaMalloc(&dTmp, image.rows * image.cols * 3);
    // cudaMemcpy(dTmp, imgGpuPtr, imgGpu.rows * imgGpu.step,
    // cudaMemcpyDeviceToDevice);

    // cv::cuda::GpuMat tGpu(image.rows, image.cols, CV_32FC3,
    //     dTmp, imgGpu.step);
    // cv::Mat a;
    // tGpu.download(a);
    // a.convertTo(a, CV_8UC3);
    // cv::imshow("aa", a);
    // cv::imshow("or", image);
    // cv::waitKey(0);




    //ptr used to download result image from device
    // uint8_t *deviceInp;
    //ptr hold resized image in device memory
    float *resized_gpu_ptr;
    cv::Mat left, downloadedLeft;
    cv::cuda::GpuMat gpuLeft;
    //resize image to 257x257 using cuda::resize
    const int net_row = 257;
    const int net_col = 257;

    float *deviceInp;
    cudaMalloc((float **)&deviceInp, net_row*net_col*3*sizeof(float));
    //malloc 257x257x3 in device memory
    cudaMalloc(&resized_gpu_ptr, net_row*net_col*3*sizeof(float));

    cv::cuda::GpuMat gpu_resized(net_row, net_col, CV_32FC3, resized_gpu_ptr, net_col*3*sizeof(float));

    //read image and resize
    left = cv::imread("../images/lanes/town01191.jpg");
    gpuLeft.upload(left);
    cv::cuda::GpuMat tmp;
    gpuLeft.convertTo(tmp, CV_32FC3);
    cv::cuda::resize(tmp, gpu_resized, cv::Size(257, 257));

    //do process in tensorrt
    cudaMemcpy(deviceInp, resized_gpu_ptr,
            net_row*net_col*3*sizeof(float), cudaMemcpyDeviceToDevice);
            
    cv::cuda::GpuMat gpuImg(net_row, net_col, CV_32FC3, deviceInp, gpu_resized.step);
    //download 257x257x3 image to cpu memory
    cv::Mat a;
    gpuImg.download(a);
    a.convertTo(a, CV_8UC3);
    imshow ("test", a);
    cv::waitKey(0);


   


}