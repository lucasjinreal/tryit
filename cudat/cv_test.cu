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

    cv::Mat image = cv::imread("../images/lanes/town01191.jpg");
    cv::imshow("aaa", image);
    cv::waitKey(0);

    cv::Mat img_float(image.rows, image.cols, CV_32FC3);
    image.convertTo(img_float, CV_32FC3);
    cv::cuda::GpuMat imgGpuSrc;
    imgGpuSrc.upload(img_float);


    void *dTmp;
    cudaMalloc(&dTmp, imgGpuSrc.rows * imgGpuSrc.step);
    cudaMemcpy(dTmp, imgGpuSrc.ptr<float>(), imgGpuSrc.rows * imgGpuSrc.step,
    cudaMemcpyDeviceToDevice);

    cv::cuda::GpuMat tGpu(image.rows, image.cols, CV_32FC3,
        dTmp, imgGpuSrc.step);
    cv::Mat a;
    tGpu.download(a);
    a.convertTo(a, CV_8UC3);
    cv::imshow("aa", a);
    cv::imshow("or", image);
    cv::waitKey(0);

   


}