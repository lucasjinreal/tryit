/**
 *
 * Convert Haikang Camera YV12 to OpenCV BGR format and show
 *
 */
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

void yv12toYUV(uchar *outYuv, uchar *inYv12, int width, int height,
               int widthStep) {
  int col, row;
  unsigned int Y, U, V;
  int tmp;
  int idx;
  for (row = 0; row < height; row++) {
    idx = row * widthStep;
    int rowptr = row * width;
    for (col = 0; col < width; col++) {
      tmp = (row / 2) * (width / 2) + (col / 2);
      Y = (unsigned int)inYv12[row * width + col];
      U = (unsigned int)inYv12[width * height + width * height / 4 + tmp];
      V = (unsigned int)inYv12[width * height + tmp];
      outYuv[idx + col * 3] = Y;
      outYuv[idx + col * 3 + 1] = U;
      outYuv[idx + col * 3 + 2] = V;
    }
  }
}

int main() {
    
    // 从SDK拿到一帧的数据PFrameInfo, 将其内存传给cv::Mat
    // 此时默认是YV12模式, 通过调用opencv方法将YV12转到YUV再转到BGR
  Mat dst(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
  Mat src(pFrameInfo->nHeight + pFrameInfo->nHeight / 2, pFrameInfo->nWidth,
          CV_8UC1, (uchar *)pBuf);
  // converts from yuv2 to BGR here
  cvtColor(src, dst, CV_YUV2BGR_YV12);
  // 如果BGR能正常显示(颜色没有变色),则可以使用
  imshow("bgr", dst);
  waitKey(1);
}