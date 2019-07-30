/**
 *
 *
 * testing for convolution algorithm
 *
 * exploring for winograd algorithm
 *
 * */

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  Mat M;
  M = imread(argv[1]);
  M.convertTo(M, CV_32S);
  // M = M - 300;
  cv::subtract(M, cv::Scalar(103.94, 116.78, 123.68), M);
  cout << M << endl;

  float h = 0.23;
  cout << h * 6.0 << endl;
  int i =  h * 6.0;
  cout << i << endl;
  cout << i%6 << endl;
}
