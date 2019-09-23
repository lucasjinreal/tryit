#include <iostream>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Eigen"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;
using namespace google;

using namespace Eigen;

int main() {
  int n_dets = 100;
  float* data = new float[n_dets * 6];
  for (int i = 0; i < n_dets * 6; i++) {
    // 600
    data[i] = 334.;
  }

  // we want convert data into a Matrix
  MatrixXf m1;
  m1 = Map<Matrix<float, 100, 6, RowMajor> >(data);

  cout << m1.size() << endl;
  cout << m1.rows() << "x" << m1.cols() << endl;
  cout << m1 << endl;

  // do some slice
  vector<float> variances = {0.1, 0.1};
  MatrixXf res;
  // res = m1.block<0, 0>(m1.rows(), 1);
  cout << endl << endl;
  // cout << res << endl;
  res = m1.leftCols(2);
  cout << res << endl;

  MatrixXf right_part;
  right_part = res.cwiseProduct(m1.rightCols(2) * variances[0]);

  cout << right_part << endl;
  cout << right_part.size() << endl;

  MatrixXf a(2, 2);
  a << 2, 3, 
    3, 4;
  MatrixXf b(2, 2);
  b << 0.4, 0.3,
  0.4, 0.7;
  cout << a << endl;

  cout << b << endl;

  b = a.cwiseProduct(2.*b);
  cout << b << endl;
  MatrixXf c = b.exp();
  cout << c << endl;
  cout << b.exp() << endl;

  return 0;
}
