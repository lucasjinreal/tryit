#include <iostream>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
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

  MatrixXf a(4, 2);
  a << 2, 3, 3, 4, 45, 54, 435, 54;
  MatrixXf b(4, 2);
  b << 0.4, 0.3, 0.4, 0.7, 0.11, 0.4, 0.5, 0.8;
  cout << "a: \n" << a << endl;
  cout << "b: \n" << b << endl;

  MatrixXf vic = b.array().exp();
  cout << "vic: \n" << vic << endl;
  // transpose vic
  MatrixXf vic_t = vic.transpose();
  cout << "vic_t: \n" << vic_t << endl;

  // solve with vic
  ArrayXf conf = vic.col(1);
  cout << "conf: " << conf << endl;

  Eigen::ArrayXi indices(5);
  indices << 4, 7, 0, 2, 1;
  Eigen::Array3i cols(0, 1, 2);
  MatrixXf selected = vic(indices, cols);
  cout << "selected rows: " << selected << endl;

  return 0;
}
