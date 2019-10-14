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

#include "thor/timer.h"

using namespace std;
using namespace cv;
using namespace google;
using namespace thor;
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

  b = a.cwiseProduct(2.*b);
  cout << b << endl;
  MatrixXf c = b.exp();
  cout << c << endl;
  MatrixXf exped = b.array().exp();
  cout << "exped:\n";
  cout << exped << endl;
  
  const int nr = 10000;
  const int nc = 5;
  MatrixXd mat = MatrixXd::Random(nr,nc);
  std::cout << "original:\n" << mat << std::endl;
  int last_col = mat.cols() - 1;

  thor::Timer timer(20);
  timer.on();
  VectorXi is_selected = (mat.col(last_col).array() > 0.3).cast<int>();
  MatrixXd mat_sel(is_selected.sum(), mat.cols());
  int rownew = 0;
  for (int i = 0; i < mat.rows(); ++i) {
    if (is_selected[i]) {       
       mat_sel.row(rownew) = mat.row(i);
       rownew++;
    }
  }
  double cost = timer.lap();
  cout << "cost: " << cost << endl;
  // cout << "select:\n" << mat_sel << endl;
  
  return 0;
}
