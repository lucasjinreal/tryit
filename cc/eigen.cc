#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "eigen3/Eigen/Dense"

using namespace std;

Eigen::Matrix3d Quaternion2RotationMatrix(const double x, const double y,
                                          const double z, const double w) {
  Eigen::Quaterniond q;
  q.x() = x;
  q.y() = y;
  q.z() = z;
  q.w() = w;
  Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  cout << "Quaternion2RotationMatrix result is:" << endl;
  cout << "R = " << endl << R << endl;
  return R;
}

Eigen::Quaterniond EulerAngle2Quaternion(const double yaw, const double pitch,
                                        const double roll) {
  // convert yaw, pitch, roll to quaternion
  Eigen::Vector3d ea0(yaw, pitch, roll);
  Eigen::Matrix3d R;
  R = ::Eigen::AngleAxisd(ea0[0], ::Eigen::Vector3d::UnitZ()) *
      ::Eigen::AngleAxisd(ea0[1], ::Eigen::Vector3d::UnitY()) *
      ::Eigen::AngleAxisd(ea0[2], ::Eigen::Vector3d::UnitX());
  // RotationMatrix to Quaterniond
  Eigen::Quaterniond q;
  q = R;
  return q;
}

int main() {
  Eigen::MatrixXd a(3, 4);
  a << 2, 4, 5, 6, 3, 5, 6, 6, 4, 5, 1, 4;

  Eigen::MatrixXf b(3, 1);
  b << 1, 1, 1;

  cout << a << endl;
  cout << a(1) << endl;

  // Eigen::MatrixXf c(3, 4);
  //   Eigen::MatrixXf R(3, 3);
  //  R << 0.03, 0.1, 1.2,
  //  0.4, 0.5, 2.2,
  //  0.3, 0.4, 0.6;
  //  cout << R*a << endl;

  Eigen::Matrix3d R;
  R = Quaternion2RotationMatrix(0.1, 0.2, 0.4, 0);

  Eigen::MatrixXd res(3, 4);
  res = R * a;
  cout << res << endl;

  for (int i = 3; i >= 0; i--) {
    cout << i << endl;
  }

  double nnn = 3;
  cout << nnn / 2 << " vs " << nnn / 2.0 << endl;

  
  double yaw = 0.2;
  double pitch = 0;
  double roll = 0.1;
  Eigen::Quaterniond q = EulerAngle2Quaternion(yaw, pitch, roll);
  cout << "q: " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() <<  endl;

  return 0;
}