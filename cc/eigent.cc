#include <Eigen/Core>
#include <Eigen/Eigen>
#include <iostream>

using namespace std;
using namespace Eigen;


int main() {

    Eigen::VectorXf a(6);
    a << 0, 3, 5, 7, 6, 9;

    cout << a[4] << " " << a[3];
}