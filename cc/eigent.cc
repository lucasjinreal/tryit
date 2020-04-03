#include <Eigen/Core>
#include <Eigen/Eigen>
#include <iostream>
#include <queue>
#include <deque>


using namespace std;
using namespace Eigen;


int main() {

    Eigen::VectorXf a(6);
    a << 0, 3, 5, 7, 6, 9;

    cout << a[4] << " " << a[3];

    deque<Eigen::VectorXf> trace;
    Eigen::VectorXf one_p(2);
    one_p << 3, 44;
    trace.push_back(one_p);
    trace.push_back(one_p);
    trace.push_front(one_p);
    cout << "dont\n";
    Eigen::VectorXf op = trace[1];
    cout << op << endl;
}