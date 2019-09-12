#include <iostream>
#include "vector"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace google;
using namespace cv;

namespace RUN_MODE
{
  enum RUN_MODE {
  FLOAT32 = 0,
  FLOAT16 = 1,
  INT8 = 2,
};
} // namespace RUN_MODE


int main()
{

  int a = RUN_MODE::INT8;
  cout << a << endl;

  return 0;
}
