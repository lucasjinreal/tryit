#include <iostream>
#include "vector"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace google;
using namespace cv;



int main()
{

   uint16_t a = 1.234;
   cout << a << endl;
   printf("%u", a);
   uint16_t b = 0.355;
   cout << a + b << endl;
   printf("%hu", a+b);
  return 0;
}
