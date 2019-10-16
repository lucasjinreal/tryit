#include <iostream>
#include "vector"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

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

// define a structure
struct alignas(float) Detection{
  float bbox[4];
  int class_id;
  float prob;
  // float landmark[10];
};


int main()
{

  int a = RUN_MODE::INT8;
  cout << a << endl;

  int n_dets = 100;
  float* data = new float[n_dets * 6];
  for (int i = 0; i < n_dets*6; i++)
  {
    // 600
    data[i] = 334.;
  }

  // new we convert data into Detections
  vector<Detection> result;
  result.resize(n_dets);
  memcpy(result.data(), data, n_dets*sizeof(Detection));
  
  cout << "result size: " << result.size() << endl;
  Detection d1 = result[99];
  cout << d1.bbox[0] << " " << d1.bbox[1] << " cls_prob: " << d1.class_id << " " << d1.prob << endl;

  int INPUT_W = 1023;
  int step = 34;
  int aaa = ceil(INPUT_W / step);
  cout << aaa << endl;

  float bb = exp(2.3);
  cout << bb << endl;
  return 0;
}
