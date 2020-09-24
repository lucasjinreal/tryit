#include <cmath>
#include <iostream>
#include <unordered_map>

#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "vector"

using namespace std;
using namespace google;
using namespace cv;

template <typename T>
using string_map = std::unordered_multimap<std::string, T>;

struct alignas(float) Detection {
  float bbox[4];
  int class_id;
  float prob;
  // float landmark[10];
};

void infer(vector<Detection> *det_res) {
  for (size_t i = 0; i < 4; i++) {
    /* code */
    Detection d{{3, 4, 5, 6}, 1, 0.8};
    det_res->emplace_back(d);
  }
}

int main() {
  // test string_map
  vector<float> oneData = {34., 4., 4., 5., 45, 54, 54};
  vector<vector<float>> indata;
  indata.push_back(oneData);
  cout << indata.size() << endl;
  assert(oneData.size() == 8);

  return 0;
}
