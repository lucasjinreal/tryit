#include <iostream>
#include "vector"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <unordered_map>


using namespace std;
using namespace google;
using namespace cv;


template<typename T>
using string_map = std::unordered_multimap<std::string, T>;


struct alignas(float) Detection
{
  float bbox[4];
  int class_id;
  float prob;
  // float landmark[10];
};

int main()
{

  // test string_map
  string_map<Detection> first;
  Detection d{{3, 4, 5, 6}, 1, 0.8};
  Detection d2{{3, 4, 56, 6}, 7, 0.8};
  first = {
    {"apple", d},
    {"banana", d2}
  };

  for (auto &elm: first) {
    cout << elm.first << ": " << elm.second.prob << endl;
  }
  cout << first.count("apple") << endl;

  return 0;
}
