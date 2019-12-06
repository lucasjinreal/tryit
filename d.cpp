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

typedef std::unordered_multimap<std::string,std::string> string_map;


int main()
{

  // test string_map
  string_map first;
  first = {
    {"apple", "red"},
    {"banana", "yellow"}
  };

  for (auto &elm: first) {
    cout << elm.first << ": " << elm.second << endl;
  }

  return 0;
}
