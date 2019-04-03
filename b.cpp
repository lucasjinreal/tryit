#include <iostream>
#include "vector"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace google;
using namespace cv;

template <typename Dtype>
class FeatureGenerator
{
public:
  FeatureGenerator();
  bool Init(vector<Dtype> v_a, bool is_last);
  void run();

private:
  Dtype v;
};

template <typename Dtype>
FeatureGenerator<Dtype>::FeatureGenerator(){

};

template <typename Dtype>
bool FeatureGenerator<Dtype>::Init(vector<Dtype> v_a, bool is_last)
{
  if (is_last)
  {
    v = v_a.back();
  }
  else
  {
    v = v_a.front();
  }
  return true;
};

template <typename Dtype>
void FeatureGenerator<Dtype>::run()
{
  LOG(INFO) << v;
};

// template FeatureGenerator<Dtype>::FeatureGenerator{
//   LOG(INFO) << "generated.";
// }

int main()
{

  // Mat image(400, 400, CV_8UC4, Scalar(0, 0, 200));
  Mat img = imread("image.png", cv::IMREAD_UNCHANGED);
  Mat image(img.size(), CV_8UC4, img.data);
  cout << image.size() << "x" << image.channels() << endl;

  Mat bgra[4];
  split(image, bgra);

  rectangle(bgra[3], Rect(100, 100, 200, 200), Scalar(255), -1);

  merge(bgra, 4, image);
  imwrite("transparent1.png", image);

  rectangle(bgra[3], Rect(150, 150, 100, 100), Scalar(127), -1);
  merge(bgra, 4, image);
  imwrite("transparent2.png", image);

  bgra[3] = bgra[3] * 0.5; // here you can change transparency %50
  // bgra[3] = bgra[3] + 50 // you can add some value
  // bgra[3] = bgra[3] - 50 // you can subtract some value

  merge(bgra, 4, image);
  imwrite("transparent3.png", image);
  imshow("aa", image);

  waitKey(0);

  return 0;
}
