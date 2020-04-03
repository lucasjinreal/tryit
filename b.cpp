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
enum RUN_MODE
{
  FLOAT32 = 0,
  FLOAT16 = 1,
  INT8 = 2,
};
} // namespace RUN_MODE

// define a structure
struct alignas(float) Detection
{
  float bbox[4];
  int class_id;
  float prob;
  // float landmark[10];
};


void getDir(const cv::Point src_p, cv::Mat res_p, float r_rad) {
  float sn = sin(r_rad);
  float cs = cos(r_rad);
  float x = src_p.x * cs - src_p.y*sn;
  float y = src_p.x*sn + src_p.y*cs;
    res_p = (cv::Mat_<float>(1, 2) << x, y);
}

cv::Mat get3rdPoint(cv::Mat a, cv::Mat b) {
    /**
     * def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
     */
    // a: (1,2)
    cv::Mat direct = a - b;
    direct = b + (cv::Mat_<float>(1, 2) << -direct.at<float>(0, 1), direct.at<float>(0, 0));
    return direct;
}


cv::Mat getAffineT(cv::Mat center, cv::Mat scale_size,
                           float rot, cv::Point output_size,
                           cv::Mat shift,
                           bool inv = false)
{
  // center:(1,2), scale_size:(1,2)
  // actually, IDK what this function doing
  int dst_w = output_size.x;
  int dst_h = output_size.y;
  float rot_rad = M_PI * rot / 180;
  cv::Mat src_dir;
  int scale_x = scale_size.at<float>(0, 0);
  getDir(cv::Point(0, scale_x * -0.5), src_dir, rot_rad);
  float dst_dir[2] = {0, (float)(-0.5 * dst_w)};

  // src 3x2, dst 3x2
  cv::Mat src = cv::Mat_<float>(3, 2);
  cv::Mat dst = cv::Mat_<float>(3, 2);

  cv::Mat src0 = center + scale_size.mul(shift);
  memcpy(src.data, src0.data, sizeof(float) * 2);
  cv::Mat src1 = center + src_dir + scale_size.mul(shift);
  memcpy(src.data + sizeof(float) * 2, src1.data, sizeof(float) * 2);
  cv::Mat dst0 = (cv::Mat_<float>(1, 2) << dst_w * 0.5, dst_h * 0.5);
  memcpy(dst.data, dst0.data, sizeof(float) * 2);
  cv::Mat dst1 = (cv::Mat_<float>(1, 2) << dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
  memcpy(dst.data + sizeof(float) * 2, dst1.data, sizeof(float));

  cv::Mat src2 = get3rdPoint(src0, src1);
  cv::Mat dst2 = get3rdPoint(dst0, dst1);
  memcpy(src.data + sizeof(float) * 4, src2.data, sizeof(float) * 2);
  memcpy(dst.data + sizeof(float) * 4, dst2.data, sizeof(float) * 2);
  cout << "src:\n" << src << endl;
  cout << "dst:\n" << dst << endl;
  if (inv)
  {
    return cv::getAffineTransform(dst, src);
  }
  else
  {
    return cv::getAffineTransform(src, dst);
  }
}

int main()
{

  int a = RUN_MODE::INT8;
  cout << a << endl;

  int n_dets = 100;
  float *data = new float[n_dets * 6];
  for (int i = 0; i < n_dets * 6; i++)
  {
    // 600
    data[i] = 334.;
  }

  int INPUT_W = 512;
  int INPUT_H = 512;
  cv::Mat c = (cv::Mat_<float>(1, 2) << INPUT_W / 2, INPUT_H / 2);
  int s = max(INPUT_W, INPUT_H);
  cv::Mat scale_size = (cv::Mat_<float>(1, 2) << s, s);
  cv::Point output_size = cv::Point(INPUT_W, INPUT_H);
  cv::Mat shift = (cv::Mat_<float>(1, 2) << 0, 0);
  cv::Mat trans_input = getAffineT(c, scale_size, 0, output_size, shift);
  cout << "trans_input: \n" << trans_input << endl;
  return 0;
}
