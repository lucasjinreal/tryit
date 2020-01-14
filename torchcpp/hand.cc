// This file is the demo code for the paper
//
// "SRHandNet: Real-time 2D Hand Pose Estimation with Simultaneous Region
// Localization"
//
//  - the code relies on Pytorch and OpenCV
//  - the code needs the Pytorch model in our webpage
//  - the code works for the real-time camera input
//  - the code is only tested under Release MODE
//  - you may disable SDL check if you use Visual Studio to compile the code
//
// For more information, please visite our webpage from
// https://www.yangangwang.com/papers/WANG-SRH-2019-07.html
//
// Yangang Wang @ seu
// 2019-2-19

#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace torch;
using namespace std;
using namespace cv;

constexpr auto TRAIN_IMAGE_HEIGHT = 256;
constexpr auto TRAIN_IMAGE_WIDTH = 256;
constexpr auto LABEL_MIN = 0.2;

const vector<float> HAND_COLORS_RENDER{
    100.f, 100.f, 100.f, 100.f, 0.f,   0.f,   150.f, 0.f,   0.f,   200.f, 0.f,
    0.f,   255.f, 0.f,   0.f,   100.f, 100.f, 0.f,   150.f, 150.f, 0.f,   200.f,
    200.f, 0.f,   255.f, 255.f, 0.f,   0.f,   100.f, 50.f,  0.f,   150.f, 75.f,
    0.f,   200.f, 100.f, 0.f,   255.f, 125.f, 0.f,   50.f,  100.f, 0.f,   75.f,
    150.f, 0.f,   100.f, 200.f, 0.f,   125.f, 255.f, 100.f, 0.f,   100.f, 150.f,
    0.f,   150.f, 200.f, 0.f,   200.f, 255.f, 0.f,   255.f};

void nmslocation(Mat& src, map<float, Point2f, greater<float>>& location,
                 float threshold) {
  // clear all the points
  location.clear();

  // set the local window size: 5*5
  int blockwidth = 2;

  // for each pixel window, search the local maximum
#pragma omp parallel for
  for (int i = blockwidth; i < src.cols - blockwidth; i++) {
    for (int j = blockwidth; j < src.rows - blockwidth; j++) {
      Point2i tmploc(i, j);

      // candidate keypoint point
      float localvalue = src.at<float>(tmploc);
      if (localvalue < threshold) continue;

      // check whether it is local maximum
      bool localmaximum = true;
      for (int m = max(tmploc.x - blockwidth, 0);
           m <= min(tmploc.x + blockwidth, src.cols - 1); m++) {
        for (int n = max(tmploc.y - blockwidth, 0);
             n <= min(tmploc.y + blockwidth, src.rows - 1); n++) {
          if (src.at<float>(Point2i(m, n)) > localvalue) {
            localmaximum = false;
            break;
          }
        }
        if (!localmaximum) break;
      }

      // output the location
      if (localmaximum) {
#pragma omp critical
        {
          if (localmaximum) {
            location.insert(make_pair(localvalue, tmploc));
          }
        }
      }
    }
  }
}

float transformNetInput(Tensor& inputTensor, const Mat& src_img,
                        int tensor_index = 0) {
  // lmbada expression
  auto fastmin = [](float a, float b) { return a < b ? a : b; };

  // convert the input image
  Mat dst;
  float ratio = fastmin(float(inputTensor.size(2)) / src_img.rows,
                        float(inputTensor.size(3)) / src_img.cols);
  Mat M = (Mat_<float>(2, 3) << ratio, 0, 0, 0, ratio, 0);
  warpAffine(src_img, dst, M, Size(inputTensor.size(3), inputTensor.size(2)),
             INTER_CUBIC, BORDER_CONSTANT, cv::Scalar(128, 128, 128));

  dst.convertTo(dst, CV_32F);
  dst = dst / 255.f - 0.5f;
  vector<Mat> chn_img;
  split(dst, chn_img);

  size_t total_bytes =
      sizeof(float) * inputTensor.size(2) * inputTensor.size(3);
  if (chn_img.size() == 1) {
    memcpy(inputTensor[tensor_index][0].data_ptr<float>(),
           (float*)chn_img[0].data, total_bytes);
    memcpy(inputTensor[tensor_index][1].data_ptr<float>(),
           (float*)chn_img[0].data, total_bytes);
    memcpy(inputTensor[tensor_index][2].data_ptr<float>(),
           (float*)chn_img[0].data, total_bytes);
  } else {
    memcpy(inputTensor[tensor_index][0].data_ptr<float>(),
           (float*)chn_img[0].data, total_bytes);
    memcpy(inputTensor[tensor_index][1].data_ptr<float>(),
           (float*)chn_img[1].data, total_bytes);
    memcpy(inputTensor[tensor_index][2].data_ptr<float>(),
           (float*)chn_img[2].data, total_bytes);
  }
  return ratio;
}

void detectBbox(vector<Rect>& handrect, jit::script::Module& model,
                const Mat& inputImage) {
  // init the tensor
  auto inputTensor =
      torch::zeros({1, 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH});

  // transform the input data
  float ratio_input_to_net = transformNetInput(inputTensor, inputImage);

  // run the network
  auto heatmap =
      model.forward({inputTensor.cuda()}).toTuple()->elements()[3].toTensor();

  // copy the 3-channel rect map
  vector<Mat> rectmap(3);
  float ratio_net_downsample = TRAIN_IMAGE_HEIGHT / float(heatmap.size(2));
  int rect_map_idx = heatmap.size(1) - 3;
  for (int i = 0; i < 3; i++) {
    rectmap[i] = Mat::zeros(heatmap.size(2), heatmap.size(3), CV_32FC1);
    auto ptr = heatmap[0][i + rect_map_idx].cpu().data_ptr<float>();
    memcpy((float*)rectmap[i].data, ptr,
           sizeof(float) * heatmap.size(2) * heatmap.size(3));
  }
  map<float, Point2f, greater<float>> locations;
  nmslocation(rectmap[0], locations, LABEL_MIN);
  handrect.clear();
  for (auto iter = locations.begin(); iter != locations.end(); iter++) {
    Point2f points = iter->second;
    int pos_x = points.x;
    int pos_y = points.y;
    float ratio_width = 0.f, ratio_height = 0.f;
    int pixelcount = 0;
    for (int m = max(pos_y - 2, 0); m < min(pos_y + 3, (int)heatmap.size(2));
         m++) {
      for (int n = max(pos_x - 2, 0); n < min(pos_x + 3, (int)heatmap.size(3));
           n++) {
        ratio_width += rectmap[1].at<float>(m, n);
        ratio_height += rectmap[2].at<float>(m, n);
        pixelcount++;
      }
    }
    if (pixelcount > 0) {
      ratio_width = min(max(ratio_width / pixelcount, 0.f), 1.f);
      ratio_height = min(max(ratio_height / pixelcount, 0.f), 1.f);

      points = points * ratio_net_downsample / ratio_input_to_net;
      float rect_w = ratio_width * TRAIN_IMAGE_WIDTH / ratio_input_to_net;
      float rect_h = ratio_height * TRAIN_IMAGE_HEIGHT / ratio_input_to_net;
      Point2f l_t = points - Point2f(rect_w / 2.f, rect_h / 2.f);
      Point2f r_b = points + Point2f(rect_w / 2.f, rect_h / 2.f);
      l_t.x = max(l_t.x, 0.f);
      l_t.y = max(l_t.y, 0.f);
      r_b.x = min(r_b.x, inputImage.cols - 1.f);
      r_b.y = min(r_b.y, inputImage.rows - 1.f);
      handrect.push_back(Rect(l_t.x, l_t.y, r_b.x - l_t.x, r_b.y - l_t.y));
    }
  }
}

void detecthand(vector<map<float, Point2f, greater<float>>>& manypoints,
                jit::script::Module& model, const Mat& inputImage,
                const vector<Rect>& handrect) {
  // transform the input data and copy to the gpu
  auto inputTensor = torch::zeros(
      {(int)handrect.size(), 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH});
  vector<float> ratio_input_to_net((int)handrect.size());
  for (int i = 0; i < handrect.size(); i++) {
    ratio_input_to_net[i] =
        transformNetInput(inputTensor, inputImage(handrect[i]), i);
  }

  // run the network
  auto net_result =
      model.forward({inputTensor.cuda()}).toTuple()->elements()[3].toTensor();

  // determine the joint position
  float ratio_net_downsample = TRAIN_IMAGE_HEIGHT / float(net_result.size(2));
  size_t total_bytes = sizeof(float) * net_result.size(2) * net_result.size(3);
  for (int rectIdx = 0; rectIdx < handrect.size(); rectIdx++) {
    for (int i = 0; i < net_result.size(1) - 3; i++) {
      Mat heatmap =
          Mat::zeros(net_result.size(2), net_result.size(3), CV_32FC1);
      memcpy((float*)heatmap.data,
             net_result[rectIdx][i].cpu().data_ptr<float>(), total_bytes);
      map<float, Point2f, greater<float>> points;
      nmslocation(heatmap, points, LABEL_MIN);

      // convert to the original image
      int count = 0;
      for (auto iter = points.begin(); iter != points.end(); iter++, count++) {
        // we only detect less than 2 hands in current implementation
        if (count >= 2) break;
        Point2f points =
            iter->second * ratio_net_downsample / ratio_input_to_net[rectIdx] +
            Point2f(handrect[rectIdx].x, handrect[rectIdx].y);
        manypoints[i].insert(make_pair(iter->first, points));
      }
    }
  }
}

vector<map<float, Point2f, greater<float>>> pyramidinference(
    jit::script::Module& model, Mat& inputImage, vector<Rect>& handrect) {
  vector<map<float, Point2f, greater<float>>> many_keypoints(21);

  // first step to determine the rough hand position in the image if not input
  // the handrect
  if (handrect.size() == 0) {
    handrect.push_back(Rect(0, 0, inputImage.cols, inputImage.rows));
    detectBbox(handrect, model, inputImage);

    if (handrect.size() == 0) return many_keypoints;
  }

  // parameters for drawing
  //
  const auto thicknessCircleRatio = 1.f / 120.f;
  const auto thicknessCircle =
      max(int(sqrt(inputImage.cols * inputImage.rows) * thicknessCircleRatio +
              0.5f),
          2);
  int numberColors = HAND_COLORS_RENDER.size();

  // for each small image detect the joint points
  detecthand(many_keypoints, model, inputImage, handrect);

  // drawing
  for (int currjointIndex = 0; currjointIndex < 21; currjointIndex++) {
    const cv::Scalar curr_color{
        HAND_COLORS_RENDER[(currjointIndex * 3) % numberColors],
        HAND_COLORS_RENDER[(currjointIndex * 3 + 1) % numberColors],
        HAND_COLORS_RENDER[(currjointIndex * 3 + 2) % numberColors]};
    for (auto it = many_keypoints[currjointIndex].begin();
         it != many_keypoints[currjointIndex].end(); it++) {
      circle(inputImage, it->second, thicknessCircle, curr_color, -1);
    }
  }

  // rectangle
  for (int i = 0; i < handrect.size(); i++) {
    rectangle(inputImage, handrect[i], cv::Scalar(0, 255, 0),
              min(inputImage.cols / TRAIN_IMAGE_WIDTH * 3.f, 3.f));
  }

  return many_keypoints;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: DemoForHand.exe [name of trained pytorch model] [mp4 file]\n");
    return 1;
  }

  // load the model
  auto model = torch::jit::load(argv[1], torch::kCUDA);
  auto data_f = argv[2];

  // -------------------------------------------------
  //
  // real-time desktop demo
  //
  vector<map<float, Point2f, greater<float>>> many_keypoints;

  VideoCapture capture(data_f);
  capture.set(CAP_PROP_FRAME_WIDTH, 1280);
  capture.set(CAP_PROP_FRAME_HEIGHT, 960);

  double t = 0, fps = 0;
  char buffer[512];

  while (1) {
    Mat frame;

    capture >> frame;

    if (frame.empty()) {
      printf("can not open the camera!\n");
      return 1;
    }

    // Load the image data
    Mat source = frame;
    t = (double)cv::getTickCount();
    vector<Rect> handrect;
    many_keypoints = pyramidinference(model, frame, handrect);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); 
    cv::putText(frame, to_string(1/t), cv::Point(frame.cols - 100, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    imshow("hand detection", frame);

    if (waitKey(1) == 27) {
      break;
    }
  }
  return 0;
}