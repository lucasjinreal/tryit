#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

at::Tensor toTensor(cv::Mat image) {
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  at::Tensor tensor_image =
      torch::from_blob(image.data, {image.rows, image.cols, 3}, at::kByte);
  tensor_image = tensor_image.unsqueeze(0);
  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image = tensor_image.to(at::kFloat);
  tensor_image /= 255;

  return tensor_image;
}

cv::Mat toImage(at::Tensor& tensor, cv::Size sizes) {
  cv::Mat image;
  image = cv::Mat(sizes, CV_8UC3, tensor.data_ptr());
  return image;
}

void color(cv::Mat& image, cv::Mat prediction_vectors) {
  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {
      if (prediction_vectors.at<uchar>(y, x) == 1) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
      } else if (prediction_vectors.at<uchar>(y, x) == 2) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
      } else if (prediction_vectors.at<uchar>(y, x) == 3) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
      } else if (prediction_vectors.at<uchar>(y, x) == 4) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 0);
      }
    }
  }
}

int main(int argc, char** argv) {
  torch::Tensor tensor = torch::rand({2, 3});

  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(
      "/media/jintian/data/ai/innovations/lane_yolov3_det/vendor/"
      "LD_CNN_classic/Lane_CNN/res/model_cpp.cnn");

  std::vector<torch::jit::IValue> inputs;
  cv::Mat in_img = cv::imread(argv[1]);
  // The network is trained with 512x256 images, better use this size
  cv::Mat image;
  cv::resize(in_img, image, cv::Size(512, 256));
  cv::imshow("image", image);

  at::Tensor image_tensor = toTensor(image);

  // Push back the input image
  inputs.push_back(image_tensor);

  // Inference phase
  at::Tensor output = module->forward(inputs).toTensor();

  std::tuple<at::Tensor, at::Tensor> output_max = at::max(output, 1);
  at::Tensor argmax = std::get<1>(output_max);
  argmax = argmax.to(at::kByte);

  // Ouptut
  cv::Mat prediction_vectors = toImage(argmax, image.size());
  cout << prediction_vectors << endl;
  color(image, prediction_vectors);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  cv::imshow("prediction", image);
  cv::waitKey(0);
  std::cout << tensor << std::endl;
}