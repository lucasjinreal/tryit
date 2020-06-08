#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "glog/logging.h"

#include "thor/os.h"



using namespace std;
using namespace cv;
using namespace google;
 
int main(int argc, char** argv){
  
  // vector<string> extensions = {".jpg", ".png", ".jpeg"};
  // vector<cv::String> all_image_files;
  // for (auto e: extensions) {
  //   vector<cv::String> tmp;
  //   string image_file_ptn = thor::os::join(argv[1], "*" + e);
  //   cout << image_file_ptn << endl;
  //   glob(image_file_ptn, tmp);
  //   all_image_files.insert(all_image_files.end(), tmp.begin(), tmp.end());
  // }
  // for (auto a: all_image_files) {
  //   cout << a << endl;
  // }


  string data_f = argv[1];
  VideoCapture cap(data_f); 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_POS_FRAMES, 60);
  while(1){

    Mat frame;
    cap >> frame;
  
    if (frame.empty())
      break;
 
    imshow( "Frame", frame );
 
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  
  cap.release();
  destroyAllWindows();

     
  return 0;
}
