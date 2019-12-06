#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "glog/logging.h"

using namespace std;
using namespace cv;
using namespace google;
 
int main(){
 
  VideoCapture cap(0); 
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
