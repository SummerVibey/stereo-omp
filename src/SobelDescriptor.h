#pragma once

#include <opencv2/opencv.hpp>

namespace lsdm {

#define DISP_SPACE_SIZE 16

class SobelDescriptor {

 public:
  
  // constructor creates filters
  SobelDescriptor(const cv::Mat& img);

  // deconstructor releases memory
  ~SobelDescriptor() 
  {
    descriptor.release();
  }

  void release();
  
  // descriptors accessible from outside
  cv::Mat descriptor;

private:
  void createDescriptor(cv::Mat& dx, cv::Mat& dy);

};

}

