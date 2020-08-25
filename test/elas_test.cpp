/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libStereoMatcher.
Authors: Andreas Geiger

libStereoMatcher is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libStereoMatcher is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libStereoMatcher; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

// Demo program showing how libStereoMatcher can be used, try "./StereoMatcher -h" for help


#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include <list>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <mutex>
#include <thread>
#include <algorithm>


#include "StereoMatcher.h"
#include "Matrix.h"
#include "boost/program_options.hpp"
#include "boost/timer/timer.hpp"



using namespace Eigen; 
using namespace std;
using namespace boost::program_options;

struct Options{
  string img_left;
  string img_right;
  string pointcloud;
  Matrix<double, 3, 4> P0, P1, P2, P3;
  Matrix3d R0;
  Matrix4d Tcl;
};

struct Beam{
  float x, y, z, intensity;
  Beam(float x, float y, float z, float intensity) : x(x),y(y),z(z),intensity(intensity) {}
};

bool readParameters(const string& setting_file, struct Options& options)
{
  cv::String addr = "/home/dst03/code/stereo/config/fusion.yaml";
  cv::FileStorage settings(addr, cv::FileStorage::READ);
  if(!settings.isOpened()) {
    return 1;
  }
  // settings["image_right"] >> options.img_right;
  // settings["P0"] >> options.P0;
  return 0;
}

bool readKittiPointCloud(const string& filename, std::vector<Eigen::Vector3f>& scans)
{
  scans.clear();
  scans.reserve(200000);

  int num_pts = 0;
  std::fstream pcfile(filename.c_str(), std::ios::in | std::ios::binary);
  if(!pcfile.good()){
		std::cerr << "Could not read file: " << filename << std::endl;
		return false;
	}
	pcfile.seekg(0, std::ios::beg);

  double horizontal_rad, pre_horizontal_rad;
  int line_index = 0, stamp = 0;
  bool first_in = true;
	for (num_pts = 0; pcfile.good() && !pcfile.eof(); num_pts++) {
    Eigen::Vector3f beam(0, 0, 0);
    float intensity;
		pcfile.read((char*)&beam(0), sizeof(float));
    pcfile.read((char*)&beam(1), sizeof(float));
    pcfile.read((char*)&beam(2), sizeof(float));
		pcfile.read((char*)&intensity, sizeof(float));
    scans.emplace_back(beam);
	}
  cout << "read point cloud: " << num_pts << endl;
	pcfile.close();
  scans.resize(num_pts);
  return true;
}

bool readKittiPointCloud(const string& filename, std::vector<std::vector<Eigen::Vector3f>>& scans)
{
  const float horizontal_delta_rad  = M_PI * 2 / 2000.0f;

  scans.clear();
  Eigen::Vector3f beam_init(0, 0, 0);
  scans = std::vector< std::vector<Eigen::Vector3f> >(130);
  for(int i = 0; i < 130; ++i) {
    scans.at(i) = std::vector<Eigen::Vector3f>(2000 + 1, beam_init);
  }

  int num_pts = 0;
  std::fstream pcfile(filename.c_str(), std::ios::in | std::ios::binary);
  if(!pcfile.good()){
		std::cerr << "Could not read file: " << filename << std::endl;
		return false;
	}
	pcfile.seekg(0, std::ios::beg);

  double horizontal_rad, pre_horizontal_rad;
  int line_index = 0, stamp = 0;
  bool first_in = true;
	for (num_pts=0; pcfile.good() && !pcfile.eof(); num_pts++) {
    Eigen::Vector3f beam(0, 0, 0);
    float intensity;

		pcfile.read((char*)&beam(0), sizeof(float));
    pcfile.read((char*)&beam(1), sizeof(float));
    pcfile.read((char*)&beam(2), sizeof(float));
		pcfile.read((char*)&intensity, sizeof(float));
    pre_horizontal_rad = horizontal_rad;
		horizontal_rad = std::atan2(beam(1), beam(0));
    
    if(first_in) {
      pre_horizontal_rad = horizontal_rad;
      first_in = false;
    }
    if((pre_horizontal_rad < 0 && horizontal_rad >= 0)) {
      line_index++;
    }
    if(horizontal_rad < 0) {
      stamp = std::round((horizontal_rad + 2 * M_PI)/ horizontal_delta_rad);
    }
    else {
      stamp = std::round(horizontal_rad / horizontal_delta_rad);
    }
    if(stamp > 2000) continue;
    scans.at(line_index).at(stamp) = beam;
    // std::cout << "line: " << line_index << " stamp: " << stamp << " beam: " << beam.x << " " << beam.y << " " << beam.z << " angle: " << horizontal_rad << endl;
	}
  cout << "read point cloud: " << num_pts << endl;
	pcfile.close();
  scans.resize(line_index);
  return true;
}

void loadImages(const std::string &sequense_path, std::vector<std::string> &left_path,
                std::vector<std::string> &right_path, std::vector<double> &timestamps)
{
  std::ifstream file_times;
  std::string time_path = sequense_path + "/times.txt";
  file_times.open(time_path.c_str());
  while(!file_times.eof())
  {
    std::string s;
    std::getline(file_times,s);
    if(!s.empty())
    {
      std::stringstream ss;
      ss << s;
      double t;
      ss >> t;
      timestamps.push_back(t);
    }
  }

  std::string left_prefix = sequense_path + "/image_2/";
  std::string right_prefix = sequense_path + "/image_3/";

  const int nTimes = timestamps.size();
  left_path.resize(nTimes);
  right_path.resize(nTimes);

  for(int i=0; i<nTimes; i++)
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    left_path[i] = left_prefix + ss.str() + ".png";
    right_path[i] = right_prefix + ss.str() + ".png";
  }
}

// compute disparities of pgm image input pair image_left, image_right
void process (const char* image_left, const char* image_right) {

  cout << "Processing: " << image_left << ", " << image_right << endl;

  cv::Mat img1, img2;
  img1 = cv::imread(image_left, cv::IMREAD_GRAYSCALE);
  img2 = cv::imread(image_right, cv::IMREAD_GRAYSCALE);


  // get image width and height
  int32_t width  = img1.rows;
  int32_t height = img1.cols;

  // allocate memory for disparity images
  const int32_t dims[3] = {width,height,width}; // bytes per line = width

  cv::Mat disp1, disp2;

  // process
  lsdm::StereoMatcher::Parameters param;
  param.postprocess_only_left = false;
  // StereoMatcher StereoMatcher(param);
  // StereoMatcher.process(I1->data,I2->data,D1_data,D2_data,dims);
  // StereoMatcher.process2(img1, img2, disp1, disp2);

  

}

// compute disparities of pgm image input pair image_left, image_right
void process (Matrix3f& intrinsic, Matrix4f& extrinsic, float baseline) {

  std::string sourcePath = "/home/dst03/dataset/KITTI/rgb00";
	std::vector<std::string> leftPath, rightPath;
	std::vector<double> timeStamp;

	loadImages(sourcePath, leftPath, rightPath, timeStamp);

  // load images
  for(int i = 0; i < leftPath.size(); ++i) {
    cv::Mat img1, img2;

    img1 = cv::imread(leftPath[i], cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(rightPath[i], cv::IMREAD_GRAYSCALE);

    std::vector<Eigen::Vector3f> scans;

    // if(!readKittiPointCloud(pointcloud, scans)) {
    //   return;
    // }

    // get image width and height
    int32_t width  = img1.cols;
    int32_t height = img1.rows;

    // allocate memory for disparity images
    const int32_t dims[3] = {width,height,width}; // bytes per line = width

    cv::Mat disp(height, width, CV_32FC1, cv::Scalar(-1));

    // process
    lsdm::StereoMatcher::Parameters param(lsdm::StereoMatcher::ROBOTICS);
    param.postprocess_only_left = false;
    Matrix<double, 3, 4> project_mat1, project_mat2;
    lsdm::StereoMatcher StereoMatcher(param, intrinsic, extrinsic, baseline);

    StereoMatcher.buildVisualDisparity(img1, img2, disp, 3, 100, 6);
    // cv::Mat imgsub1, imgsub2;
    // pyrDown(img1, imgsub1, cv::Size(floor(img1.cols / 2), floor(img1.rows / 2)));
    // pyrDown(img2, imgsub2, cv::Size(floor(img2.cols / 2), floor(img2.rows / 2)));
    // StereoMatcher.process(I1->data,I2->data,D1_data,D2_data,dims);
    // StereoMatcher.process2(imgsub1, imgsub2, scans, disp1, disp2);
  }

  

}

int main (int argc, char** argv) {

  // {
  //   double t1 = cv::getTickCount();
  //   MatrixXd A1, A2, A3;
  //   for(int i = 0; i < 1000; ++i) {
  //     A1.resize(10, 10);
  //     A1.setRandom();
  //     A2.resize(10, 10);
  //     A2.setRandom();
  //     A3 = A1 * A2;
  //   }
  //   double t2 = cv::getTickCount();
  //   cout << "double matrix time cost: " << (t2 - t1) / cv::getTickFrequency() << endl;    
  // }

  // {
  //   double t3 = cv::getTickCount();
  //   MatrixXf A1, A2, A3;
  //   for(int i = 0; i < 1000; ++i) {
  //     A1.resize(10, 10);
  //     A1.setRandom();
  //     A2.resize(10, 10);
  //     A2.setRandom();
  //     A3 = A1 * A2;
  //   }
  //   double t4 = cv::getTickCount();
  //   cout << "float matrix time cost: " << (t4 - t3) / cv::getTickFrequency() << endl;
  // }

   
  Matrix4d R0, Tcl;
  R0 << 9.999128000000e-01, 1.009263000000e-02, -8.511932000000e-03, 0.000000000000e+00, 
        -1.012729000000e-02, 9.999406000000e-01, -4.037671000000e-03, 0.000000000000e+00,
         8.470675000000e-03, 4.123522000000e-03, 9.999556000000e-01, 0.000000000000e+00,
         0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;
  // Tcl << 6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02, 
  //        -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02, 
  //         9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01,
  //         0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;

  Tcl <<  4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, 
          -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 
          9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
          0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;

  Matrix<double, 3, 4> P0, P1, P2, P3; 
  P0 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
         0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 
         0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;
         
  P1 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02, 
         0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 
         0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;

  P2 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 4.538225000000e+01, 
         0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, -1.130887000000e-01, 
         0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 3.779761000000e-03;

  P3 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.372877000000e+02, 
         0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 2.369057000000e+00, 
         0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.915215000000e-03;
          
  
  MatrixXd Pl, Pr;
  Pl = P2 * Tcl;
  Pr = P3 * Tcl;
  float baseline = (P2.block(0, 3, 3, 1) - P3.block(0, 3, 3, 1)).norm()/P2(0, 0);
  Matrix3f K = P2.block(0, 0, 3, 3).cast<float>();
  Matrix4f T = Tcl.cast<float>();
  process(K, T, baseline);
  cout << "... done!" << endl;

  // // run demo
  // if (argc==2 && !strcmp(argv[1],"demo")) {
  //   process("img/cones_left.pgm",   "img/cones_right.pgm");
  //   process("img/aloe_left.pgm",    "img/aloe_right.pgm");
  //   process("img/raindeer_left.pgm","img/raindeer_right.pgm");
  //   process("img/urban1_left.pgm",  "img/urban1_right.pgm");
  //   process("img/urban2_left.pgm",  "img/urban2_right.pgm");
  //   process("img/urban3_left.pgm",  "img/urban3_right.pgm");
  //   process("img/urban4_left.pgm",  "img/urban4_right.pgm");
  //   cout << "... done!" << endl;

  // // compute disparity from input pair
  // } 
  // else if(argc == 3) {
  //   process(argv[1],argv[2]);
  //   cout << "... done!" << endl;

  // // display help
  // } 
  // else if(argc == 4) {
  //   process(argv[1],argv[2], argv[3]);
  //   cout << "... done!" << endl;
  // }
  // else {
  //   cout << endl;
  //   cout << "StereoMatcher demo program usage: " << endl;
  //   cout << "./StereoMatcher demo ................ process all test images (image dir)" << endl;
  //   cout << "./StereoMatcher left.pgm right.pgm .. process a single stereo pair" << endl;
  //   cout << "./StereoMatcher -h .................. shows this help" << endl;
  //   cout << endl;
  //   cout << "Note: All images must be pgm greylevel images. All output" << endl;
  //   cout << "      disparities will be scaled such that disp_max = 255." << endl;
  //   cout << endl;
  // }

  return 0;
}


