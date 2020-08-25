#include "SobelDescriptor.h"

namespace lsdm {



SobelDescriptor::SobelDescriptor(const cv::Mat& img)
{
  std::vector<int> size = {DISP_SPACE_SIZE, img.rows, img.cols};

  cv::Mat dx, dy;
  descriptor = cv::Mat(img.rows, img.cols, CV_8UC(DISP_SPACE_SIZE), cv::Scalar(0));;

  cv::Sobel(img, dx, CV_16SC1, 1, 0, 3);
  cv::Sobel(img, dy, CV_16SC1, 0, 1, 3);

  dx.convertTo(dx, CV_8U, 1, 128);
  dy.convertTo(dy, CV_8U, 1, 128);
  // cv::equalizeHist(dx, dx);
  // cv::equalizeHist(dy, dy);
  createDescriptor(dx, dy);
  dx.release();
  dy.release();
}

void SobelDescriptor::release()
{
  descriptor.release();
}

void SobelDescriptor::createDescriptor(cv::Mat& dx, cv::Mat& dy)
{
  assert(dx.size() == dy.size());
  int width = dx.cols;
  int height = dx.rows;
  // create filter strip
  for (int v = 3; v < height - 3; v++) {
    for (int u = 3; u < width - 3; u++) {
      uchar* descriptor_ptr = descriptor.ptr<uchar>(v, u);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v - 2, u    );
      *(descriptor_ptr++) = *dx.ptr<uchar>(v - 1, u - 2);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v - 1, u    );
      *(descriptor_ptr++) = *dx.ptr<uchar>(v - 1, u + 2);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v    , u - 1);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v    , u    );
      *(descriptor_ptr++) = *dx.ptr<uchar>(v    , u    );
      *(descriptor_ptr++) = *dx.ptr<uchar>(v    , u + 1);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v + 1, u - 2);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v + 1, u    );
      *(descriptor_ptr++) = *dx.ptr<uchar>(v + 1, u + 2);
      *(descriptor_ptr++) = *dx.ptr<uchar>(v + 2, u    );
      *(descriptor_ptr++) = *dy.ptr<uchar>(v - 1, u    );
      *(descriptor_ptr++) = *dy.ptr<uchar>(v    , u - 1);
      *(descriptor_ptr++) = *dy.ptr<uchar>(v    , u + 1);
      *(descriptor_ptr++) = *dy.ptr<uchar>(v + 1, u    );
      // descriptor.ptr<uchar>(v, u)[ 0] = *dx.ptr<uchar>(v - 2, u    );
      // descriptor.ptr<uchar>(v, u)[ 1] = *dx.ptr<uchar>(v - 1, u - 2);
      // descriptor.ptr<uchar>(v, u)[ 2] = *dx.ptr<uchar>(v - 1, u    );
      // descriptor.ptr<uchar>(v, u)[ 3] = *dx.ptr<uchar>(v - 1, u + 2);
      // descriptor.ptr<uchar>(v, u)[ 4] = *dx.ptr<uchar>(v    , u - 1);
      // descriptor.ptr<uchar>(v, u)[ 5] = *dx.ptr<uchar>(v    , u    );
      // descriptor.ptr<uchar>(v, u)[ 6] = *dx.ptr<uchar>(v    , u    );
      // descriptor.ptr<uchar>(v, u)[ 7] = *dx.ptr<uchar>(v    , u + 1);
      // descriptor.ptr<uchar>(v, u)[ 8] = *dx.ptr<uchar>(v + 1, u - 2);
      // descriptor.ptr<uchar>(v, u)[ 9] = *dx.ptr<uchar>(v + 1, u    );
      // descriptor.ptr<uchar>(v, u)[10] = *dx.ptr<uchar>(v + 1, u + 2);
      // descriptor.ptr<uchar>(v, u)[11] = *dx.ptr<uchar>(v + 2, u    );
      // descriptor.ptr<uchar>(v, u)[12] = *dy.ptr<uchar>(v - 1, u    );
      // descriptor.ptr<uchar>(v, u)[13] = *dy.ptr<uchar>(v    , u - 1);
      // descriptor.ptr<uchar>(v, u)[14] = *dy.ptr<uchar>(v    , u + 1);
      // descriptor.ptr<uchar>(v, u)[15] = *dy.ptr<uchar>(v + 1, u    );
      // du: 0 0 x 0 0
      //     x 0 x 0 x
      //     0 x t x 0
      //     x 0 x 0 x
      //     0 0 x 0 0
      // t = 2*x

      // dv: 0 x 0
      //     x 0 x
      //     0 x 0
    }
  }
}

// kernel   t11 t12 t13 t14 t15
//          t21 t22 t23 t24 t25
//          t31 t32 t33 t34 t35
//          t41 t42 t43 t44 t45
//          t51 t52 t53 t54 t55

}