#pragma once

#include "SobelDescriptor.h"
#include "Delaunay.h"
#include "Matrix.h"
#include "Timer.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

using namespace Eigen;
using namespace cv;
using namespace std;

namespace lsdm {


#define INVALID_DISP -1
#define INVALID_VAR  -1
#define FLOAT_EPSILON 1e-4  

#define imref_f32(img, y, x) (img.ptr<float>(y)[x])
#define imref_u8(img, y, x) (img.ptr<uchar>(y)[x])
#define min3(a,b,c) (a<b?(a<c?a:c):(b<c?b:c))
#define max3(a,b,c) (a>b?(a>c?a:c):(b>c?b:c))


class StereoMatcher {
  
public:
  
  enum setting {ROBOTICS,MIDDLEBURY};
  
  // parameter settings
  struct Parameters {
    int32_t disp_min;               // min disparity
    int32_t disp_max;               // max disparity
    float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
    float   depth_diff_threshold;   // the difference of nearby pivot's depth, too large means different object
    float   visual_range_threshold; // the difference of vertexs' image coordinate, too large means different object
    float   lidar_range_threshold;  // the difference of vertexs' polar coordinate, too large means different object
    int32_t support_texture;        // min texture for support points
    float   support_gradient;       // min gradient for support points
    int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
    int32_t incon_window_size;      // window size of inconsistent support point check
    int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
    int32_t incon_min_support;      // minimum number of consistent support points
    bool    add_corners;            // add support points at image corners with nearest neighbor disparities
    int32_t grid_size;              // size of neighborhood for additional support point extrapolation
    float   beta;                   // image likelihood parameter
    float   gamma;                  // prior constant
    float   sigma;                  // prior sigma
    float   sigma_lidar;            // prior sigma of lidar
    float   sigma_visual;           // prior sigma of stereo
    float   sradius;                // prior sigma radius
    int32_t match_texture;          // min texture for dense matching
    int32_t lr_threshold;           // disparity threshold for left/right consistency check
    float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
    int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
    int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
    bool    filter_median;          // optional median filter (approximated)
    bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
    bool    postprocess_only_left;  // saves time by not postprocessing the right image
    bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
                                    // note: for this option D1 and D2 must be passed with size
                                    //       width/2 x height/2 (rounded towards zero)
    
    // constructor
    Parameters (setting s=ROBOTICS) {
      
      // default settings in a robotics environment
      // (do not produce results in half-occluded areas
      //  and are a bit more robust towards lighting etc.)
      if (s==ROBOTICS) {
        disp_min               = 0;
        disp_max               = 100;
        support_threshold      = 0.85;
        depth_diff_threshold   = 3.0; 
        visual_range_threshold = 0.2; // scale  
        lidar_range_threshold  = 0.2; // rad
        support_texture        = 10;
        support_gradient       = 5;
        candidate_stepsize     = 4;
        incon_window_size      = 5;
        incon_threshold        = 5;
        incon_min_support      = 5;
        add_corners            = 0;
        grid_size              = 20;
        beta                   = 0.02;
        gamma                  = 3;
        sigma                  = 1;
        sigma_lidar            = 0.1;
        sigma_visual           = 5;
        sradius                = 2;
        match_texture          = 1;
        lr_threshold           = 2;
        speckle_sim_threshold  = 1;
        speckle_size           = 200;
        ipol_gap_width         = 3;
        filter_median          = 0;
        filter_adaptive_mean   = 1;
        postprocess_only_left  = 1;
        subsampling            = 0;
        
      // default settings for middlebury benchmark
      // (interpolate all missing disparities)
      } else {
        disp_min              = 2;
        disp_max              = 255;
        support_threshold     = 0.95;
        depth_diff_threshold  = 300.0;
        visual_range_threshold = 0.2;
        lidar_range_threshold  = 0.015;
        support_texture       = 10;
        support_gradient       = 5;
        candidate_stepsize    = 5;
        incon_window_size     = 5;
        incon_threshold       = 5;
        incon_min_support     = 5;
        add_corners           = 1;
        grid_size             = 20;
        beta                  = 0.02;
        gamma                 = 5;
        sigma                 = 1;
        sigma_lidar           = 0.01;
        sigma_visual          = 1;
        sradius               = 3;
        match_texture         = 0;
        lr_threshold          = 2;
        speckle_sim_threshold = 1;
        speckle_size          = 200;
        ipol_gap_width        = 5000;
        filter_median         = 1;
        filter_adaptive_mean  = 0;
        postprocess_only_left = 0;
        subsampling           = 0;
      }
    }
  };

  struct Beam {
    float x;
    float y;
    float z;
    float intensity;
    Beam(float x, float y, float z, float intensity) : x(x),y(y),z(z),intensity(intensity) {}
  };

  // use float because introduce lidar pointcloud, avoiding accuracy transforming loss
  struct Pivot {
    float u; // x in image
    float v; // y in image
    float d; // disparity between stereo
    Pivot(float u,float v,float d):u(u),v(v),d(d) {}
    bool operator==(const Pivot& other) const { return (other.u == u && other.v == v && other.d == d); }
    bool operator!=(const Pivot& other) const { return (other.u != u || other.v != v || other.d != d); }
  };

  struct Triangle {
    int32_t c1,c2,c3;    // index of vector<Pivot>
    float   t1a,t1b,t1c; // plane parameters in left image   au + bv + c = d
    float   t2a,t2b,t2c; // plane parameters in right image  au + bv + c = d
    Triangle(int32_t c1,int32_t c2,int32_t c3):c1(c1),c2(c2),c3(c3){}
  };

  enum SensorType
  {
    LIDAR = 0,
    STEREO = 1,
    MONOCULAR = 2,
    LIDAR_STEREO = 3,
    LIDAR_MONOCULAR = 4,
  };

  enum RefineMethod
  {
    MAX_POSTERIORI = 0,
    WEIGHT_GAUSSIAN = 1,
  };

  // constructor, input: parameters  
  StereoMatcher (Parameters param,
                 Eigen::Matrix3f &intrinsic, Eigen::Matrix4f &extrinsic, float baseline) 
  : param(param), intrinsic(intrinsic), extrinsic(extrinsic)
  {
    Eigen::Matrix<float, 3, 4> intrinsic34;
    intrinsic34.leftCols(3) = intrinsic;
    intrinsic34.rightCols(1) = Vector3f::Zero();
    projectMat = intrinsic34 * extrinsic;
    fb = baseline * intrinsic(0, 0);
  }

  // deconstructor
  ~StereoMatcher () 
  {
  }

  inline uint32_t getAddressOffsetImage (const int32_t& u,const int32_t& v,const int32_t& width) {
    return v*width+u;
  }

  inline uint32_t getAddressOffsetGrid (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
    return (y*width+x)*disp_num+d;
  }

  inline float calcGradient(const int u, const int v, const cv::Mat& gray) {
    if(u + 1 < 0 || u + 1 >= gray.cols || v + 1 < 0 || v + 1 >= gray.rows) {
      return 0;
    }
    float dx = 0.5f * (imref_u8(gray, v, u+1) - imref_u8(gray, v, u-1));
    float dy = 0.5f * (imref_u8(gray, v+1, u) - imref_u8(gray, v-1, u));
    return sqrtf(dx*dx+dy*dy);
  }

  // extension of original matching function
  // inputs: pointers to left (I1) and right (I2) intensity image (uint8, input)
  //         pointers to left (D1) and right (D2) disparity image (float, output)
  //         dims[0] = width of I1 and I2
  //         dims[1] = height of I1 and I2
  //         dims[2] = bytes per line (often equal to width, but allowed to differ)
  //         note: D1 and D2 must be allocated before (bytes per line = width)
  //               if subsampling is not active their size is width x height,
  //               otherwise width/2 x height/2 (rounded towards zero)
  void process2(const cv::Mat& img1, 
                const cv::Mat& img2, 
                const std::vector<Eigen::Vector3f>& scans,
                cv::Mat& disp1, 
                cv::Mat& disp2);

  void buildVisualDisparity(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& idepth,
                                         float depthMin, float depthMax, float gridSize);

  void buildSparseStereo(const cv::Mat& gray, SobelDescriptor& desc1, SobelDescriptor& desc2, const int disp_min, const int disp_max, int grid_size);

  void removeRedundantSupportPoints(cv::Mat& disp, int32_t redun_max_dist, int32_t redun_threshold, bool vertical);

  void addCornerSupportPoints(vector<Pivot> &pivots);

  void calcDelaunay(const vector<Pivot>& pivots, vector<Triangle>& triangles, int32_t is_right);

  bool checkDelaunay(const std::vector<Pivot>& pivots, const Triangle& triangle);

  void calcPlanes(std::vector<Pivot> pivots, std::vector<Triangle> &tri,int32_t is_right);

  void updatePrior(cv::Mat& disp, cv::Mat& var, const std::vector<Pivot>& pivots, const std::vector<Triangle>& triangles, SensorType type);

  inline void refineDisparity(int32_t u, int32_t v, const SobelDescriptor& desc1, const SobelDescriptor& desc2,
                        float disp_in, float var_in, float& disp_out, float& var_out);

  void leftRightConsistencyCheck(cv::Mat& disp1, cv::Mat& disp2, cv::Mat& var1, cv::Mat& var2); 

  void updatePosteriori(cv::Mat& disp, cv::Mat& var, SobelDescriptor& desc1, SobelDescriptor& desc2, SensorType type);

  inline float findMatch(const int u, const int v,
                         const int width, const int height,
                         const int disp_min, const int disp_max,
                         const SobelDescriptor& desc1, const SobelDescriptor& desc2,
                         bool is_right);
                  
  inline void removeInconsistentSupportPoints (cv::Mat& disp, int incon_window_size = 5, int incon_threshold = 5, int incon_min_support = 5);

  void createGrid(vector<Pivot> pivots, int32_t* disparity_grid, int32_t* grid_dims,bool is_right) ;

  inline void updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
		const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);

  inline void updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,
		const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);

  inline void findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
		int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
		int32_t *P,int32_t &plane_radius,bool &valid,bool &is_right,float* D);

  void computeDisparity(vector<Pivot> pivots,vector<Triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
		uint8_t* I1_desc,uint8_t* I2_desc,bool is_right,float* D);

  // postprocessing
  void removeSmallSegments (float* D);
  void gapInterpolation (float* D);

  // optional postprocessing
  void adaptiveMean (float* D);
  void median (float* D);


  inline float fitQuadraticFuncion(Vector2f& p1, Vector2f& p2, Vector2f& p3);

  inline std::vector<cv::Point2i> selectRegion(const std::vector<Pivot>& pivots, const Triangle& triangle);

  inline bool detectRegion(const cv::Point2f& point, const cv::Point2f& pa, const cv::Point2f& pb, const cv::Point2f& pc);

  inline cv::Scalar dispshader(float disparity); 

  inline cv::Scalar covshader(float var);

  inline float lidar_sigma(float d, float sigma0, float fb) {
    return fb / d * sigma0;
  }

  inline float gaussianPdf(float input, float mean, float sigma2) {
    return std::exp(-0.5*(input-mean)*(input-mean)/sigma2);
  }

  inline float exponentialPdf(float input, float lamda) {
    return std::exp(- lamda * input);
  }

  inline float gaussian_uniform_pdf(float input, float gamma, float mean, float sigma2) {
    return gamma + std::exp(-0.5*(input-mean)*(input-mean)/sigma2);
  }

  inline float exponential_val(float beta, const __m128i &xmm1, const __m128i &xmm2) {
    __m128i sad = _mm_sad_epu8(xmm1,xmm2);
    return beta * (float)(_mm_extract_epi16(sad,0)+_mm_extract_epi16(sad,4));
  }

  inline float gaussian_val(float input, float mean, float sigma2) {
    return 0.5*(input-mean)*(input-mean)/sigma2;
  }

  // visualization functions
  void drawDisparityAndvariance(const cv::Mat& gray, const cv::Mat& disparity, const cv::Mat& variance);
  void drawLidarProjections(const cv::Mat& img, const std::vector<StereoMatcher::Pivot>& pivots);
  void drawDelaunay(const cv::Mat& img, const std::vector<Pivot>& pivots, const std::vector<Triangle>& triangles);
  void drawMatching(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Pivot>& pivots);
  void drawPivots(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Pivot>& pivots);
    
public:

  // parameter set
  Parameters param;
  SensorType sensorType;

  // memory aligned input images + dimensions
  uint8_t *I1,*I2;
  int width, height, bpl;

  cv::Mat visualDisp, lidarDisp, visualVar, lidarVar;

  // sensor extrinsics and intrinsics
  Eigen::Matrix3f intrinsic;
  Eigen::Matrix4f extrinsic;
  Eigen::Matrix<float, 3, 4> projectMat;
  double fb; 

  vector<Pivot> visual_pivots;
  vector<Triangle> visual_triangles1, visual_triangles2;

  Timer timer;
};

}
