#include "StereoMatcher.h"

namespace lsdm {

void StereoMatcher::buildVisualDisparity(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& idepth,
                                         float depthMin, float depthMax, float gridSize)
{
  assert(idepth.size() == img1.size() && img1.size() == img2.size() && idepth.type() == CV_32F);

  param.disp_min = fb / depthMax;
  param.disp_max = fb / depthMin;
  height = img1.rows;
  width = img1.cols;

  cv::Mat disp1(img1.size(), CV_32FC1, cv::Scalar(-1)), disp2(img2.size(), CV_32FC1, cv::Scalar(-1)),
          var1(img1.size(), CV_32FC1, cv::Scalar(-1)), var2(img2.size(), CV_32FC1, cv::Scalar(-1));

  timer.start("Descriptor");
  SobelDescriptor desc1(img1);
  SobelDescriptor desc2(img2);

  // allocate memory for disparity grid
	int32_t grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
	int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
	int32_t grid_dims[3] = {param.disp_max+2,grid_width,grid_height};
	int32_t* disparity_grid_1 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));
	int32_t* disparity_grid_2 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));

  timer.start("Extract Sparse Pivots");
  buildSparseStereo(img1, desc1, desc2, param.disp_min, param.disp_max, gridSize);
  // timer.start("Draw Sparse Pivots");
  // drawPivots(img1, img2, visual_pivots);

  timer.start("Build Delaunay and Planes");
#pragma omp parallel num_threads(2)
	{
#pragma omp sections
		{
#pragma omp section
			{
				calcDelaunay(visual_pivots, visual_triangles1, 0);
        calcPlanes(visual_pivots, visual_triangles1, 0);
				createGrid(visual_pivots,disparity_grid_1,grid_dims,0);
				//computeDisparity(pivots,tri_1,disparity_grid_1,grid_dims,desc1.I_desc,desc2.I_desc,0,D1);
			}
#pragma omp section
			{
				calcDelaunay(visual_pivots, visual_triangles2, 0);
				calcPlanes(visual_pivots, visual_triangles2, 0);
				createGrid(visual_pivots,disparity_grid_2,grid_dims,1);
				//computeDisparity(pivots,tri_2,disparity_grid_2,grid_dims,desc1.I_desc,desc2.I_desc,1,D2);
			}

		}
	}
  // timer.start("Draw Visual Triangles");
  // drawDelaunay(img1, visual_pivots, visual_triangles1);

	timer.start("Matching");
#pragma omp sections
	{
	#pragma omp section
		updatePrior(disp1, var1, visual_pivots, visual_triangles1, STEREO);
    updatePosteriori(disp1, var1, desc1, desc2, STEREO);
	#pragma omp section
		updatePrior(disp2, var2, visual_pivots, visual_triangles1, STEREO);
    updatePosteriori(disp2, var2, desc1, desc2, STEREO);
	}

//   timer.start("Matching");
// #pragma omp sections
// 	{
// 	#pragma omp section
// 		computeDisparity(visual_pivots,visual_triangles1,disparity_grid_1,grid_dims,desc1.descriptor.data,desc2.descriptor.data,0,(float*)disp1.data);
// 	#pragma omp section
// 		computeDisparity(visual_pivots,visual_triangles2,disparity_grid_2,grid_dims,desc1.descriptor.data,desc2.descriptor.data,1,(float*)disp2.data);
// 	}


  timer.start("L/R Consistency Check");
  leftRightConsistencyCheck(disp1, disp2, var1, var2);


  // timer.start("Remove Small Segments");
  // removeSmallSegments((float*)disp1.data);

  timer.start("Gap Interpolation");
  gapInterpolation((float*)disp1.data);

  // timer.start("Adaptive Mean");
  // adaptiveMean((float*)disp1.data);

  // timer.start("Median");
  // median((float*)disp1.data);


  timer.plot();
  drawDisparityAndvariance(img2, disp2, var2);
}

// void StereoMatcher::setImage(const cv::Mat& img1, const cv::Mat& img2)
// {
//   assert(img1.size() == img2.size() && img1.type() == CV_8U && img2.type() == CV_8U);

//   image1 = img1.clone();
//   image2 = img2.clone();
//   this->width = img1.cols;
//   this->height = img1.rows;

//   {
//     double t1 = cv::getTickCount();
//     desc1.reset(img1);
//     desc2.reset(img2);
//     double t2 = cv::getTickCount();


void StereoMatcher::buildSparseStereo(const cv::Mat& gray, SobelDescriptor& desc1, SobelDescriptor& desc2, const int disp_min, const int disp_max, int grid_size)
{
  int disp_can_width  = 0;
  int disp_can_height = 0;
  for (int u=0; u<width;  u+=grid_size) disp_can_width++;
  for (int v=0; v<height; v+=grid_size) disp_can_height++;

  cv::Mat disp_img(disp_can_height, disp_can_width, CV_32FC1, cv::Scalar(INVALID_DISP));
  visual_pivots.clear();

  // loop variables
  int u,v;
  int16_t d,d2;
   
  // for all point candidates in image 1 do
  for (int u_can=1; u_can<disp_can_width; u_can++) {
    u = u_can*grid_size;
    for (int v_can=1; v_can<disp_can_height; v_can++) {
      v = v_can*grid_size;

      if(calcGradient(u, v, gray) < param.support_gradient) continue;

      // find forwards
      // using left and right consistency check
      d = findMatch(u, v, width, height, param.disp_min, param.disp_max, desc1, desc2, false);
      if (d>=param.disp_min) {
        // find backwards
        d2 = findMatch(u-d, v, width, height, param.disp_min, param.disp_max, desc1, desc2, true);
        // check l2r and r2l 
        if (d2 >= param.disp_min && abs(d-d2) <= param.lr_threshold)
          imref_f32(disp_img, v_can, u_can) = d;
      }
    }
  }

  // remove inconsistent support points
  removeInconsistentSupportPoints(disp_img);

  removeRedundantSupportPoints(disp_img,5,1,true);
	removeRedundantSupportPoints(disp_img,5,1,false);
  
  // move support points from image representation into a vector representation
  int valid_count = 0;
  visual_pivots.reserve((height/grid_size - 1) * (width/grid_size - 1));
  for (int u_can=1; u_can<disp_can_width; u_can++) {
    for (int v_can=1; v_can<disp_can_height; v_can++) {
      if (imref_f32(disp_img, v_can, u_can) >= param.disp_min) {
        valid_count++;
        visual_pivots.emplace_back(Pivot(u_can*grid_size,
                                      v_can*grid_size,
                                      imref_f32(disp_img, v_can, u_can)));

      }
    }
  }
  addCornerSupportPoints(visual_pivots);
  Pivot temp(0, 0, 0);
  visual_pivots.resize(valid_count, temp);

  
}

void StereoMatcher::removeRedundantSupportPoints(cv::Mat& disp, int32_t redun_max_dist, int32_t redun_threshold, bool vertical) 
{

  int disp_width = disp.cols;
  int disp_height = disp.rows;

	// parameters
	int32_t redun_dir_u[2] = {0,0};
	int32_t redun_dir_v[2] = {0,0};
	if (vertical) {
		redun_dir_v[0] = -1;
		redun_dir_v[1] = +1;
	} else {
		redun_dir_u[0] = -1;
		redun_dir_u[1] = +1;
	}

	// for all valid support points do
	#pragma omp for
	for (int32_t u_can=0; u_can<disp_width; u_can++) {
		for (int32_t v_can=0; v_can<disp_height; v_can++) {
			int16_t d_can = imref_f32(disp, v_can, u_can);
			if (d_can>=0) {

				// check all directions for redundancy
				bool redundant = true;
				for (int32_t i=0; i<2; i++) {

					// search for support
					int32_t u_can_2 = u_can;
					int32_t v_can_2 = v_can;
					int16_t d_can_2;
					bool support = false;
					for (int32_t j=0; j<redun_max_dist; j++) {
						u_can_2 += redun_dir_u[i];
						v_can_2 += redun_dir_v[i];
						if (u_can_2<0 || v_can_2<0 || u_can_2>=disp_width || v_can_2>=disp_height)
							break;
						d_can_2 = imref_f32(disp, v_can_2, u_can_2);
						if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
							support = true;
							break;
						}
					}

					// if we have no support => point is not redundant
					if (!support) {
						redundant = false;
						break;
					}
				}

				// invalidate support point if it is redundant
				if (redundant)
					imref_f32(disp, v_can, u_can) = -1;
			}
		}
	}
}

void StereoMatcher::addCornerSupportPoints(vector<Pivot> &pivots) 
{

	// list of border points
	vector<Pivot> p_border;
	p_border.push_back(Pivot(0,0,0));
	p_border.push_back(Pivot(0,height-1,0));
	p_border.push_back(Pivot(width-1,0,0));
	p_border.push_back(Pivot(width-1,height-1,0));

	// find closest d
	for (int32_t i=0; i<p_border.size(); i++) {
		int32_t best_dist = 10000000;
		for (int32_t j=0; j<pivots.size(); j++) {
			int32_t du = p_border[i].u-pivots[j].u;
			int32_t dv = p_border[i].v-pivots[j].v;
			int32_t curr_dist = du*du+dv*dv;
			if (curr_dist<best_dist) {
				best_dist = curr_dist;
				p_border[i].d = pivots[j].d;
			}
		}
	}

	// for right image
	p_border.push_back(Pivot(p_border[2].u+p_border[2].d,p_border[2].v,p_border[2].d));
	p_border.push_back(Pivot(p_border[3].u+p_border[3].d,p_border[3].v,p_border[3].d));

	// add border points to support points
	for (int32_t i=0; i<p_border.size(); i++)
		pivots.push_back(p_border[i]);
}


inline float StereoMatcher::findMatch(const int u, const int v,
                              const int width, const int height,
                              const int disp_min, const int disp_max,
                              const SobelDescriptor& desc1, const SobelDescriptor& desc2,
                              bool is_right)
{
  const int u_step      = 2;
  const int v_step      = 2;
  const int window_size = 3;

  // feature map 
  // x 0 0 0 x
  // 0 0 0 0 0
  // 0 0 c 0 0
  // 0 0 0 0 0
  // x 0 0 0 x 

  int desc_offset_1 = -16*u_step-16*width*v_step; 
  int desc_offset_2 = +16*u_step-16*width*v_step; 
  int desc_offset_3 = -16*u_step+16*width*v_step; 
  int desc_offset_4 = +16*u_step+16*width*v_step; 
  
  //__m128i means 128bits integer, when we want to write it , use __m128i _mm_set1_epi32(int i)
  __m128i xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;
  // 128 means 8 * 16, is exactly the dimension of feature vector
  // check if we are inside the image region
  if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
    
    // compute desc and start addresses
    int  line_offset = 16 * width * v;   // because get (v, u, x) == 16*(width*v+u)+x
    uint8_t *img1_line_addr,*img2_line_addr;
    if (!is_right) {
      img1_line_addr = desc1.descriptor.data + line_offset;
      img2_line_addr = desc2.descriptor.data + line_offset;
    } else {
      img1_line_addr = desc2.descriptor.data + line_offset;
      img2_line_addr = desc1.descriptor.data + line_offset;
    }

    // compute I1 block start addresses
    uint8_t* img1_block_addr = img1_line_addr + 16*u;
    uint8_t* img2_block_addr;
    
    // we require at least some texture
    // this equals the sum of feature vector can't be too small
    int sum = 0;

    // load first blocks to xmm registers
    // equals load feature vector
    xmm1 = _mm_load_si128((__m128i*)(img1_block_addr + desc_offset_1));
    xmm2 = _mm_load_si128((__m128i*)(img1_block_addr + desc_offset_2));
    xmm3 = _mm_load_si128((__m128i*)(img1_block_addr + desc_offset_3));
    xmm4 = _mm_load_si128((__m128i*)(img1_block_addr + desc_offset_4));

    // declare match energy for each disparity
    int u_warp;
    
    // best match
    int16_t min_1_E = 32767;
    int16_t min_1_d = -1;
    int16_t min_2_E = 32767;
    int16_t min_2_d = -1;

    // get valid disparity range
    int disp_min_valid = max(param.disp_min,0);
    int disp_max_valid = param.disp_max;
    if (!is_right) disp_max_valid = min(param.disp_max, u-window_size-u_step);
    else              disp_max_valid = min(param.disp_max, width-u-window_size-u_step);
    
    // assume, that we can compute at least 10 disparities for this pixel
    if (disp_max_valid-disp_min_valid<10)
      return -1;

    vector<int> var_space;
    var_space.reserve(disp_max_valid - disp_min_valid + 1);
    // for all disparities do
    for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {

      // warp u coordinate
      if (!is_right) u_warp = u-d;
      else              u_warp = u+d;

      // compute I2 block start addresses
      img2_block_addr = img2_line_addr+16*u_warp;

      // compute match energy at this disparity
      xmm6 = _mm_load_si128((__m128i*)(img2_block_addr + desc_offset_1));
      xmm6 = _mm_sad_epu8(xmm1,xmm6);
      xmm5 = _mm_load_si128((__m128i*)(img2_block_addr + desc_offset_2));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm2,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(img2_block_addr + desc_offset_3));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm3,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(img2_block_addr + desc_offset_4));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm4,xmm5),xmm6);
      sum  = _mm_extract_epi16(xmm6,0)+_mm_extract_epi16(xmm6,4);
      var_space.emplace_back(sum);
      
      // best + second best match
      if (sum<min_1_E) {
        min_2_E = min_1_E;   
        min_2_d = min_1_d;
        min_1_E = sum;
        min_1_d = d;
      } else if (sum<min_2_E) {
        min_2_E = sum;
        min_2_d = d;
      }
    }
    // check if best and second best match are available and if matching ratio is sufficient
    if (min_1_d>=0 && 
        min_2_d>=0 && 
        abs(min_2_d - min_1_d) == 1 && 
        (float)min_1_E < 0.85f*(float)min_2_E )
    {
      if(min_1_d == disp_min_valid || min_1_d == disp_max_valid) {
        return min_1_d;
      }
      else {
        Vector2f left(-1, var_space.at(min_1_d - disp_min_valid - 1)),
                 mid(0, var_space.at(min_1_d - disp_min_valid)),
                 right(1, var_space.at(min_1_d - disp_min_valid + 1));
        return fitQuadraticFuncion(left, mid, right) + min_1_d;;
      }
    }
    else
      return -1;
  } 
  else
    return -1;
}

inline void StereoMatcher::removeInconsistentSupportPoints(cv::Mat& disp, int incon_window_size, int incon_threshold, int incon_min_support) 
{
  
  // for all valid support points do
  const int disp_can_height = disp.rows;
  const int disp_can_width = disp.cols;
  for (int u_can=0; u_can<disp_can_width; u_can++) {
    for (int v_can=0; v_can<disp_can_height; v_can++) {
      float d_can = imref_f32(disp, v_can, u_can);
      if (d_can>=0) {
        
        // compute number of other points supporting the current point
        int support = 0;
        for (int u_can_2 = u_can - incon_window_size; u_can_2 <= u_can + incon_window_size; u_can_2++) {
          for (int v_can_2 = v_can - incon_window_size; v_can_2 <= v_can + incon_window_size; v_can_2++) {
            if (u_can_2 >= 0 && v_can_2 >= 0 && u_can_2 < disp_can_width && v_can_2 < disp_can_height) {
              // int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
              float d_can_2 = imref_f32(disp, v_can, u_can);
              if (d_can_2 >= 0 && fabs(d_can - d_can_2) <= incon_threshold)
                support++;
            }
          }
        }
        
        // invalidate support point if number of supporting points is too low
        if (support < incon_min_support)
          imref_f32(disp, v_can, u_can) = -1;
      }
    }
  }
}

inline float StereoMatcher::fitQuadraticFuncion(Vector2f& p1, Vector2f& p2, Vector2f& p3)
{
  Matrix3f A;
  Vector3f x, b;
  A(0, 0) = p1(0) * p1(0); A(0, 1) = p1(0); A(0, 2) = 1;
  A(1, 0) = p2(0) * p2(0); A(1, 1) = p2(0); A(1, 2) = 1;
  A(2, 0) = p3(0) * p3(0); A(2, 1) = p3(0); A(2, 2) = 1;
  b(0) = p1(1); b(1) = p2(1); b(2) = p3(1);
  x = A.fullPivHouseholderQr().solve(b);
  return -0.5f * x(1) / x(0);
}

inline cv::Scalar StereoMatcher::dispshader(float disparity)
{
  if(fabs(disparity - INVALID_DISP) < FLOAT_EPSILON) {
    return cv::Scalar(0, 0, 0);
  }
  int disp_band = 60;
  int disp_band_one_third = disp_band / 3;
  int blue, green, red;
  float disparity_bias = disparity - ((int)disparity / disp_band) * disp_band;
  if(disparity_bias >= 0 && disparity_bias < disp_band_one_third) {
    blue = std::round(((float)(disp_band_one_third - disparity_bias) / disp_band_one_third) * 255.0f);
    green = std::round(((float)(disparity_bias - 0) / disp_band_one_third) * 255.0f);
    red = 0;
  }
  else if(disparity_bias >= disp_band_one_third && disparity_bias < disp_band_one_third * 2) {
    blue = 0;
    green = std::round(((float)(disp_band_one_third * 2 - disparity_bias) / disp_band_one_third) * 255.0f);
    red = std::round(((float)(disparity_bias - disp_band_one_third) / disp_band_one_third) * 255.0f);
  }
  else {
    blue = std::round(((float)(disparity_bias - disp_band_one_third * 2) / disp_band_one_third) * 255.0f);
    green = 0;
    red = std::round(((float)(disp_band_one_third * 3 - disparity_bias) / disp_band_one_third) * 255.0f);
  }
  return cv::Scalar(blue, green, red);
} 

inline cv::Scalar StereoMatcher::covshader(float var)
{
  if(fabs(var - INVALID_VAR) < FLOAT_EPSILON) 
    return cv::Scalar(0, 0, 0);
  float bandwidth = 2;
  float half_bandwidth = bandwidth / 2.0f;
  int blue, green, red;
  if(var - std::floor(var/bandwidth)*bandwidth) {
    blue = std::round(((half_bandwidth - var) / half_bandwidth) * 255.0f);
    green = std::round(var / half_bandwidth * 255.0f);
    red = 0;
  }
  else {
    blue = 0;
    green = std::round((bandwidth - var) / half_bandwidth * 255.0f);
    red = std::round((var - half_bandwidth) / half_bandwidth * 255.0f);
  }
  return cv::Scalar(blue, green, red);
}

void StereoMatcher::calcDelaunay(const vector<Pivot>& pivots, vector<Triangle>& triangles, int32_t is_right)
{

  triangles.clear();
  // input/output structure for triangulation
  struct triangulateio in, out;
  int32_t k;

  // inputs
  in.numberofpoints = pivots.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
  k=0;
  if (!is_right) {
    for (int32_t i=0; i<pivots.size(); i++) {
      in.pointlist[k++] = pivots[i].u;
      in.pointlist[k++] = pivots[i].v;
    }
  } else {
    for (int32_t i=0; i<pivots.size(); i++) {
      in.pointlist[k++] = pivots[i].u - pivots[i].d;
      in.pointlist[k++] = pivots[i].v;
    }
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;
  
  // outputs
  out.pointlist              = NULL;
  out.pointattributelist     = NULL;
  out.pointmarkerlist        = NULL;
  out.trianglelist           = NULL;
  out.triangleattributelist  = NULL;
  out.neighborlist           = NULL;
  out.segmentlist            = NULL;
  out.segmentmarkerlist      = NULL;
  out.edgelist               = NULL;
  out.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zQB";
  triangulate(parameters, &in, &out, NULL);
  
  // put resulting triangles into vector triangles
  int invalid_triangle = 0;
  triangles.reserve(out.numberoftriangles);
  k=0;
  for (int32_t i=0; i<out.numberoftriangles; i++) {
    Triangle triangle_i = Triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]);
    k+=3;
    // if(checkDelaunay(pivots, triangle_i)) {
      triangles.emplace_back(triangle_i);
      invalid_triangle++;
    // }
  }
  Triangle temp(0, 0, 0);
  triangles.resize(invalid_triangle, temp);
  // free memory used for triangulation
  free(in.pointlist);
  free(out.pointlist);
  free(out.trianglelist);

}

bool StereoMatcher::checkDelaunay(const std::vector<Pivot>& pivots, const Triangle& triangle)
{
  const Pivot &p1 = pivots.at(triangle.c1), &p2 = pivots.at(triangle.c2), &p3 = pivots.at(triangle.c3);
  float depth1 = fb / p1.d, depth2 = fb / p2.d, depth3 = fb / p3.d;

  float delta_u = max3(p1.u, p2.u, p3.u) - min3(p1.u, p2.u, p3.u);
  float delta_v = max3(p1.v, p2.v, p3.v) - min3(p1.v, p2.v, p3.v);
  float delta_depth = max3(depth1, depth2, depth3) - min3(depth1, depth2, depth3); 

  // if(delta_depth > param.depth_diff_threshold || delta_u > param.visual_range_threshold * width || delta_v > param.visual_range_threshold * height) 
  if(delta_depth > param.depth_diff_threshold) 
    return false;
  return true;
}

void StereoMatcher::calcPlanes(vector<Pivot> pivots, vector<Triangle>& triangles, int32_t is_right) 
{
  // init matrices
  micro_matrix_lib::Matrix A(3,3);
  micro_matrix_lib::Matrix b(3,1);
  
  // for all triangles do
  for (int32_t i=0; i<triangles.size(); i++) {
    // get triangle corner indices
    int32_t c1 = triangles[i].c1;
    int32_t c2 = triangles[i].c2;
    int32_t c3 = triangles[i].c3;

    // compute matrix A for linear system of left triangle
    A.val[0][0] = pivots[c1].u;
    A.val[1][0] = pivots[c2].u;
    A.val[2][0] = pivots[c3].u;
    A.val[0][1] = pivots[c1].v; A.val[0][2] = 1;
    A.val[1][1] = pivots[c2].v; A.val[1][2] = 1;
    A.val[2][1] = pivots[c3].v; A.val[2][2] = 1;
    
    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = pivots[c1].d;
    b.val[1][0] = pivots[c2].d;
    b.val[2][0] = pivots[c3].d;
    
    // on success of gauss jordan elimination
    if (b.solve(A)) {
      // grab results from b
      triangles[i].t1a = b.val[0][0];
      triangles[i].t1b = b.val[1][0];
      triangles[i].t1c = b.val[2][0];
    // otherwise: invalid
    } 
    else {
      triangles[i].t1a = 0;
      // compute matrix A for linear system of right triangle
      A.val[0][0] = pivots[c1].u-pivots[c1].d;
      A.val[1][0] = pivots[c2].u-pivots[c2].d;
      A.val[2][0] = pivots[c3].u-pivots[c3].d;
      A.val[0][1] = pivots[c1].v; A.val[0][2] = 1;
      A.val[1][1] = pivots[c2].v; A.val[1][2] = 1;
      A.val[2][1] = pivots[c3].v; A.val[2][2] = 1;
      
      // compute vector b for linear system (containing the disparities)
      b.val[0][0] = pivots[c1].d;
      b.val[1][0] = pivots[c2].d;
      b.val[2][0] = pivots[c3].d;
      
      // on success of gauss jordan elimination
      if (b.solve(A)) {
        // grab results from b
        triangles[i].t2a = b.val[0][0];
        triangles[i].t2b = b.val[1][0];
        triangles[i].t2c = b.val[2][0];
        
      // otherwise: invalid
      } 
      else {
        triangles[i].t2a = 0;
        triangles[i].t2b = 0;
        triangles[i].t2c = 0;
      }
    }  
  }
}

void StereoMatcher::updatePrior(cv::Mat& disp, cv::Mat& var, const std::vector<Pivot>& pivots, const std::vector<Triangle>& triangles, SensorType type)
{
  for(int i = 0; i < triangles.size(); ++i) {

    const Triangle& triangle = triangles.at(i);
    Pivot p1 = pivots.at(triangle.c1), p2 = pivots.at(triangle.c2), p3 = pivots.at(triangle.c3);
    const float &a = triangle.t1a, &b = triangle.t1b, &c = triangle.t1c;

    std::vector<cv::Point2i> validPoints = selectRegion(pivots, triangle);

    for(int j = 0; j < validPoints.size(); ++j) {
      cv::Point2i point = Point2i(validPoints.at(j).x, validPoints.at(j).y);

      // float depth_buffer[3] = {distance(point, Point2f(p1.u, p1.v)),
      //                          distance(point, Point2f(p2.u, p2.v)),
      //                          distance(point, Point2f(p3.u, p3.v))};
      // float depth_buffer[3] = {disp2depth(p1.d, fb, f), disp2depth(p2.d, fb, f), disp2depth(p3.d, fb, f)};
      float disp_ij = a * point.x + b * point.y + c;
      float stdcov_ij;
      if(type == STEREO) {
        stdcov_ij = param.sigma_visual;
      }
      else if(type == LIDAR) {
        stdcov_ij = lidar_sigma(disp_ij, param.sigma_lidar, fb);
      }
      if(imref_f32(disp, point.y, point.x) == INVALID_DISP) {
        imref_f32(disp, point.y, point.x) = disp_ij;
        imref_f32(var, point.y, point.x) = stdcov_ij * stdcov_ij;
      }
    }
  }
}

inline void StereoMatcher::refineDisparity(int32_t u, int32_t v, const SobelDescriptor& desc1, const SobelDescriptor& desc2,
                                    float disp_in, float var_in, float& disp_out, float& var_out)
{
  // does this patch have enough texture?
  const int32_t windowSize = 2;
  int32_t sum = 0;
  uint8_t *img1_block_addr = desc1.descriptor.data + 16 * (u + width * v);

  float disp_refined;
  float cov_refined;

  int32_t disp_cur, u_warp, val; // (d_i in disp space)  (u_i in right)  (loss of descriptor) 
  __m128i xmm1    = _mm_load_si128((__m128i*)img1_block_addr);
  __m128i xmm2;
  int32_t dispMin = std::max(param.disp_min, (int32_t)std::ceil(disp_in - 3 * std::sqrt(var_in)));
  int32_t dispMax = std::min(param.disp_max, (int32_t)std::floor(disp_in + 3 * std::sqrt(var_in)));

  // assert(disp_in <= dispMax && disp_in >= dispMin);

  // const float halfRegion = std::min((disp_in - dispMin), (dispMax - disp_in));
  // dispMin = std::ceil(disp_in - halfRegion);
  // dispMax = std::floor(disp_in + halfRegion);
  // std::vector<int32_t> val_space(disp_max - disp_min + 1);

  float dispMoment1d = 0, dispMoment2d = 0, prob_sum = 0;
  for(disp_cur = dispMin; disp_cur <= dispMax; ++disp_cur) {
    u_warp = u - disp_cur;
    if(u_warp < windowSize || u_warp >= width-windowSize) {
      disp_out = INVALID_DISP;
      var_out = INVALID_VAR;
    }
    uint8_t *img2BlockAddr = desc2.descriptor.data + 16 * (u_warp + width * v);
    xmm2 = _mm_load_si128((__m128i*)img2BlockAddr);
    xmm2 = _mm_sad_epu8(xmm1,xmm2);
    val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
    // val_space.emplace_back(val);
    const float prob_fusion = gaussianPdf(disp_cur, disp_in, var_in) * exponentialPdf((float)val, param.beta);
    // const float prob_fusion = gaussian_uniform_pdf(disp_cur, param.gamma, disparity, covariance) * exponentialPdf((float)val, param.beta);
    dispMoment1d += disp_cur * prob_fusion;
    dispMoment2d += disp_cur * disp_cur * prob_fusion;
    prob_sum += prob_fusion;
    float disp_ave = (dispMax + dispMin) / 2 / prob_sum;
    disp_refined = dispMoment1d / prob_sum;
    cov_refined = dispMoment2d / prob_sum - disp_ave * disp_ave;
  }
  disp_out = disp_refined;
  var_out = cov_refined;
}

void StereoMatcher::leftRightConsistencyCheck(cv::Mat& disp1, cv::Mat& disp2, cv::Mat& var1, cv::Mat& var2)
{
  cv::Mat disp1_copy, disp2_copy;
  disp1_copy = disp1.clone();
  disp2_copy = disp2.clone();

	// loop variables
	uint32_t addr,addr_warp;
	float    u_warp1,u_warp2,d1,d2;

	// for all image points do
	for(int u = 0; u < width; u++) {
		for(int v = 0; v < height; v++) {

      d1 = imref_f32(disp1_copy, v, u);
      d2 = imref_f32(disp2_copy, v, u);
      u_warp1 = (float)u - d1;
      u_warp2 = (float)u + d2;

			// check if left disparity is valid
			if (d1 >= 0 && u_warp1 >= 0 && u_warp1 < width) {
				// if check failed
        if(fabs(imref_f32(disp2_copy, v, (int)u_warp1) - d1)>param.lr_threshold) {
          imref_f32(disp1, v, u) = -1.0f;
          imref_f32(var1, v, u) = -1.0f;
        }
			} 
      else {
        imref_f32(disp1, v, u) = -1.0f;
        imref_f32(var1, v, u) = -1.0f;
      }

			// check if right disparity is valid
			if (d2>=0 && u_warp2>=0 && u_warp2 < width) {

				// if check failed
        if(fabs(imref_f32(disp1_copy, v, (int)u_warp2) - d2) > param.lr_threshold) {
          imref_f32(disp2, v, u) = -1.0f;
          imref_f32(var2, v, u) = -1.0f;
        }
			} 
      else {
        imref_f32(disp2, v, u) = -1.0f;
        imref_f32(var2, v, u) = -1.0f;
      }
		}
	}
}

void StereoMatcher::updatePosteriori(cv::Mat& disp, cv::Mat& var, SobelDescriptor& desc1, SobelDescriptor& desc2, SensorType type)
{
  for(int u = 0; u < width; ++u) {
    for(int v = 0; v < height; ++v) {
      float dispPrior_ij, varPrior_ij;
      dispPrior_ij = imref_f32(disp, v, u);
      varPrior_ij = imref_f32(var, v, u);
      std::pair<float, float> result;
      float disp_posteriori_ij, var_posteriori_ij;

      if(dispPrior_ij == INVALID_DISP && varPrior_ij == INVALID_DISP) 
        continue;
    
      refineDisparity(u, v, desc1, desc2, dispPrior_ij, varPrior_ij, disp_posteriori_ij, var_posteriori_ij);
      imref_f32(disp, v, u) = disp_posteriori_ij;
      imref_f32(var, v, u) = var_posteriori_ij;
    }
  }
}

void StereoMatcher::createGrid(vector<Pivot> pivots,int32_t* disparity_grid,int32_t* grid_dims,bool right_image) {

	// get grid dimensions
	int32_t grid_width  = grid_dims[1];
	int32_t grid_height = grid_dims[2];

	// allocate temporary memory
	int32_t* temp1 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
	int32_t* temp2 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));

	// for all support points do
	for (int32_t i=0; i<pivots.size(); i++) {

		// compute disparity range to fill for this support point [dmin dcur dmax]
		int32_t x_curr = pivots[i].u;
		int32_t y_curr = pivots[i].v;
		int32_t d_curr = pivots[i].d;
		int32_t d_min  = max(d_curr-1,0);
		int32_t d_max  = min(d_curr+1,param.disp_max);

		// fill disparity grid helper
		for (int32_t d=d_min; d<=d_max; d++) {
			int32_t x;
			if (!right_image)
				x = floor((float)(x_curr/param.grid_size));
			else
				x = floor((float)(x_curr-d_curr)/(float)param.grid_size);
			int32_t y = floor((float)y_curr/(float)param.grid_size);

			// point may potentially lay outside (corner points)
			if (x>=0 && x<grid_width &&y>=0 && y<grid_height) {
				int32_t addr = getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1);
				*(temp1+addr) = 1;
			}
		}
	}

	// diffusion pointers
	const int32_t* tl = temp1 + (0*grid_width+0)*(param.disp_max+1);
	const int32_t* tc = temp1 + (0*grid_width+1)*(param.disp_max+1);
	const int32_t* tr = temp1 + (0*grid_width+2)*(param.disp_max+1);
	const int32_t* cl = temp1 + (1*grid_width+0)*(param.disp_max+1);
	const int32_t* cc = temp1 + (1*grid_width+1)*(param.disp_max+1);
	const int32_t* cr = temp1 + (1*grid_width+2)*(param.disp_max+1);
	const int32_t* bl = temp1 + (2*grid_width+0)*(param.disp_max+1);
	const int32_t* bc = temp1 + (2*grid_width+1)*(param.disp_max+1);
	const int32_t* br = temp1 + (2*grid_width+2)*(param.disp_max+1);

	int32_t* result    = temp2 + (1*grid_width+1)*(param.disp_max+1);
	int32_t* end_input = temp1 + grid_width*grid_height*(param.disp_max+1);

	// diffuse temporary grid
	for( ; br != end_input; tl++, tc++, tr++, cl++, cc++, cr++, bl++, bc++, br++, result++ )
		*result = *tl | *tc | *tr | *cl | *cc | *cr | *bl | *bc | *br;

	// for all grid positions create disparity grid
	for (int32_t x=0; x<grid_width; x++) {
		for (int32_t y=0; y<grid_height; y++) {

			// start with second value (first is reserved for count)
			int32_t curr_ind = 1;

			// for all disparities do
			for (int32_t d=0; d<=param.disp_max; d++) {

				// if yes => add this disparity to current cell
				if (*(temp2 + getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1))>0) {
					*(disparity_grid + getAddressOffsetGrid(x,y,curr_ind,grid_width,param.disp_max+2))=d;
					curr_ind++;
				}
			}

			// finally set number of indices
			*(disparity_grid + getAddressOffsetGrid(x,y,0,grid_width,param.disp_max+2))=curr_ind-1;
		}
	}

	// release temporary memory
	free(temp1);
	free(temp2);
}

inline void StereoMatcher::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
		const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
	xmm2 = _mm_load_si128(I2_block_addr);
	xmm2 = _mm_sad_epu8(xmm1,xmm2);
	val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4)+w;
	if (val<min_val) {
		min_val = val;
		min_d   = d;
	}
}

inline void StereoMatcher::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,
		const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
	xmm2 = _mm_load_si128(I2_block_addr);
	xmm2 = _mm_sad_epu8(xmm1,xmm2);
	val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
	if (val<min_val) {
		min_val = val;
		min_d   = d;
	}
}

inline void StereoMatcher::findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
		int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
		int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D){

	// get image width and height
	const int32_t disp_num    = grid_dims[0]-1;
	const int32_t window_size = 2;

	// address of disparity we want to compute
	uint32_t d_addr;
  d_addr = getAddressOffsetImage(u,v,width);

	// check if u is ok
	if (u<window_size || u>=width-window_size)
		return;

	// compute line start address
	int32_t  line_offset = 16*width*max(min(v,height-3),2);
	uint8_t *I1_line_addr,*I2_line_addr;
	if (!right_image) {
		I1_line_addr = I1_desc+line_offset;
		I2_line_addr = I2_desc+line_offset;
	} else {
		I1_line_addr = I2_desc+line_offset;
		I2_line_addr = I1_desc+line_offset;
	}

	// compute I1 block start address
	uint8_t* I1_block_addr = I1_line_addr+16*u;

	// does this patch have enough texture?
	int32_t sum = 0;
	for (int32_t i=0; i<16; i++)
		sum += abs((int32_t)(*(I1_block_addr+i))-128);
	if (sum<param.match_texture)
		return;

	// compute disparity, min disparity and max disparity of plane prior
	int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
	int32_t d_plane_min = max(d_plane-plane_radius,0);
	int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

	// get grid pointer
	int32_t  grid_x    = (int32_t)floor((float)u/(float)param.grid_size);
	int32_t  grid_y    = (int32_t)floor((float)v/(float)param.grid_size);
	uint32_t grid_addr = getAddressOffsetGrid(grid_x,grid_y,0,grid_dims[1],grid_dims[0]);
	int32_t  num_grid  = *(disparity_grid+grid_addr);
	int32_t* d_grid    = disparity_grid+grid_addr+1;

	// loop variables
	int32_t d_curr, u_warp, val;
	int32_t min_val = 10000;
	int32_t min_d   = -1;
	__m128i xmm1    = _mm_load_si128((__m128i*)I1_block_addr);
	__m128i xmm2;

	// left image
	if (!right_image) {
		for (int32_t i=0; i<num_grid; i++) {
			d_curr = d_grid[i];
			if (d_curr<d_plane_min || d_curr>d_plane_max) {
				u_warp = u-d_curr;
				if (u_warp<window_size || u_warp>=width-window_size)
					continue;
				updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
			}
		}
		for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
			u_warp = u-d_curr;
			if (u_warp<window_size || u_warp>=width-window_size)
				continue;
			updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
		}

		// right image
	} else {
		for (int32_t i=0; i<num_grid; i++) {
			d_curr = d_grid[i];
			if (d_curr<d_plane_min || d_curr>d_plane_max) {
				u_warp = u+d_curr;
				if (u_warp<window_size || u_warp>=width-window_size)
					continue;
				updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
			}
		}
		for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
			u_warp = u+d_curr;
			if (u_warp<window_size || u_warp>=width-window_size)
				continue;
			updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
		}
	}

	// set disparity value
	if (min_d>=0) *(D+d_addr) = min_d; // MAP value (min neg-Log probability)
	else          *(D+d_addr) = -1;    // invalid disparity
}

// TODO: %2 => more elegantly
void StereoMatcher::computeDisparity(vector<Pivot> pivots,vector<Triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
		uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D) 
{

	// number of disparities
	//const int32_t disp_num  = grid_dims[0]-1;
	int disp_num = grid_dims[0]-1;

	// descriptor window_size
	int32_t window_size = 2;

	// init disparity image to -10
	if (param.subsampling) {
		for (int32_t i=0; i<(width/2)*(height/2); i++)
			*(D+i) = -10;
	} else {
		for (int32_t i=0; i<width*height; i++)
			*(D+i) = -10;
	}

	// pre-compute prior
	float two_sigma_squared = 2*param.sigma*param.sigma;
	int32_t* P = new int32_t[disp_num];
	for (int32_t delta_d=0; delta_d<disp_num; delta_d++)
		P[delta_d] = (int32_t)((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
	int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius),(float)2.0);

	// loop variables
	int32_t c1, c2, c3;
	float plane_a,plane_b,plane_c,plane_d;
	uint32_t i;

	// for all triangles do
#pragma omp parallel for num_threads(3) default(none)\
	private(i, plane_a, plane_b, plane_c, plane_d, c1, c2, c3)\
	shared(P, plane_radius, two_sigma_squared, disp_num, window_size, pivots, tri, disparity_grid, grid_dims, I1_desc, I2_desc, right_image, D)
	for (i=0; i<tri.size(); i++) {

		//printf("Matching thread %d\n", omp_get_thread_num());
		// get plane parameters
		uint32_t p_i = i*3;
		if (!right_image) {
			plane_a = tri[i].t1a;
			plane_b = tri[i].t1b;
			plane_c = tri[i].t1c;
			plane_d = tri[i].t2a;
		} else {
			plane_a = tri[i].t2a;
			plane_b = tri[i].t2b;
			plane_c = tri[i].t2c;
			plane_d = tri[i].t1a;
		}

		// triangle corners
		c1 = tri[i].c1;
		c2 = tri[i].c2;
		c3 = tri[i].c3;

		// sort triangle corners wrt. u (ascending)
		float tri_u[3];
		if (!right_image) {
			tri_u[0] = pivots[c1].u;
			tri_u[1] = pivots[c2].u;
			tri_u[2] = pivots[c3].u;
		} else {
			tri_u[0] = pivots[c1].u-pivots[c1].d;
			tri_u[1] = pivots[c2].u-pivots[c2].d;
			tri_u[2] = pivots[c3].u-pivots[c3].d;
		}
		float tri_v[3] = {pivots[c1].v,pivots[c2].v,pivots[c3].v};

		for (uint32_t j=0; j<3; j++) {
			for (uint32_t k=0; k<j; k++) {
				if (tri_u[k]>tri_u[j]) {
					float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
					float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
				}
			}
		}

		// rename corners
		float A_u = tri_u[0]; float A_v = tri_v[0];
		float B_u = tri_u[1]; float B_v = tri_v[1];
		float C_u = tri_u[2]; float C_v = tri_v[2];

		// compute straight lines connecting triangle corners
		float AB_a = 0; float AC_a = 0; float BC_a = 0;
		if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
		if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
		if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
		float AB_b = A_v-AB_a*A_u;
		float AC_b = A_v-AC_a*A_u;
		float BC_b = B_v-BC_a*B_u;

		// a plane is only valid if itself and its projection
		// into the other image is not too much slanted
		bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

		// first part (triangle corner A->B)
		if ((int32_t)(A_u)!=(int32_t)(B_u)) {
			for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
				if (!param.subsampling || u%2==0) {
					int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
					int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
					for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
						if (!param.subsampling || v%2==0) {
							findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
									I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
						}
				}
			}
		}

		// second part (triangle corner B->C)
		if ((int32_t)(B_u)!=(int32_t)(C_u)) {
			for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
				if (!param.subsampling || u%2==0) {
					int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
					int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
					for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
						if (!param.subsampling || v%2==0) {
							findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
									I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
						}
				}
			}
		}

	}

	delete[] P;
}



void StereoMatcher::removeSmallSegments (float* D) {

	// get disparity image dimensions
	int32_t D_width        = width;
	int32_t D_height       = height;
	int32_t D_speckle_size = param.speckle_size;
	if (param.subsampling) {
		D_width        = width/2;
		D_height       = height/2;
		D_speckle_size = sqrt((float)param.speckle_size)*2;
	}

	// allocate memory on heap for dynamic programming arrays
	int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t seg_list_count;
	int32_t seg_list_curr;
	int32_t u_neighbor[4];
	int32_t v_neighbor[4];
	int32_t u_seg_curr;
	int32_t v_seg_curr;

	// declare loop variables
	int32_t addr_start, addr_curr, addr_neighbor;

	// for all pixels do
	for (int32_t u=0; u<D_width; u++) {
		for (int32_t v=0; v<D_height; v++) {

			// get address of first pixel in this segment
			addr_start = getAddressOffsetImage(u,v,D_width);

			// if this pixel has not already been processed
			if (*(D_done+addr_start)==0) {

				// init segment list (add first element
				// and set it to be the next element to check)
				*(seg_list_u+0) = u;
				*(seg_list_v+0) = v;
				seg_list_count  = 1;
				seg_list_curr   = 0;

				// add neighboring segments as long as there
				// are none-processed pixels in the seg_list;
				// none-processed means: seg_list_curr<seg_list_count
				while (seg_list_curr<seg_list_count) {

					// get current position from seg_list
					u_seg_curr = *(seg_list_u+seg_list_curr);
					v_seg_curr = *(seg_list_v+seg_list_curr);

					// get address of current pixel in this segment
					addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,D_width);

					// fill list with neighbor positions
					u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
					u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
					u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
					u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;

					// for all neighbors do
					for (int32_t i=0; i<4; i++) {

						// check if neighbor is inside image
						if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {

							// get neighbor pixel address
							addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],D_width);

							// check if neighbor has not been added yet and if it is valid
							if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

								// is the neighbor similar to the current pixel
								// (=belonging to the current segment)
								if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {

									// add neighbor coordinates to segment list
									*(seg_list_u+seg_list_count) = u_neighbor[i];
									*(seg_list_v+seg_list_count) = v_neighbor[i];
									seg_list_count++;

									// set neighbor pixel in I_done to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									*(D_done+addr_neighbor) = 1;
								}
							}

						}
					}

					// set current pixel in seg_list to "done"
					seg_list_curr++;

					// set current pixel in I_done to "done"
					*(D_done+addr_curr) = 1;

				} // end: while (seg_list_curr<seg_list_count)

				// if segment NOT large enough => invalidate pixels
				if (seg_list_count<D_speckle_size) {

					// for all pixels in current segment invalidate pixels
					for (int32_t i=0; i<seg_list_count; i++) {
						addr_curr = getAddressOffsetImage(*(seg_list_u+i),*(seg_list_v+i),D_width);
						*(D+addr_curr) = -10;
					}
				}
			} // end: if (*(I_done+addr_start)==0)

		}
	}

	// free memory
	free(D_done);
	free(seg_list_u);
	free(seg_list_v);
}

void StereoMatcher::gapInterpolation(float* D) {

	// get disparity image dimensions
	int32_t D_width          = width;
	int32_t D_height         = height;
	int32_t D_ipol_gap_width = param.ipol_gap_width;
	if (param.subsampling) {
		D_width          = width/2;
		D_height         = height/2;
		D_ipol_gap_width = param.ipol_gap_width/2+1;
	}

	// discontinuity threshold
	float discon_threshold = 3.0;

	// declare loop variables
	int32_t count,addr,v_first,v_last,u_first,u_last;
	float   d1,d2,d_ipol;

	// 1. Row-wise:
	// for each row do
	for (int32_t v=0; v<D_height; v++) {

		// init counter
		count = 0;

		// for each element of the row do
		for (int32_t u=0; u<D_width; u++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {

				// check if speckle is small enough
				if (count>=1 && count<=D_ipol_gap_width) {

					// first and last value for interpolation
					u_first = u-count;
					u_last  = u-1;

					// if value in range
					if (u_first>0 && u_last<D_width-1) {

						// compute mean disparity
						d1 = *(D+getAddressOffsetImage(u_first-1,v,D_width));
						d2 = *(D+getAddressOffsetImage(u_last+1,v,D_width));
						if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
						else                              d_ipol = min(d1,d2);

						// set all values to d_ipol
						for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
							*(D+getAddressOffsetImage(u_curr,v,D_width)) = d_ipol;
					}

				}

				// reset counter
				count = 0;

				// otherwise increment counter
			} else {
				count++;
			}
		}

		// if full size disp map requested
		if (param.add_corners) {

			// extrapolate to the left
			for (int32_t u=0; u<D_width; u++) {

				// get address of this location
				addr = getAddressOffsetImage(u,v,D_width);

				// if disparity valid
				if (*(D+addr)>=0) {
					for (int32_t u2=max(u-D_ipol_gap_width,0); u2<u; u2++)
						*(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
					break;
				}
			}

			// extrapolate to the right
			for (int32_t u=D_width-1; u>=0; u--) {

				// get address of this location
				addr = getAddressOffsetImage(u,v,D_width);

				// if disparity valid
				if (*(D+addr)>=0) {
					for (int32_t u2=u; u2<=min(u+D_ipol_gap_width,D_width-1); u2++)
						*(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
					break;
				}
			}
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int32_t u=0; u<D_width; u++) {

		// init counter
		count = 0;

		// for each element of the column do
		for (int32_t v=0; v<D_height; v++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {

				// check if gap is small enough
				if (count>=1 && count<=D_ipol_gap_width) {

					// first and last value for interpolation
					v_first = v-count;
					v_last  = v-1;

					// if value in range
					if (v_first>0 && v_last<D_height-1) {

						// compute mean disparity
						d1 = *(D+getAddressOffsetImage(u,v_first-1,D_width));
						d2 = *(D+getAddressOffsetImage(u,v_last+1,D_width));
						if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
						else                              d_ipol = min(d1,d2);

						// set all values to d_ipol
						for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
							*(D+getAddressOffsetImage(u,v_curr,D_width)) = d_ipol;
					}

				}

				// reset counter
				count = 0;

				// otherwise increment counter
			} else {
				count++;
			}
		}
	}
}

// implements approximation to bilateral filtering
void StereoMatcher::adaptiveMean (float* D) {

	// get disparity image dimensions
	int32_t D_width          = width;
	int32_t D_height         = height;
	if (param.subsampling) {
		D_width          = width/2;
		D_height         = height/2;
	}

	// allocate temporary memory
	float* D_copy = (float*)malloc(D_width*D_height*sizeof(float));
	float* D_tmp  = (float*)malloc(D_width*D_height*sizeof(float));
	memcpy(D_copy,D,D_width*D_height*sizeof(float));

	// zero input disparity maps to -10 (this makes the bilateral
	// weights of all valid disparities to 0 in this region)
	for (int32_t i=0; i<D_width*D_height; i++) {
		if (*(D+i)<0) {
			*(D_copy+i) = -10;
			*(D_tmp+i)  = -10;
		}
	}

	__m128 xconst0 = _mm_set1_ps(0);
	__m128 xconst4 = _mm_set1_ps(4);
	__m128 xval,xweight1,xweight2,xfactor1,xfactor2;

	float *val     = (float *)_mm_malloc(8*sizeof(float),16);
	float *weight  = (float*)_mm_malloc(4*sizeof(float),16);
	float *factor  = (float*)_mm_malloc(4*sizeof(float),16);

	// set absolute mask
	__m128 xabsmask = _mm_set1_ps(0x7FFFFFFF);

	// when doing subsampling: 4 pixel bilateral filter width
	if (param.subsampling) {

		// horizontal filter
		for (int32_t v=3; v<D_height-3; v++) {

			// init
			for (int32_t u=0; u<3; u++)
				val[u] = *(D_copy+v*D_width+u);

			// loop
			for (int32_t u=3; u<D_width; u++) {

				// set
				float val_curr = *(D_copy+v*D_width+(u-1));
				val[u%4] = *(D_copy+v*D_width+u);

				xval     = _mm_load_ps(val);
				xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight1 = _mm_and_ps(xweight1,xabsmask);
				xweight1 = _mm_sub_ps(xconst4,xweight1);
				xweight1 = _mm_max_ps(xconst0,xweight1);
				xfactor1 = _mm_mul_ps(xval,xweight1);

				_mm_store_ps(weight,xweight1);
				_mm_store_ps(factor,xfactor1);

				float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
				float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

				if (weight_sum>0) {
					float d = factor_sum/weight_sum;
					if (d>=0) *(D_tmp+v*D_width+(u-1)) = d;
				}
			}
		}

		// vertical filter
		for (int32_t u=3; u<D_width-3; u++) {

			// init
			for (int32_t v=0; v<3; v++)
				val[v] = *(D_tmp+v*D_width+u);

			// loop
			for (int32_t v=3; v<D_height; v++) {

				// set
				float val_curr = *(D_tmp+(v-1)*D_width+u);
				val[v%4] = *(D_tmp+v*D_width+u);

				xval     = _mm_load_ps(val);
				xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight1 = _mm_and_ps(xweight1,xabsmask);
				xweight1 = _mm_sub_ps(xconst4,xweight1);
				xweight1 = _mm_max_ps(xconst0,xweight1);
				xfactor1 = _mm_mul_ps(xval,xweight1);

				_mm_store_ps(weight,xweight1);
				_mm_store_ps(factor,xfactor1);

				float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
				float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

				if (weight_sum>0) {
					float d = factor_sum/weight_sum;
					if (d>=0) *(D+(v-1)*D_width+u) = d;
				}
			}
		}

		// full resolution: 8 pixel bilateral filter width
	} else {


		// horizontal filter
		for (int32_t v=3; v<D_height-3; v++) {

			// init
			for (int32_t u=0; u<7; u++)
				val[u] = *(D_copy+v*D_width+u);

			// loop
			for (int32_t u=7; u<D_width; u++) {

				// set
				float val_curr = *(D_copy+v*D_width+(u-3));
				val[u%8] = *(D_copy+v*D_width+u);

				xval     = _mm_load_ps(val);
				xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight1 = _mm_and_ps(xweight1,xabsmask);
				xweight1 = _mm_sub_ps(xconst4,xweight1);
				xweight1 = _mm_max_ps(xconst0,xweight1);
				xfactor1 = _mm_mul_ps(xval,xweight1);

				xval     = _mm_load_ps(val+4);
				xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight2 = _mm_and_ps(xweight2,xabsmask);
				xweight2 = _mm_sub_ps(xconst4,xweight2);
				xweight2 = _mm_max_ps(xconst0,xweight2);
				xfactor2 = _mm_mul_ps(xval,xweight2);

				xweight1 = _mm_add_ps(xweight1,xweight2);
				xfactor1 = _mm_add_ps(xfactor1,xfactor2);

				_mm_store_ps(weight,xweight1);
				_mm_store_ps(factor,xfactor1);

				float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
				float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

				if (weight_sum>0) {
					float d = factor_sum/weight_sum;
					if (d>=0) *(D_tmp+v*D_width+(u-3)) = d;
				}
			}
		}

		// vertical filter
		for (int32_t u=3; u<D_width-3; u++) {

			// init
			for (int32_t v=0; v<7; v++)
				val[v] = *(D_tmp+v*D_width+u);

			// loop
			for (int32_t v=7; v<D_height; v++) {

				// set
				float val_curr = *(D_tmp+(v-3)*D_width+u);
				val[v%8] = *(D_tmp+v*D_width+u);

				xval     = _mm_load_ps(val);
				xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight1 = _mm_and_ps(xweight1,xabsmask);
				xweight1 = _mm_sub_ps(xconst4,xweight1);
				xweight1 = _mm_max_ps(xconst0,xweight1);
				xfactor1 = _mm_mul_ps(xval,xweight1);

				xval     = _mm_load_ps(val+4);
				xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
				xweight2 = _mm_and_ps(xweight2,xabsmask);
				xweight2 = _mm_sub_ps(xconst4,xweight2);
				xweight2 = _mm_max_ps(xconst0,xweight2);
				xfactor2 = _mm_mul_ps(xval,xweight2);

				xweight1 = _mm_add_ps(xweight1,xweight2);
				xfactor1 = _mm_add_ps(xfactor1,xfactor2);

				_mm_store_ps(weight,xweight1);
				_mm_store_ps(factor,xfactor1);

				float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
				float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

				if (weight_sum>0) {
					float d = factor_sum/weight_sum;
					if (d>=0) *(D+(v-3)*D_width+u) = d;
				}
			}
		}
	}

	// free memory
	_mm_free(val);
	_mm_free(weight);
	_mm_free(factor);
	free(D_copy);
	free(D_tmp);
}

void StereoMatcher::median (float* D) {

	// get disparity image dimensions
	int32_t D_width          = width;
	int32_t D_height         = height;
	if (param.subsampling) {
		D_width          = width/2;
		D_height         = height/2;
	}

	// temporary memory
	float *D_temp = (float*)calloc(D_width*D_height,sizeof(float));

	int32_t window_size = 3;

	float *vals = new float[window_size*2+1];
	int32_t i,j;
	float temp;

	// first step: horizontal median filter
	for (int32_t u=window_size; u<D_width-window_size; u++) {
		for (int32_t v=window_size; v<D_height-window_size; v++) {
			if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {
				j = 0;
				for (int32_t u2=u-window_size; u2<=u+window_size; u2++) {
					temp = *(D+getAddressOffsetImage(u2,v,D_width));
					i = j-1;
					while (i>=0 && *(vals+i)>temp) {
						*(vals+i+1) = *(vals+i);
						i--;
					}
					*(vals+i+1) = temp;
					j++;
				}
				*(D_temp+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
			} else {
				*(D_temp+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
			}

		}
	}

	// second step: vertical median filter
	for (int32_t u=window_size; u<D_width-window_size; u++) {
		for (int32_t v=window_size; v<D_height-window_size; v++) {
			if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {
				j = 0;
				for (int32_t v2=v-window_size; v2<=v+window_size; v2++) {
					temp = *(D_temp+getAddressOffsetImage(u,v2,D_width));
					i = j-1;
					while (i>=0 && *(vals+i)>temp) {
						*(vals+i+1) = *(vals+i);
						i--;
					}
					*(vals+i+1) = temp;
					j++;
				}
				*(D+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
			} else {
				*(D+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
			}
		}
	}

	free(D_temp);
	free(vals);
}




inline std::vector<cv::Point2i> StereoMatcher::selectRegion(const std::vector<Pivot>& pivots, const Triangle& triangle)
{
  std::vector<cv::Point2i> points;
  const float& u1 = pivots.at(triangle.c1).u, u2 = pivots.at(triangle.c2).u, u3 = pivots.at(triangle.c3).u;
  const float& v1 = pivots.at(triangle.c1).v, v2 = pivots.at(triangle.c2).v, v3 = pivots.at(triangle.c3).v;
  const float& d1 = pivots.at(triangle.c1).d, d2 = pivots.at(triangle.c2).d, d3 = pivots.at(triangle.c3).d;
  const int u_min = std::ceil(min3(u1, u2, u3));
  const int u_max = std::floor(max3(u1, u2, u3)); 
  const int v_min = std::ceil(min3(v1, v2, v3));
  const int v_max = std::floor(max3(v1, v2, v3)); 

  points.reserve((u_max - u_min + 1) * (v_max - v_min + 1));

  int num_valid = 0;
  for(int v = v_min; v <= v_max; ++v) {
    for(int u = u_min; u <= u_max; ++u) {
      if(detectRegion(cv::Point2i(u, v), cv::Point2i(u1, v1), cv::Point2i(u2, v2), cv::Point2i(u3, v3))) {
        points.push_back(cv::Point2i(u, v));
        num_valid++;
      }
    }
  }
  points.resize(num_valid);
  return points;
}

inline bool StereoMatcher::detectRegion(const cv::Point2f& point, const cv::Point2f& pa, const cv::Point2f& pb, const cv::Point2f& pc)
{
  if(pa == pb || pa == pc || pb == pc) {
    return false;
  }
  const Eigen::Vector3f vec_ab(pb.x - pa.x, pb.y - pa.y, 0),
                        vec_ac(pc.x - pa.x, pc.y - pa.y, 0), 
                        vec_ap(point.x - pa.x, point.y - pa.y, 0),
                        vec_bp(point.x - pb.x, point.y - pb.y, 0),
                        vec_cp(point.x - pc.x, point.y - pc.y, 0);
  const Eigen::Vector3f &vec_ba = - vec_ab,
                        &vec_bc = vec_ac - vec_ab,
                        &vec_ca = - vec_ac,
                        &vec_cb = - vec_bc;

  return ((vec_ab.cross(vec_ac)).dot(vec_ab.cross(vec_ap)) >= 0) && 
         ((vec_bc.cross(vec_ba)).dot(vec_bc.cross(vec_bp)) >= 0) &&
         ((vec_ca.cross(vec_cb)).dot(vec_ca.cross(vec_cp)) >= 0);
}


void StereoMatcher::drawDisparityAndvariance(const cv::Mat& gray, const cv::Mat& disparity, const cv::Mat& variance)
{
  assert(disparity.size() == variance.size() && disparity.size() == gray.size());
  const int& height = disparity.rows;
  const int& width = disparity.cols;
  cv::Mat show_disparity(disparity.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat show_variance(variance.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Scalar scalar;
  for(int v = 0; v < height; ++v) {
    for(int u = 0; u < width; ++u) {
      float disp = imref_f32(disparity, v, u);
      float var = imref_f32(variance, v, u);
      if(disp == 0 || var == 0) {
        continue;
      }
      scalar = dispshader(disp);
      cv::circle(show_disparity, Point2i(u, v), 0, scalar, -1);
      scalar = covshader(var);
      cv::circle(show_variance, Point2i(u, v), 0, scalar, -1);
    }
  }
  imshow("show_disparity", show_disparity);
  imshow("show_variance", show_variance);
  imshow("show gray image", gray);
  while(1) {
    if(waitKey(0) == 'q') 
      break;
  }
}

void StereoMatcher::drawLidarProjections(const cv::Mat& img, const std::vector<StereoMatcher::Pivot>& pivots)
{
  cv::Mat show = img.clone();
  cvtColor(show, show, cv::COLOR_GRAY2BGR);
  
  for(int i = 0; i < pivots.size(); ++i) {
    cv::Point2i pi(std::round(pivots.at(i).u), std::round(pivots.at(i).v));
    // cv::Scalar scalar = shader(pivots.at(i).d, disp_min, disp_max);
    cv::Scalar scalar = dispshader(pivots.at(i).d);
    cv::circle(show, pi, 0, scalar, -1);
  }
  imshow("show", show);
  while(1) {
    if(waitKey(0) == 'q') 
      break;
  }
}

void StereoMatcher::drawDelaunay(const cv::Mat& img, 
                                     const std::vector<Pivot>& pivots, 
                                     const std::vector<Triangle>& triangles)
{
  cv::Mat show = img.clone();
  if(img.channels() == 1)
    cvtColor(show, show, cv::COLOR_GRAY2BGR);
  for(int i = 0; i < triangles.size(); ++i) {
    
    const Triangle& triangle_plane_i = triangles.at(i);
    Point2i p1(pivots.at(triangle_plane_i.c1).u, pivots.at(triangle_plane_i.c1).v);
    Point2i p2(pivots.at(triangle_plane_i.c2).u, pivots.at(triangle_plane_i.c2).v);
    Point2i p3(pivots.at(triangle_plane_i.c3).u, pivots.at(triangle_plane_i.c3).v);
    cv::line(show, p1, p2, cv::Scalar(255, 0, 0), 1);
    cv::line(show, p3, p2, cv::Scalar(255, 0, 0), 1);
    cv::line(show, p1, p3, cv::Scalar(255, 0, 0), 1);
    
  }
  imshow("show", show);
  while(1) {
    if(waitKey(0) == 'q') 
      break;
  
  }
}

void StereoMatcher::drawMatching(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Pivot>& pivots)
{
  const int &height = img1.rows;
  const int &width = img1.cols;
  for(int i = 0; i < pivots.size(); ++i) {
    cv::Mat concat;
    cv::vconcat(img1, img2, concat);
    if(concat.channels() == 1)
      cvtColor(concat, concat, cv::COLOR_GRAY2BGR);
    const Pivot &pivot_i = pivots.at(i);
    Point2f p1(pivot_i.u, pivot_i.v), p2(pivot_i.u - pivot_i.d, pivot_i.v + height);
    cv::line(concat, p1, p2, Scalar(0, 255, 0), 1);
    imshow("vconcat", concat);
    while(1) {
      if(waitKey(0) == 'q') break;
    }
  }
}

void StereoMatcher::drawPivots(const cv::Mat& img1, const cv::Mat& img2, const std::vector<Pivot>& pivots)
{
  const int height = img1.rows;
  cv::Mat show;
  cv::vconcat(img1, img2, show);
  cvtColor(show, show, cv::COLOR_GRAY2BGR);
  for(int i = 0; i < pivots.size(); ++i) {
    Point2i point_left(pivots.at(i).u, pivots.at(i).v), point_right(pivots.at(i).u - pivots.at(i).d, pivots.at(i).v + height);
    cv::Scalar scalar = dispshader(pivots.at(i).d);
    cv::circle(show, point_left, 0, scalar, -1);
    cv::circle(show, point_right, 0, scalar, -1);
  }
  imshow("pivots", show);
  while(1) {
    if(waitKey(0) == 'q') break;
  }
}


}