#include <opencv2/opencv.hpp>
#ifndef _RUNSUM_H
#define _RUNSUM_H	

using namespace cv;

// Running sum of B-spline coefficients
Mat runSum(const Mat img, int tx, int ty);

// Prefiltering via B-spline kernels
// prefilter simple
void prefilter(const Mat rs_img, Mat& oo, Mat& oe, Mat &eo, Mat &ee);  
// prefilter complex
void prefilter(const Mat rs_img1, const Mat rs_img2, const Mat rs_img3, Mat& xx1, Mat& xx2, Mat& yy1, Mat& yy2, Mat& xy1, Mat& xy2);

// Running sum of directional B-spline
Mat directRunSum(const Mat img);

// Calculate interpolated value of directional B-spline
double Ffun(double delta_x, double delta_y);

void calcParams(double a, struct Params *param);

double gbSum(const Mat gb, int r, int c, double *dx, double *dy);
#endif
