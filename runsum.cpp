#include <iostream>
#include <math.h>

#include "runsum.h"
#include "utils.h"

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//------------------Running sum----------------------//
Mat runSum(const Mat img, int tx, int ty)
{
	// set up variables for data access
	int height = img.rows;
	int width  = img.cols;

	Mat rs_img(height, width, CV_64FC1);

	if(tx>0)
	{
		// Running sum in the x direction
		for(int i = 0; i < height; i++)
		{
			double rs = 0.0;
			//int is = i*step;
			for(int j=0; j<width; j++)
			{
				rs += img.at<double>(i,j);
				rs_img.at<double>(i,j) = rs;
			}
		}

		for(int times = 0; times< tx-1; times++)
		{
			for(int i = 0; i < height; i++)
			{	
				double rs = 0.0;
				//int is = i*step;
				for(int j=0; j<width; j++)
				{
					rs += rs_img.at<double>(i,j);
					rs_img.at<double>(i,j) = rs;
				}
			}
		}

		// Running sum in the y direction
		for(int times = 0; times < ty; times++)
		{
			for(int j = 0; j < width; j++)
			{
				double rs = 0.0;
				for(int i=0; i<height; i++)
				{
					rs += rs_img.at<double>(i,j);
					rs_img.at<double>(i,j) = rs;
				}
			}		
		}
	
	}
	else if(ty>0)
	{
		// Running sum in the y direction
		for(int j = 0; j < width; j++)
		{
			double rs = 0.0;
			for(int i=0; i<height; i++)
			{
				rs += img.at<double>(i,j);
				rs_img.at<double>(i,j) = rs;
			}
		}

		for(int times = 0; times < ty-1; times++)
		{
			for(int j = 0; j < width; j++)
			{
				double rs = 0.0;
				for(int i=0; i<height; i++)
				{
					rs += rs_img.at<double>(i,j); 
					rs_img.at<double>(i,j) = rs;
				}
			}
		}
	}
	else
	{
		error("At least one of tx and ty should be larger than zero!");
	}
	// Return the running sum of B-spline coefficients
	return rs_img;
}


// -----------------Prefiltering via B-spline kernels----------------//
// prefilter, simple
void prefilter(const Mat rs_img, Mat& oo, Mat& oe, Mat& eo, Mat& ee)
{
	
	// Prefiltering with phi(x)
	double phi2oo[3][3] = {{1.0/64,6.0/64,1.0/64},{6.0/64,36.0/64,6.0/64},
			       {1.0/64,6.0/64,1.0/64}};

	double phi2oe[2][3] = {{1.0/16,6.0/16,1.0/16},{1.0/16,6.0/16,1.0/16}};

	double phi2eo[3][2] = {{1.0/16,1.0/16},{6.0/16,6.0/16},{1.0/16,1.0/16}};

	double phi2ee[2][2] = {{1.0/4,1.0/4},{1.0/4,1.0/4}};
	
	
	Mat Phi2oo(3, 3, CV_64FC1, phi2oo);
    Mat Phi2oe(2, 3, CV_64FC1, phi2oe);
	Mat Phi2eo(3, 2, CV_64FC1, phi2eo);
	Mat Phi2ee(2, 2, CV_64FC1, phi2ee);
	
	//  0    0
	//cvFilter2D(rs_img, oo, &Phi2oo,cvPoint(1,1));
	filter2D(rs_img, oo, CV_64F, Phi2oo, Point(1,1)); 
	//  0   1/2
	//cvFilter2D(rs_img, oe, &Phi2oe,cvPoint(1,0));
	filter2D(rs_img, oe, CV_64F, Phi2oe, Point(1,0));
	// 1/2   0
	//cvFilter2D(rs_img, eo, &Phi2eo,cvPoint(0,1));
	filter2D(rs_img, eo, CV_64F, Phi2eo, Point(0,1));
	// 1/2  1/2
	//cvFilter2D(rs_img, ee, &Phi2ee,cvPoint(0,0));
	filter2D(rs_img, ee, CV_64F, Phi2ee, Point(0,0));
}

// prefilter complex
void prefilter(const Mat rs_img1, const Mat rs_img2, const Mat rs_img3,
	       Mat& xxo, Mat& xxe, Mat& yyo, Mat& yye, Mat& xyo, Mat& xye)
{
	
	double phi24oo[5][3] = {{1.0/3072,6.0/3072,1.0/3072},
				 {76.0/3072,456.0/3072,76.0/3072},
				 {230.0/3072,1380.0/3072,230.0/3072},
				 {76.0/3072,456.0/3072,76.0/3072},
				 {1.0/3072,6.0/3072,1.0/3072}};

	double phi24ee[4][2] = {{1.0/48,1.0/48},{11.0/48,11.0/48},
				 {11.0/48,11.0/48},{1.0/48,1.0/48}};

	double phi42oo[3][5] = {{1.0/3072,76.0/3072,230.0/3072,76.0/3072,1.0/3072},
				 {6.0/3072,456.0/3072,1380.0/3072,456.0/3072,6.0/3072},
				 {1.0/3072,76.0/3072,230.0/3072,76.0/3072,1.0/3072}};

	double phi42ee[2][4] = {{1.0/48,11.0/48,11.0/48,1.0/48},
				 {1.0/48,11.0/48,11.0/48,1.0/48}};

	double phi33oo[3][3] = {{1.0/36,4.0/36,1.0/36},
				 {4.0/36,16.0/36,4.0/36},
				 {1.0/36,4.0/36,1.0/36}};

	double phi33ee[4][4] = {{1.0/2304,23.0/2304,23.0/2304,1.0/2304},
				 {23.0/2304,529.0/2304,529.0/2304,23.0/2304},
				 {23.0/2304,529.0/2304,529.0/2304,23.0/2304},
				 {1.0/2304,23.0/2304,23.0/2304,1.0/2304}};
	
	Mat Phi24oo(5, 3, CV_64FC1, phi24oo);
	Mat Phi24ee(4, 2, CV_64FC1, phi24ee); 
	Mat Phi42oo(3, 5, CV_64FC1, phi42oo);
	Mat Phi42ee(2, 4, CV_64FC1, phi42ee);
    Mat Phi33oo(3, 3, CV_64FC1, phi33oo);
	Mat Phi33ee(3, 3, CV_64FC1, phi33ee);
	
	//Lxx
	filter2D(rs_img1, xxo, CV_64F, Phi24oo);
	filter2D(rs_img1, xxe, CV_64F, Phi24ee, Point(0,1));
	
	//Lyy
	filter2D(rs_img2, yyo, CV_64F, Phi42oo);
	filter2D(rs_img2, yye, CV_64F, Phi42ee, Point(1,0));

	//Lxy
	filter2D(rs_img3, xyo, CV_64F, Phi33oo);
	filter2D(rs_img3, xye, CV_64F, Phi33ee, Point(1,1));
}

Mat directRunSum(const Mat img)
{
	int rows = img.rows;
	int cols = img.cols;
	Mat drs_im_0(img.size(), img.type());
	Mat drs_im_1(img.size(), img.type());
	Mat drs_im_2(img.size(), img.type());
	Mat drs_im_3(img.size(), img.type());

	drs_im_0.at<double>(0,0) = img.at<double>(0,0);
	drs_im_1.at<double>(0,0) = sqrt(2.0) * drs_im_0.at<double>(0,0);
	drs_im_2.at<double>(0,0) = drs_im_1.at<double>(0,0);
	drs_im_3.at<double>(0,0) = sqrt(2.0) * drs_im_2.at<double>(0,0);

	// for the first column
	for(int i = 1; i < rows; i++)
	{
		drs_im_0.at<double>(i,0) = img.at<double>(i,0) + drs_im_0.at<double>(i-1,0);
		drs_im_1.at<double>(i,0) = sqrt(2.0) * drs_im_0.at<double>(i,0);
		drs_im_2.at<double>(i,0) = drs_im_1.at<double>(i,0);
		drs_im_3.at<double>(i,0) = sqrt(2.0) * drs_im_2.at<double>(i,0);
	}

	// for the first row
	for(int j = 1; j < cols; j++)
	{
		drs_im_0.at<double>(0,j) = img.at<double>(0,j);
		drs_im_1.at<double>(0,j) = sqrt(2.0) * drs_im_0.at<double>(0,j);
		drs_im_2.at<double>(0,j) = drs_im_1.at<double>(0,j) + drs_im_2.at<double>(0,j-1);
	}

	// for the remaining
	for(int j = 1; j < cols-1; j++)
	{
		for(int i = 1; i < rows-1; i++)
		{
			drs_im_0.at<double>(i,j) = img.at<double>(i,j) + drs_im_0.at<double>(i-1,j);
			drs_im_1.at<double>(i,j) = sqrt(2.0) * drs_im_0.at<double>(i,j) + drs_im_1.at<double>(i-1,j-1);
			drs_im_2.at<double>(i,j) = drs_im_1.at<double>(i,j) + drs_im_2.at<double>(i,j-1);
			drs_im_3.at<double>(i,j) = sqrt(2.0) * drs_im_2.at<double>(i,j) + drs_im_3.at<double>(i+1,j-1);
		}
		drs_im_3.at<double>(0,j) = sqrt(2.0) * drs_im_2.at<double>(0,j) + drs_im_3.at<double>(1,j-1);
	}

	// for the last row
	for(int j = 1; j < cols-1; j++)
	{
		drs_im_0.at<double>(rows-1,j) = img.at<double>(rows-1,j) + drs_im_0.at<double>(rows-2,j);
		drs_im_1.at<double>(rows-1,j) = sqrt(2.0) * drs_im_0.at<double>(rows-1,j) + drs_im_1.at<double>(rows-2,j-1);
		drs_im_2.at<double>(rows-1,j) = drs_im_1.at<double>(rows-1,j) + drs_im_2.at<double>(rows-1,j-1);
	}

	// for the last column
	for(int i = 1; i < rows-1; i++)
	{
		drs_im_0.at<double>(i,cols-1) = img.at<double>(i,cols-1) + drs_im_0.at<double>(i-1, cols-1);
		drs_im_1.at<double>(i,cols-1) = sqrt(2.0) * drs_im_0.at<double>(i,cols-1) + drs_im_1.at<double>(i-1,cols-2);
		drs_im_2.at<double>(i,cols-1) = drs_im_1.at<double>(i,cols-1) + drs_im_2.at<double>(i,cols-2);
		drs_im_3.at<double>(i,cols-1) = sqrt(2.0) * drs_im_2.at<double>(i,cols-1) + drs_im_3.at<double>(i+1,cols-2);
	}

	// for the last vertex
	drs_im_0.at<double>(rows-1,cols-1) = img.at<double>(rows-1,cols-1) + drs_im_0.at<double>(rows-2,cols-1);
	drs_im_1.at<double>(rows-1,cols-1) = sqrt(2.0) * drs_im_0.at<double>(rows-1,cols-1) + drs_im_1.at<double>(rows-2,cols-2);
	drs_im_2.at<double>(rows-1,cols-1) = drs_im_1.at<double>(rows-1,cols-1) + drs_im_2.at<double>(rows-1,cols-2);
	drs_im_3.at<double>(rows-1,cols-1) = sqrt(2.0) * drs_im_2.at<double>(rows-1,cols-1);

	return drs_im_3;
}

double Ffun(double delta_x, double delta_y)
{
	double Fjk_0, Fjk_x, Fjk_y, Fjk_yx, Fjk_xy;

	Fjk_0 = 1.0 - delta_x*delta_x - delta_y*delta_y;

	if (delta_x >= -3/2.0 && delta_x < -1/2.0)
		Fjk_x = (delta_x+1/2.0)*(delta_x+1/2.0);
	else if (delta_x >= -1/2.0 && delta_x <= 1/2.0)
		Fjk_x = 0;
	else if (delta_x > 1/2.0 && delta_x <= 3/2.0)
		Fjk_x = (delta_x-1/2.0)*(delta_x-1/2.0);
	else
		Fjk_x = 0;

	if (delta_y >= -3/2.0 && delta_y < -1/2.0)
		Fjk_y = (delta_y+1/2.0)*(delta_y+1/2.0);
	else if (delta_y >= -1/2.0 && delta_y <= 1/2.0)
		Fjk_y = 0;
	else if (delta_y > 1/2.0 && delta_y <= 3/2.0)
		Fjk_y = (delta_y-1/2.0)*(delta_y-1/2.0);
	else
		Fjk_y = 0;

	double delta_xy_diff = delta_y - delta_x;
	if (delta_xy_diff >= -2 && delta_xy_diff < -1)
		Fjk_yx = (delta_xy_diff+1)*(delta_xy_diff+1)/2.0;
	else if (delta_xy_diff >= -1 && delta_xy_diff <= 1)
		Fjk_yx = 0;
	else if (delta_xy_diff >= 1 && delta_xy_diff <= 2)
		Fjk_yx = (delta_xy_diff-1)*(delta_xy_diff-1)/2.0;
	else
		Fjk_yx = 0;

	double delta_xy_sum = delta_x + delta_y;
	if (delta_xy_sum >= -2 && delta_xy_sum < -1)
		Fjk_xy = (delta_xy_sum+1)*(delta_xy_sum+1)/2.0;
	else if (delta_xy_sum >= -1 && delta_xy_sum <= 1)
		Fjk_xy = 0;
	else if (delta_xy_sum > 1 && delta_xy_sum <= 2)
		Fjk_xy = (delta_xy_sum-1)*(delta_xy_sum-1)/2.0;
	else
		Fjk_xy = 0;

	double interp_value = Fjk_0 + Fjk_x + Fjk_y + Fjk_xy + Fjk_yx;
	return interp_value;
}


void calcParams(double a, struct Params *param)
{
	param->tau1 = (sqrt(2.0) * a - sqrt(2.0)) / (2*sqrt(2.0));
	param->tau2 = (a + sqrt(2.0) * a + a - 3 * sqrt(2.0)) / (2*sqrt(2.0));
	double a1 = a/sqrt(2.0);
	param->xi = {0,0, a,0, a1,a1, a+a1,a1,
		0,a, a,a, a1,a+a1, a+a1,a+a1,
		-a1,a1, a-a1,a1, 0,a1+a1, a,a1+a1,
		-a1,a+a1, a-a1,a+a1, 0,a+a1+a1, a,a+a1+a1};
	param->w = 1.0/(a*a*a*a);
}

double gbSum(const Mat gb, int r, int c, double *dx, double *dy)
{
	double res = 0;
	
	res = gb.at<double>(r+dx[0],c+dy[0])-gb.at<double>(r+dx[1],c+dy[1])-gb.at<double>(r+dx[2],c+dy[2])+gb.at<double>(r+dx[3],c+dy[3])-gb.at<double>(r+dx[4],c+dy[4])+gb.at<double>(r+dx[5],c+dy[5])+gb.at<double>(r+dx[6],c+dy[6])-gb.at<double>(r+dx[7],c+dy[7])-gb.at<double>(r+dx[8],c+dy[8])+gb.at<double>(r+dx[9],c+dy[9])+gb.at<double>(r+dx[10],c+dy[10])-gb.at<double>(r+dx[11],c+dy[11])+gb.at<double>(r+dx[12],c+dy[12])-gb.at<double>(r+dx[13],c+dy[13])-gb.at<double>(r+dx[14],c+dy[14])+gb.at<double>(r+dx[15],c+dy[15]);

	return res;
}

