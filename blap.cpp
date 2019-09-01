#include <vector>
#include <algorithm>
#include <iostream>
#define NDEBUG
#include <cassert>
#include "utils.h"
#include "blap.h"

using namespace std;
using namespace cv;

// Constructor blap4
BLap::BLap(const Mat& pre_int,
	std::vector<Ipoint> &ipts, const int directs, const int octaves,
	const double sigma0, const double sigmaN, const double thres, 
	int borderR, int borderL)
	: ipts(ipts)
{
	// Initialize pointers
	pre_int.copyTo(rs_im);
	responseMap = nullptr;
	filteredMap = nullptr;

	// Initialize variables
	this->directs = directs;
	this->octaves = octaves;
	this->sigma0 = sigma0;
	this->sigmaN = sigmaN;
	this->thres = thres;
	this->borderR = borderR;
	this->borderL = borderL;
	this->width = pre_int.cols-borderR-borderL;
	this->height = pre_int.rows-borderR-borderL;
}

// Destructor
BLap::~BLap()
{
	if(filteredMap)
	{
		for(int i = 0; i < octaves; i++)
		{
			delete []filteredMap[i];
		}
		delete []filteredMap;
	}

	if(responseMap)
	{
		for(int i = 0; i < octaves-2; i++)
		{
			delete []responseMap[i];
		}
		delete []responseMap;
	}
}

// Detect keypoints
void BLap::getIpoints()
{	
	if (directs == 2 || directs == 4)
	{
		buildResponseMap();
	}
	else 
		error("Detector not supported");

	int samplingStep = 1;
	for(int i = 0; i < octaves-3; i++)
	{
		double scale = pow(2.0,double(i)/2)*sigma0;
		double *b = responseMap[i];
		double *m = responseMap[i+1];
		double *t = responseMap[i+2];
	
		for(int r = 0; r < height; r++)
		{
			for(int c = 0; c < width; c++)
			{
				if (isExtremum(r, c, t, m, b, 8, height, width))
					interpolateExtremum(r, c, t, m, b, height, width, samplingStep, scale);
			}
		}
	}
}

// Response Map 
void BLap::buildResponseMap()
{
	// Build response map
	filteredMap = new double* [octaves];
	responseMap = new double* [octaves-1];
	double sigma, Fval[1];
	Params param4;

	// Get ocatves and calculate DoB response map
	for(int j = 0; j < octaves; j++) 
	{
		filteredMap[j] = new double[width*height];
		sigma = sigma0*pow(2.0,double(j)/2);
		if (j > 0)
		{
			responseMap[j-1] = new double[width*height];
			if (directs == 2)
				buildResponseLayerBox(filteredMap[j-1], filteredMap[j], responseMap[j-1], sigma);
			else if (directs == 4)
				buildResponseLayerOct(filteredMap[j-1], filteredMap[j], responseMap[j-1], sigma, Fval[0], &param4);
		}
		else
		{
			if (directs == 2)
				buildResponseLayerBox(filteredMap[j], sigma);
			else if (directs == 4)
				buildResponseLayerOct(filteredMap[j], sigma, Fval, &param4);
		}
	}
}

// Calculate bspline responses for supplied layer, blap2
void BLap::buildResponseLayerBox(double *filtered, double sigma)
{
	double a = sigma * sqrt(24/double(directs));	
	int count = 0;
	int border = round(a/2);
	assert(border < min(borderL, borderR));
	double area = 4*border*border;
	for(int r = borderL; r < height + borderL; r++)
	{
		for(int c = borderL; c < width + borderL; c++)
		{	
			if (r-borderL<border || r+border>height+borderL-1 || c-borderL<border || c+border>width+borderL-1)
				filtered[count] = 0;
			else
				filtered[count] = (rs_im.at<double>(r+border,c+border) - rs_im.at<double>(r+border,c-border) - rs_im.at<double>(r-border,c+border) + rs_im.at<double>(r-border,c-border))/area;
			count++;
		}
	}
}

void BLap::buildResponseLayerBox(double *filtered_b, double *filtered, double *response, double sigma)
{
	double a = sigma * sqrt(24/double(directs));
	int count = 0;
	int border = round(a/2);
	assert(border < min(borderL, borderR));
	double area = 4*border*border;
	for(int r = borderL; r < height + borderL; r++)
	{
		for(int c = borderL; c < width + borderL; c++)
		{
			if (r-borderL<border || r+border>height+borderL-1 || c-borderL<border || c+border>width+borderL-1)
			{
				filtered[count] = 0;
				response[count] = 0;
			}
			else
			{
				filtered[count] = (rs_im.at<double>(r+border,c+border) - rs_im.at<double>(r+border,c-border) - rs_im.at<double>(r-border,c+border) + rs_im.at<double>(r-border,c-border))/area;
				response[count] = abs(filtered_b[count] - filtered[count]);
			}
			count++;
		}
	}
}

// Calculate bspline responses for supplied layer, blap4
void BLap::buildResponseLayerOct(double *filtered, double sigma, double *Fval, struct Params *param)
{
	double a_val = sigma * sqrt(24/double(directs));
	int border = round(a_val/2);
	assert(border < min(borderL, borderR));
	int count = 0;

	calcParams(a_val, param);

	Mat xall_1(7, 2, CV_64FC1, param->xall1);
	Mat xall_2(7, 2, CV_64FC1, param->xall2);
	Mat xall_3(7, 2, CV_64FC1, param->xall3);
	Mat xall_4(7, 2, CV_64FC1, param->xall4);
	Mat xall(112, 2, CV_64FC1);

	double add_x[16], add_y[16], dx[16], dy[16];
	double Fvals[16];
	for(int k = 0; k < 16; k++)
	{
		add_x[k] = param->tau1 - param->xi[k*2];
		dx[k] = round(add_x[k]);
		add_y[k] = param->tau2 - param->xi[k*2+1];
		dy[k] = round(add_y[k]);
		double delta_x = round(add_x[k]) - add_x[k];
		double delta_y = round(add_y[k]) - add_y[k];

		Mat roi(xall(cv::Rect(0,k*7,2,7)));

		if (delta_x > delta_y)
		{
			if (delta_x > -delta_y)
				xall_1.copyTo(roi);
			else
				xall_2.copyTo(roi);
		}
		else
		{
			if (delta_x > -delta_y)
				xall_3.copyTo(roi);
			else
				xall_4.copyTo(roi);
		}

		Fvals[k] = 0;

		for (int p = 0; p < 7; p++)
		{
			double delta_x_in = xall.at<double>(k*7+p,0) + delta_x;
			double delta_y_in = xall.at<double>(k*7+p,1) + delta_y;
			Fvals[k] += Ffun(delta_x_in, delta_y_in);
		}
	}

	Fval[0] = Fvals[0];

	for(int r = borderL; r < height + borderL; r++)
	{
		for(int c = borderL; c < width + borderL; c++)
		{
			if (r-borderL<border || r+border>height+borderL-1 || c-borderL<border || c+border>width+borderL-1)
				filtered[count] = 0;
			else
			{
				filtered[count] = Fval[0]*param->w*gbSum(rs_im, r, c, dx, dy);
			}
			count++;
		}
	}
}

void BLap::buildResponseLayerOct(double *filtered_b, double *filtered, double *response, double sigma, double Fval, struct Params *param)
{
	double a_val = sigma * sqrt(24/double(directs));
	int border = round(a_val/2);
	assert(border < min(borderL, borderR));
	int count = 0;

	calcParams(a_val, param);
	
	double add_x[16], add_y[16];
	for(int k = 0; k < 16; k++)
	{
		add_x[k] = round(param->tau1 - param->xi[k*2]);
		add_y[k] = round(param->tau2 - param->xi[k*2+1]);
	}

	for(int r = borderL; r < height + borderL; r++)
	{
		for(int c = borderL; c < width + borderL; c++)
		{
			if (r-borderL<border || r+border>height+borderL-1 || c-borderL<border || c+border>width+borderL-1)
			{
				filtered[count] = 0;
				response[count] = 0;
			}
			else
			{
				filtered[count] = Fval*param->w*gbSum(rs_im, r, c, add_x, add_y);
				response[count] = abs(filtered_b[count]- filtered[count]);
			}
			count++;
		}
	}
}

// Non-maxmima suppression
int BLap::isExtremum(int r, int c, double *t, double *m, double *b, int border, int i_height, int i_width)
{
	if (r <= border || r >= i_height - 1 - border || c <= border || c >= i_width - 1 - border)
		return 0;
	
	int offset = r*i_width + c;
	double *pt_t = t + offset;
	double *pt_m = m + offset;
	double *pt_b = b + offset;

	if (pt_m[0] <= 0.8*thres)
		return 0;
	
	for (int rr = -1; rr <=1; rr++)
	{
		for (int cc = -1; cc <=1; cc++)
		{
			int shift = rr*i_width + cc;
			if(
				*(pt_t+shift) >= pt_m[0] ||
				((rr != 0 || cc != 0) && *(pt_m+shift) >= pt_m[0]) ||
				*(pt_b+shift) >= pt_m[0]
			  )
			return 0;
		}
	}
	return 1;
}


// Interpolation
void BLap::interpolateExtremum(int r, int c, double *t, double *m, double *b, int i_height, int i_width, int samplingStep, double sigma)
{
	double Dx=0, Dy=0, Ds=0, Dxx=0, Dyy=0, Dss=0, Dxy=0, Dxs=0, Dys=0;
	int dr = 0, dc = 0; int offset;
	const int max_iter = 5;
	Mat A, d, xhat;
	
	for(int i = 0; i < max_iter; i++)
	{
		r += dr;
		c += dc;
		offset = r*i_width + c;
		double *pt_t = t + offset;
		double *pt_m = m + offset;
		double *pt_b = b + offset;

		Dx = 0.5*(*(pt_m + 1) - *(pt_m - 1));
		Dy = 0.5*(*(pt_m + i_width) - *(pt_m - i_width));
		Ds = 0.5*(*pt_t - *pt_b);

		d = Mat(3, 1, CV_64FC1);
		d.at<double>(0) = -Dx;
		d.at<double>(1) = -Dy;
		d.at<double>(2) = -Ds;

		Dxx = *(pt_m+1) + *(pt_m-1) - 2* *pt_m;
		Dyy = *(pt_m+i_width) + *(pt_m-i_width) - 2* *pt_m;
		Dss = *pt_t + *pt_b - 2* *pt_m;

		Dxy = 0.25*(*(pt_m + i_width + 1) + *(pt_m - i_width - 1) - *(pt_m + i_width - 1) - *(pt_m - i_width + 1)); 
		Dxs = 0.25*(*(pt_t + 1) + *(pt_b - 1) - *(pt_t - 1) - *(pt_b + 1)); 
		Dys = 0.25*(*(pt_t + i_width) + *(pt_b - i_width) - *(pt_t - i_width) - *(pt_b + i_width)); 

		A = Mat(3, 3, CV_64FC1);
		A.at<double>(0,0) = Dxx;
		A.at<double>(0,1) = Dxy;
		A.at<double>(0,2) = Dxs;
		A.at<double>(1,0) = Dxy;
		A.at<double>(1,1) = Dyy;
		A.at<double>(1,2) = Dys;
		A.at<double>(2,0) = Dxs;
		A.at<double>(2,1) = Dys;
		A.at<double>(2,2) = Dss;

		xhat = Mat(3, 1, CV_64FC1);
		solve(A, d, xhat, CV_SVD);

		dc= ((xhat.at<double>(0) > 0.6 && c < i_width - 2) ?  1 : 0)
		  + ((xhat.at<double>(0)  < -0.6 && c > 1  ) ? -1 : 0 );
		dr= ((xhat.at<double>(1) > 0.6 && r < i_height - 2) ?  1 : 0)
		  + ((xhat.at<double>(1)  < -0.6 && r > 1  ) ? -1 : 0);

		if( dr == 0 && dc == 0 ) break;
	}
	
	double val = *(m + offset) + 0.5*(Dx*xhat.at<double>(0) + Dy*xhat.at<double>(1) + Ds*xhat.at<double>(2));
	
	if(fabs(val)>thres && fabs(xhat.at<double>(0))<1.5 && fabs(xhat.at<double>(1))<1.5 && fabs(xhat.at<double>(2))<1)
	{
		Ipoint ipt;
		ipt.x = (c + xhat.at<double>(0))*samplingStep;
		ipt.y = (r + xhat.at<double>(1))*samplingStep;
		ipt.scale = sigma*pow(2.0, xhat.at<double>(2)/2);
		ipts.push_back(ipt);
	}
}
