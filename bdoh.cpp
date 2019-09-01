#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "bdoh.h"

using namespace cv;

// Constructor simple
BDoH::BDoH(const Mat& pre_oo, const Mat& pre_oe, const Mat& pre_eo, const Mat& pre_ee, 
	std::vector<Ipoint> &ipts, const char type, const int octaves, const int scales, 
	const double sigma0, const double sigmaN, const double thres, int borderR, int borderL)
	: ipts(ipts)
{
	// Initialize pointers
	pre_oo.copyTo(oo);
	pre_oe.copyTo(oe);
	pre_eo.copyTo(eo);
	pre_ee.copyTo(ee);
	responseMap = NULL;

	// Initialize variables
	this->type = type;
	this->octaves = octaves;
	this->scales = scales;
	this->sigma0 = sigma0;
	this->sigmaN = sigmaN;
	this->thres = thres;
	this->borderR = borderR;
	this->borderL = borderL;
	this->width = pre_oo.cols-borderR-borderL;
	this->height = pre_oo.rows-borderR-borderL;
}

// Constructor complex
BDoH::BDoH(const Mat& pre_xxo, const Mat& pre_xxe, const Mat& pre_yyo, const Mat& pre_yye, const Mat& pre_xyo, const Mat& pre_xye,
	std::vector<Ipoint> &ipts, const char type, const int octaves, const int scales, 
	const double sigma0, const double sigmaN, const double thres, int borderR, int borderL)
	: ipts(ipts)
{
	// Initialize pointers
	pre_xxo.copyTo(xxo);
	pre_xxe.copyTo(xxe);
	pre_yyo.copyTo(yyo);
	pre_yye.copyTo(yye);
	pre_xyo.copyTo(xyo);
	pre_xye.copyTo(xye);
	responseMap = NULL;

	// Initialize variables
	this->type = type;
	this->octaves = octaves;
	this->scales = scales;
	this->sigma0 = sigma0;
	this->sigmaN = sigmaN;
	this->thres = thres;
	this->borderR = borderR;
	this->borderL = borderL;
	this->width = pre_xxo.cols-borderR-borderL;
	this->height = pre_xxo.rows-borderR-borderL;
}

// Destructor
BDoH::~BDoH()
{
	if(responseMap)
	{
		for(int i = 0; i < octaves*scales+2; i++)
		{
			delete []responseMap[i];
		}
		delete []responseMap;
	}
}

// Detect keypoints
void BDoH::getIpoints()
{	
	if(type=='s')
		buildResponseMapSimple();
	else if(type=='c')
		buildResponseMapComplex();
	else 
		error("Detector not supported");

	int samplingStep = 1;
	for(int i = 0; i< octaves; i++)
	{
		int i_width  = (int)ceil(double(width)/samplingStep); 
		int i_height = (int)ceil(double(height)/samplingStep);
		for(int j = 0; j < scales; j++)
		{
			int index = i*octaves + j;	
			double scale = pow(2.0,i)*sigma0*pow(2.0,double(j+1)/scales);
			double *b = responseMap[index];
			double *m = responseMap[index+1];
			double *t = responseMap[index+2];

			for (int r = 0; r < i_height; r++)
			{
				for (int c = 0; c < i_width; c++)
				{
					if (isExtremum(r, c, t, m, b, 8, i_height, i_width))
					{
						interpolateExtremum(r, c, t, m, b, i_height, i_width, samplingStep, scale);
					}
				}
			}	
		}
		samplingStep = 2*samplingStep;
	}

}

// Response Map Simple
void BDoH::buildResponseMapSimple()
{
	// Temp variables
	int i_width = width; 
	int i_height = height;
	
	// Build response map
	responseMap = new double* [octaves*(scales+2)];
	double sigma;

	// First ocatve
	for( int j = 0; j < scales+2; j++) 
	{
		responseMap[j] = new double[i_width*i_height];
		sigma = sigma0*pow(2.0,double(j)/scales);
		sigma = sqrt(sigma*sigma - sigmaN*sigmaN);
		buildResponseLayerSimple(responseMap[j], sigma, 1);
	}
	
	// second- octaves
	for(int i = 1; i < octaves; i++)
	{
		for(int j = 0; j < 2; j++) 
		{
			responseMap[i*(scales+2)+j] = halveSize(responseMap[(i-1)*(scales+2)+j+2], i_height, i_width);
		}

		i_width  = (int)ceil(double(i_width)/2); 
		i_height = (int)ceil(double(i_height)/2);

		for(int j = 2; j < scales+2; j++) 
		{
			responseMap[i*(scales+2)+j] = new double[(i_width)*(i_height)];
			sigma = pow(2.0,i)*sigma0*pow(2.0,double(j)/scales);
			sigma = sqrt(sigma*sigma - sigmaN*sigmaN);
			buildResponseLayerSimple(responseMap[i*(scales+2)+j], sigma, (int)pow(2.0,i));
		}
	}
}

// Response Map Complex
void BDoH::buildResponseMapComplex()
{
	// Temp variables
	int i_width = width; 
	int i_height = height;
	
	// Build response map
	responseMap = new double* [octaves*(scales+2)];
	double sigma;

	// First ocatve
	for( int j = 0; j < scales+2; j++) 
	{
		responseMap[j] = new double[i_width*i_height];
		sigma = sigma0*pow(2.0,double(j)/scales);
		sigma = sqrt(sigma*sigma - sigmaN*sigmaN);
		buildResponseLayerComplex(responseMap[j], sigma, 1);
	}
	
	// second- octaves
	for(int i = 1; i < octaves; i++)
	{
		for(int j = 0; j < 2; j++) 
		{
			responseMap[i*(scales+2)+j] = halveSize(responseMap[(i-1)*(scales+2)+j+2], i_height, i_width);
		}

		i_width  = (int)ceil(double(i_width)/2); 
		i_height = (int)ceil(double(i_height)/2);

		for(int j = 2; j < scales+2; j++) 
		{
			responseMap[i*(scales+2)+j] = new double[(i_width)*(i_height)];
			sigma = pow(2.0,i)*sigma0*pow(2.0,double(j)/scales);
			sigma = sqrt(sigma*sigma - sigmaN*sigmaN);
			buildResponseLayerComplex(responseMap[i*(scales+2)+j], sigma, (int)pow(2.0,i));
		}
	}
}

// Calculate DoH responses for supplied layer, simple
void BDoH::buildResponseLayerSimple(double *response, double sigma, int samplingStep)
{
	int s0 = (int)(sigma*sqrt(12.0)+0.5);
	int s1 = (int)(sigma*sqrt(6.0)+0.5);
	int s2 = (int)(sigma*2+0.5);
	int step = ee.step/sizeof(double);
	double inverse1 = 1/pow((double)s0,2)/pow((double)s2,2);
	double inverse2 = pow(0.7979,4)/pow(0.9679,2)/pow((double)s1,4);
	// Choose the right prefiltered image
	// Shift of the kernel
	double sh2  = (double)(3*s2-1)/2;
	double sh0  = (double)(s0-1)/2;   
	int sh2i = (3*s2-1)/2;
	int sh0i = (s0-1)/2;
	int sh1i = (2*s1-1)/2;

	int index0[2], index1[3], index2[4];
	for(int i = 0; i < 3; i++)
	{
		index1[i] = sh1i - i*s1;
	}

	for(int i = 0; i < 4; i++)
	{
		index2[i] = sh2i - i*s2;
	}

	for(int i = 0; i < 2; i++)
	{
		index0[i] = sh0i - i*s0;
	}

	double *dataxy = (double*)ee.data;
	double *dataxx = NULL;
	double *datayy = NULL;
	if(sh2 == sh2i && sh0 == sh0i)
	{
		dataxx = (double*)oo.data;
		datayy = (double*)oo.data;
	}
	else if(sh2 == sh2i && sh0 != sh0i)
	{
		dataxx = (double*)oe.data;
		datayy = (double*)eo.data;
	}
	else if(sh2 != sh2i && sh0 == sh0i)
	{
		dataxx = (double*)eo.data;
		datayy = (double*)oe.data;
	}
	else if(sh2 != sh2i && sh0 != sh0i)
	{
		dataxx = (double*)ee.data;
		datayy = (double*)ee.data;
	}

	int count = 0;
	for(int r = borderL; r < height + borderL; r += samplingStep)
	{
		int pos = r*step;
		int pos20 = pos + index2[0]*step;
		int pos21 = pos + index2[1]*step;
		int pos22 = pos + index2[2]*step;
		int pos23 = pos + index2[3]*step;
		int pos00 = pos + index0[0]*step;
		int pos01 = pos + index0[1]*step;
		int pos10 = pos + index1[0]*step;
		int pos11 = pos + index1[1]*step;
		int pos12 = pos + index1[2]*step;

		for(int c = borderL; c < width + borderL; c += samplingStep)
		{
			double Lxx = dataxx[pos00+index2[0]+c] - 3*dataxx[pos00+index2[1]+c] + 3*dataxx[pos00+index2[2]+c] - dataxx[pos00+index2[3]+c]
					   - dataxx[pos01+index2[0]+c] + 3*dataxx[pos01+index2[1]+c] - 3*dataxx[pos01+index2[2]+c] + dataxx[pos01+index2[3]+c];

			double Lyy = datayy[pos20+index0[0]+c] - 3*datayy[pos21+index0[0]+c] + 3*datayy[pos22+index0[0]+c] - datayy[pos23+index0[0]+c]
				       - datayy[pos20+index0[1]+c] + 3*datayy[pos21+index0[1]+c] - 3*datayy[pos22+index0[1]+c] + datayy[pos23+index0[1]+c];

			double Lxy =   dataxy[pos10+index1[0]+c] - 2*dataxy[pos10+index1[1]+c] + dataxy[pos10+index1[2]+c]
				       - 2*dataxy[pos11+index1[0]+c] + 4*dataxy[pos11+index1[1]+c] - 2*dataxy[pos11+index1[2]+c]
				       +   dataxy[pos12+index1[0]+c] - 2*dataxy[pos12+index1[1]+c] + dataxy[pos12+index1[2]+c];
			
			response[count++] = fabs(inverse1*Lxx*Lyy - inverse2*Lxy*Lxy);
		}
	}
}

// Calculate DoH responses for supplied layer, complex
void BDoH::buildResponseLayerComplex(double *response, double sigma, int samplingStep)
{
	int s2 = (int)(sigma*2+0.5);
	int i_height = (int)ceil(double(height)/samplingStep);
	int i_width  = (int)ceil(double(width)/samplingStep);
	double inverse = 1/pow((double)s2,8);
	int step = xxe.step/sizeof(double);
	
	// Choose the right prefiltered image
	// shift of the kernel
	double sh2 = (double)(3*s2-1)/2; 
	double sh1 = (double)(3*s2-2)/2;
	double sh0 = (double)(3*s2-3)/2;
	int sh2i = (3*s2-1)/2;
	int sh1i = (3*s2-2)/2;
	int sh0i = (3*s2-3)/2;
	
	double *dataxx = NULL;
	double *datayy = NULL;
	double *dataxy = NULL;
	if(sh2 == sh2i)
	{
		dataxx = (double*)xxo.data;
		datayy = (double*)yyo.data;
		dataxy = (double*)xye.data;
	}
	else
	{
		dataxx = (double*)xxe.data;
		datayy = (double*)yye.data;
		dataxy = (double*)xyo.data;
	}

	// Filter index
	int index2[4], index1[4], index0[4];

	for(int i = 0; i < 4; i++)
	{
		index2[i] = sh2i - i*s2;
	}
	for(int i = 0; i < 4; i++)
	{
		index1[i] = sh1i - i*s2;
	}
	for(int i = 0; i < 4; i++)
	{
		index0[i] = sh0i - i*s2;
	}

	int count = 0;
	for(int r = borderL; r < height + borderL; r += samplingStep)
	{
		int pos = r*step;
		int pos20 = pos + index2[0]*step;
		int pos21 = pos + index2[1]*step;
		int pos22 = pos + index2[2]*step;
		int pos23 = pos + index2[3]*step;
		int pos10 = pos + index1[0]*step;
		int pos11 = pos + index1[1]*step;
		int pos12 = pos + index1[2]*step;
		int pos13 = pos + index1[3]*step;
		int pos00 = pos + index0[0]*step;
		int pos01 = pos + index0[1]*step;
		int pos02 = pos + index0[2]*step;
		int pos03 = pos + index0[3]*step;

		for(int c = borderL; c < width + borderL; c += samplingStep)
		{
			double Lxx = dataxx[pos00+index2[0]+c] - 3*dataxx[pos00+index2[1]+c] + 3*dataxx[pos00+index2[2]+c] - dataxx[pos00+index2[3]+c]
					   - 3*dataxx[pos01+index2[0]+c] + 9*dataxx[pos01+index2[1]+c] - 9*dataxx[pos01+index2[2]+c] + 3*dataxx[pos01+index2[3]+c]
					   + 3*dataxx[pos02+index2[0]+c] - 9*dataxx[pos02+index2[1]+c] + 9*dataxx[pos02+index2[2]+c] - 3*dataxx[pos02+index2[3]+c]
					   - dataxx[pos03+index2[0]+c] + 3*dataxx[pos03+index2[1]+c] - 3*dataxx[pos03+index2[2]+c] + dataxx[pos03+index2[3]+c];
			
			double Lyy = datayy[pos20+index0[0]+c] - 3*datayy[pos20+index0[1]+c] + 3*datayy[pos20+index0[2]+c] - datayy[pos20+index0[3]+c]
					   - 3*datayy[pos21+index0[0]+c] + 9*datayy[pos21+index0[1]+c] - 9*datayy[pos21+index0[2]+c] + 3*datayy[pos21+index0[3]+c]
					   + 3*datayy[pos22+index0[0]+c] - 9*datayy[pos22+index0[1]+c] + 9*datayy[pos22+index0[2]+c] - 3*datayy[pos22+index0[3]+c]
					   - datayy[pos23+index0[0]+c] + 3*datayy[pos23+index0[1]+c] - 3*datayy[pos23+index0[2]+c] + datayy[pos23+index0[3]+c];

			double Lxy = dataxy[pos10+index1[0]+c] - 3*dataxy[pos10+index1[1]+c] + 3*dataxy[pos10+index1[2]+c] - dataxy[pos10+index1[3]+c]
					   - 3*dataxy[pos11+index1[0]+c] + 9*dataxy[pos11+index1[1]+c] - 9*dataxy[pos11+index1[2]+c] + 3*dataxy[pos11+index1[3]+c]
					   + 3*dataxy[pos12+index1[0]+c] - 9*dataxy[pos12+index1[1]+c] + 9*dataxy[pos12+index1[2]+c] - 3*dataxy[pos12+index1[3]+c]
					   - dataxy[pos13+index1[0]+c] + 3*dataxy[pos13+index1[1]+c] - 3*dataxy[pos13+index1[2]+c] + dataxy[pos13+index1[3]+c];
			
			response[count++] = fabs(inverse*(Lxx*Lyy - Lxy*Lxy));
		}
	}
}


// Non-maxmima suppression
int BDoH::isExtremum(int r, int c, double *t, double *m, double *b, int border, int i_height, int i_width)
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
void BDoH::interpolateExtremum(int r, int c, double *t, double *m, double *b, int i_height, int i_width, int samplingStep, double sigma)
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
		ipt.scale = sigma*pow(2.0, xhat.at<double>(2)/scales);
		ipts.push_back(ipt);
	}

}
