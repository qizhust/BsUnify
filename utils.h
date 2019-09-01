#ifndef _UTILS_H
#define _UTILS_H

#include <opencv2/imgproc/imgproc_c.h>

#include "ipoint.h"

#include <vector>

using namespace cv;

struct Params
{
	double tau1;
	double tau2;
    std::vector<double> xi;
	double w;
	double wi[16] = {1, -1, -1, 1, -1, 1, 1, -1,
		-1, 1, 1, -1, 1, -1, -1, 1};
	double xall1[7][2] = {{0,0}, {1,0}, {0,1}, {-1,1}, {-1,0}, {-1,-1}, {0,-1}};
	double xall2[7][2] = {{0,0}, {0,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}};
	double xall3[7][2] = {{0,0}, {0,1}, {-1,0}, {-1,-1}, {0,-1}, {1,-1}, {1,0}};
	double xall4[7][2] = {{0,0}, {-1,0}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}};
};

// Display error message and terminate program
void error(const char *msg);

// Convert image to single channel 64F
Mat getGray(const Mat img, int borderR, int borderL);

// Subsampling
double* halveSize(double *src, int height, int width);

// Save the features to file
void saveFeatures(std::string filename, const std::vector<Ipoint> &ipts);

#endif
