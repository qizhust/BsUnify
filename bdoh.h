#ifndef _BDoH_H
#define _BDoH_H

#include <opencv2/imgproc/imgproc_c.h>
#include <vector>
#include <iostream>
#include "runsum.h"
#include "ipoint.h"

class BDoH{

public:

	// Constructor simple
	BDoH(const Mat& pre_oo,
		 const Mat& pre_oe,
		 const Mat& pre_eo,
		 const Mat& pre_ee,
		 std::vector<Ipoint> &ipts,
		 const char type = 's',
         const int octaves = 4, 
         const int scales = 2, 
         const double sigma0 = 1.6, 
		 const double sigmaN = 0.5,
         const double thres = 0.08,
		 const int borderR = 0, 
		 const int borderL = 0);

	// Constructor complex
	BDoH(const Mat& pre_xxo,
		 const Mat& pre_xxe,
		 const Mat& pre_yyo,
		 const Mat& pre_yye,
		 const Mat& pre_xyo,
		 const Mat& pre_xye,
		 std::vector<Ipoint> &ipts,
		 const char type = 'c',
		 const int octaves = 4, 
		 const int scales = 2, 
		 const double sigma0 = 1.6, 
		 const double sigmaN = 0.5,
		 const double thres = 0.08,
		 const int borderR = 0, 
		 const int borderL = 0);


	// Destructor
	~BDoH();

	// Find the image features and write into vector of features
	void getIpoints();

private:

	//---------------- Private Functions -----------------//

	// Build map of blob responses
	void buildResponseMapSimple();
	void buildResponseMapComplex();

	// Calculate DoH responses for supplied layer
	void buildResponseLayerSimple(double *response, double sigma, int samplingStep);
	void buildResponseLayerComplex(double *response, double sigma, int samplingStep);

	// Local extrema detection
	int isExtremum(int r, int c, double *t, double *m, double *b, int border, int i_height, int i_width); //max
	void interpolateExtremum(int r, int c, double *t, double *m, double *b, int i_height, int i_width, int samplingStep, double sigma); //interpolation
	
	//---------------- Private Variables -----------------//
    
	// Pointer to the running sumed image, complex
	Mat xxo, xxe, yyo, yye, xyo, xye;

	// Pointer to the running sumed image, simple
	Mat oo, oe, eo, ee;

	// Image borders
	int borderR, borderL;

	// Image width and height
	int width, height;
	
	// Number of Octaves
	int octaves;

	// Number of scales in each octave
	int scales;

	// Sigma for initial smoothing
	double sigma0;

	// Sigma for nominal smoothing
	double sigmaN;

	// Blob response threshold 
	double thres;

	// Reference to vector of features passed from outside
	std::vector<Ipoint> &ipts;

	// Response map
	double **responseMap;

	// Detector type
	int type;
};	

#endif
