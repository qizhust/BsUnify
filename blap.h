#ifndef _BLap_H
#define _BLap_H

#include <opencv2/imgproc/imgproc_c.h>
#include <vector>
#include <iostream>
#include "runsum.h"
#include "ipoint.h"
#include "utils.h"

class BLap{

public:

	// Constructor blap4
	BLap(const Mat& pre_int,
		std::vector<Ipoint> &ipts,
		const int directs = 4,
		const int octaves = 8, 
		const double sigma0 = 1.6, 
		const double sigmaN = 0.5,
		const double thres = 0.08,
		const int borderR = 0, 
		const int borderL = 0);

	// Destructor
	~BLap();

	// Find the image features and write into vector of features
	void getIpoints();

private:

	//---------------- Private Functions -----------------//

	// Build map of blob responses
	void buildResponseMap();

	// Calculate bspline responses for supplied layer
	void buildResponseLayerBox(double *filtered_b, double *filtered, double *response, double sigma);
	void buildResponseLayerBox(double *filtered, double sigma);
	void buildResponseLayerOct(double *filtered_b, double *filtered, double *response, double sigma, double Fval, struct Params *param);
	void buildResponseLayerOct(double *filtered, double sigma, double *Fval, struct Params *param);

	// Local extrema detection
	int isExtremum(int r, int c, double *t, double *m, double *b, int border, int i_height, int i_width); //max
	void interpolateExtremum(int r, int c, double *t, double *m, double *b, int i_height, int i_width, int samplingStep, double sigma); //interpolation
	
	//---------------- Private Variables -----------------//
    
	// Pointer to the (directional) running sumed image, blap2/blap4
	Mat rs_im;

	// Image borders
	int borderR, borderL;

	// Image width and height
	int width, height;
	
	// Number of Octaves
	int octaves;

	// Sigma for initial smoothing
	double sigma0;

	// Sigma for nominal smoothing
	double sigmaN;

	// Blob response threshold 
	double thres;

	// Reference to vector of features passed from outside
	std::vector<Ipoint> &ipts;

	// Response map
	double **filteredMap;
	double **responseMap;

	// Detector type
	int directs;
};	

#endif
