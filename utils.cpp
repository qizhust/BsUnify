#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>

#include "utils.h"

using namespace std;
using namespace cv;

//-------------------------------------------------------

// Display error message and terminate program
void error(const char *msg) 
{
  cout << "\nError: " << msg;
  getchar();
  exit(0);
}

// Convert image to single channel 64F
Mat getGray(const Mat img, int borderR, int borderL)
{
  // Check we have been supplied a non-null img pointer
  if (!(&img)) error("Unable to create grayscale image.  No image supplied");

  Mat gray8, gray64, gray64_pad;

  gray64 = Mat(img.rows, img.cols, CV_64FC1);

  // Padding with zeros
  gray64_pad = Mat(img.rows+borderR+borderL, img.cols+borderR+borderL, CV_64FC1);

  if(img.channels() == 1)
    img.copyTo(gray8);
  else {
    gray8 = Mat(img.rows, img.cols, img.depth());
    cvtColor(img, gray8, CV_BGR2GRAY);
  }

  gray8.convertTo(gray64, CV_64F, 1.0 / 255.0, 0);
  copyMakeBorder(gray64, gray64_pad, borderL, borderR, borderL, borderR, BORDER_CONSTANT, 0);
  return gray64_pad;
}

double* halveSize(double *src, int height, int width)
{
	int i_height = (int)ceil(double(height)/2);
	int i_width  = (int)ceil(double(width)/2); 
	double *dst = new double[i_height*i_width];

	int count = 0;
	for (int i=0; i<height; i+=2)
	{
		double *line = src+i*width;
		for (int j=0; j<width; j+=2)
		{
			dst[count++] = line[j];
		}
	}

	return dst;
}

void saveFeatures(string sFileName, const vector< Ipoint >& ipts) 
{
  ofstream ipfile(sFileName.c_str());

  double sc;
  unsigned count = (unsigned)ipts.size();

  ipfile << 1 << endl << count << endl;

  for (unsigned n=0; n<ipts.size(); n++){
    // circular regions with diameter 2 x scale
    sc = 2 * ipts[n].scale; sc*=sc;
    ipfile  << ipts[n].x 
            << " " << ipts[n].y
            << " " << 1.0/sc 
            << " " << 0.0     
            << " " << 1.0/sc << endl;
  }

  ipfile.close();
}
