#include <vector>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>

#include "bdoh.h"
#include "blap.h"
#include "runsum.h"
#include "utils.h"

using namespace std;
using namespace cv;

int main (int argc, char **argv)
{
  // Number of analysed octaves (default 4)
  int octaves = 4;
  // Detector type: {'s': bdoh-simple, 'c': bdoh-complex, '2': blap-2, '4': blap-4}
  char type = 'c'; 
  // Number of scale levels per octave
  int scales = 2; // param for bdoh only
  // Sigma for initial smoothing
  double sigma0 = 1.6;
  // Assumed blur for input image
  double sigmaN = 0.5;
  // Blob response treshold
  double thres = 0.08;
  // verbose output
  bool verb = true;

  // Print command line help
  if (argc==1) {
    cerr << "./bunif -i img.pgm -o img.bunif [options]\n"
         << "  blob response threshold:            -thres 800\n"
         << "  number of octaves:                  -oc 4\n"
         << "  number of scales per octave:        -ss 3\n"
         << "  sigma for initial smoothing:        -s0 1.6\n"
         << "  version                             -t  s/c/2/4\n"
         << "  verbose output:                     -vb 1/0\n";
    return(0);
  }

  // Read the arguments
  int arg = 0;
  string fn = "out.bunif";

  // border
  int border  = (int)(2*sigma0*pow(2.0,(double)(scales*octaves+1)/(double)scales)+0.5);
  int borderR = (3*border-1)/2+2;
  int borderL = 2-(borderR-2-3*border);
  Mat im, img;

  while (++arg < argc){
	  if (! strcmp(argv[arg], "-i")){
		  im = imread(argv[++arg]);
		  // Convert the image to single channel 64f
		  img = getGray(im, borderR, borderL);
	  }

	  if (! strcmp(argv[arg], "-o"))
		  fn = argv[++arg];

	  if (! strcmp(argv[arg], "-thres"))
		  thres = atof(argv[++arg])/10000;

	  if (! strcmp(argv[arg], "-oc"))
		  octaves = atoi(argv[++arg]);

	  if (! strcmp(argv[arg], "-sl"))
		  scales = atoi(argv[++arg]);

	  if (! strcmp(argv[arg], "-s0"))
		  sigma0 = atof(argv[++arg]);

	  if (! strcmp(argv[arg], "-t"))
		  type = *argv[++arg];

	  if (! strcmp(argv[arg], "-vb"))
		  verb = atoi(argv[++arg])==1 ? true:false;
  }          

  //Start keypoint detection
  clock_t start = clock();

  if(verb)
  {
	  if (type == 's')
		cout << "B-DoH-S:Finding keypoints...\n";
	  else if (type == 'c')
		cout << "B-DoH-C:Finding keypoints...\n"; 
	  else if (type == '2')
	  	cout << "B-Lap-2: Finding keypoints...\n";
	  else if (type == '4')
	    cout << "B-Lap-4: Finding keypoints...\n";
  }

  //Keypoints
  vector<Ipoint> ipts;

  if(type == 's')
  {
	  // Running sum of B-spline coefficients
	  Mat rs_img = runSum(img, 1, 1);

	  // Prefiltering
	  Mat pre_oo(rs_img.rows, rs_img.cols, rs_img.type());
	  Mat pre_oe(rs_img.rows, rs_img.cols, rs_img.type());
	  Mat pre_eo(rs_img.rows, rs_img.cols, rs_img.type());
	  Mat pre_ee(rs_img.rows, rs_img.cols, rs_img.type());
	  prefilter(rs_img, pre_oo, pre_oe, pre_eo, pre_ee);

	  // Extract interest points with the simple detector B-DoH-S
	  BDoH bh(pre_oo, pre_oe, pre_eo, pre_ee,
		      ipts,
		      type,
		      octaves,
		      scales,
		      sigma0,
		      sigmaN,
		      thres,
			  borderR,
			  borderL);

	  // Extract interest points and store in vector ipts
      bh.getIpoints();
  }
  else if(type == 'c')
  {
	  Mat rs_img  = runSum(img, 1, 1);
	  Mat rs_img1 = runSum(rs_img, 0, 2);
	  Mat rs_img2 = runSum(rs_img, 2, 0);
	  Mat rs_img3 = runSum(rs_img, 1, 1);
	  Mat pre_xxo(rs_img1.size(), rs_img1.type());
	  Mat pre_xxe(rs_img1.size(), rs_img1.type());
	  Mat pre_yyo(rs_img2.size(), rs_img2.type());
	  Mat pre_yye(rs_img2.size(), rs_img2.type());
	  Mat pre_xyo(rs_img3.size(), rs_img3.type());
	  Mat pre_xye(rs_img3.size(), rs_img3.type());
	  prefilter(rs_img1, rs_img2, rs_img3, pre_xxo, pre_xxe, pre_yyo, pre_yye, pre_xyo, pre_xye);

	  // Extract interest points with the complex detector B-DoH-C
	  BDoH bh(pre_xxo, pre_xxe, pre_yyo, pre_yye, pre_xyo, pre_xye,
		      ipts,
		      type,
		      octaves,
		      scales,
		      sigma0,
		      sigmaN,
		      thres,
			  borderR,
			  borderL);

	  // Extract interest points and store in vector ipts
	  bh.getIpoints();
  }
  else if(type == '2')
  {
	  // Separable filtering 
	  Mat pre_int = runSum(img, 1, 1);

	  // Extract interest points with the simple detector B-DoH-S
	  BLap bl(pre_int, ipts, type-'0', octaves, sigma0, sigmaN, thres,
		borderR, borderL);

	  // Extract interest points and store in vector ipts
      bl.getIpoints();

  }
  else if(type == '4')
  {
	  Mat direct_pre_int = directRunSum(img);
	  // Extract interest points with the complex detector B-DoH-C
	  BLap bl(direct_pre_int, ipts, type-'0', octaves, sigma0, sigmaN, 
	  	thres, borderR, borderL);

	  // Extract interest points and store in vector ipts
	  bl.getIpoints();
  }
  else
  {
	  error("Detector not supported");
  }

  clock_t end = clock();

  if(verb)
  {
	  cout<< ipts.size() << " keypoints found" << endl;
	  cout<< "It took: " << 1000*(float)(end - start) / CLOCKS_PER_SEC  << " mseconds" << endl;
  }

  // Save features
  saveFeatures(fn, ipts);

  return 0;
}
