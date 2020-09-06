#pragma once

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp> 
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <string>  
#include <list>  
#include <vector>  
#include <map>  
#include "CImg.h"
#include "FileOp.h"
using namespace cimg_library;
using namespace cv;
using namespace std;

class SVM
{
public:
	cv::Ptr<cv::ml::SVM> svm;
	FileOp fileop;
	void SVMTrain(string trainpath);
	int SVMTest(Mat inMat);
	void SVMTest(string testpath);
	SVM();
	~SVM();
};

