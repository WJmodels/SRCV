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
using namespace cimg_library;
using namespace cv;
using namespace std;

class FileOp
{
public:
	void getFiles(string path, vector<string>& files);
	int getLabel(string filename);
	void getTrainSet(string trainsetpath, Mat& trainingImages, vector<int>& trainingLabels);
	FileOp();
	~FileOp();
};

