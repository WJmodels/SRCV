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
#include "SVM.h"
#include "OCR.h"
using namespace cimg_library;
using namespace cv;
using namespace std;

class CutImg
{
public:


	SVM svm;
	// 输入图片，转换为二值图片,背景是0，前景是1
	Mat ConvertToBinImage(Mat image, int MaxValue, int BinaryType);


	int icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
	// 输入原图像，得到标尺信息
	vector<double> ScaleDetect(Mat image, OCR& ocr);

	vector<double> PeakDetect(Mat image, OCR& ocr);

	cv::Scalar icvprGetRandomColor();


	void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);

	void CutNum(Mat image, string save_filename);




	CutImg();
	~CutImg();
};

