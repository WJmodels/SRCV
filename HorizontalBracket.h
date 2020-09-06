#pragma once
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <string>  
#include <list>  
#include <vector>  
#include <map>  
#include "SVM.h"
#include "CImg.h"
#include "Adaboost.h"
#include "KNN.h"
#include "CutImg.h"
#include <sstream>
#include "OCR.h"

class HorizontalBracket
{
private:
	void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);
	cv::Scalar icvprGetRandomColor();
	int icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
	// 输入图片，转换为二值图片,背景是0，前景是1
	Mat ConvertToBinImage(Mat image, int MaxValue, int BinaryType);
public:
	// 检测水平括号图像的括号和数字
	void detectHorizontalBracketAndNum(Mat image, OCR &ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos);
	
	// 检测垂直括号图像的括号
	vector<pair<int, int> > detectVerticalBracket(Mat image);
	// 检测垂直括号图像括号内的数字
	vector<string> detectVerticalBracketNum(Mat image, OCR &ocr);
	// 进行括号位置的映射，scale:起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
	vector<pair<double, double>> convertToActualBracket(vector<double>scale, vector<pair<int, int> > bracket_pos);
	// 检测C谱数字
	void detectCNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos);

	void detectWNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int>>& bracket_pos);

	HorizontalBracket();
	~HorizontalBracket();
};

