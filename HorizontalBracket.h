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
	// ����ͼƬ��ת��Ϊ��ֵͼƬ,������0��ǰ����1
	Mat ConvertToBinImage(Mat image, int MaxValue, int BinaryType);
public:
	// ���ˮƽ����ͼ������ź�����
	void detectHorizontalBracketAndNum(Mat image, OCR &ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos);
	
	// ��ⴹֱ����ͼ�������
	vector<pair<int, int> > detectVerticalBracket(Mat image);
	// ��ⴹֱ����ͼ�������ڵ�����
	vector<string> detectVerticalBracketNum(Mat image, OCR &ocr);
	// ��������λ�õ�ӳ�䣬scale:��ʼ������������col_start������col_start_num����λ���ȵ����ش�Сunit_pixels,�Լ���λ����unit
	vector<pair<double, double>> convertToActualBracket(vector<double>scale, vector<pair<int, int> > bracket_pos);
	// ���C������
	void detectCNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos);

	void detectWNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int>>& bracket_pos);

	HorizontalBracket();
	~HorizontalBracket();
};

