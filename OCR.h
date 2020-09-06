#pragma once
#include "SVM.h"
#include "KNN.h"


class OCR
{
public:
	string DetectNum(vector<Mat> inMats);
	KNN detector;
	OCR();
	~OCR();
};

