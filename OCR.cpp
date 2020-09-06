#include "OCR.h"

string OCR::DetectNum(vector<Mat> inMats) {
	string result;
	for (int i = 0; i < inMats.size(); i++) {
		if (inMats[i].rows <= 7 && inMats[i].cols <= 8) {
			result += ".";
		}
		else {
			result += to_string(detector.KNNTest(inMats[i]));
		}
	}
	return result;
}


OCR::OCR()
{	
	detector.KNNTrain("trainset");
	//cout << "OCR³õÊ¼»¯" << endl;
}


OCR::~OCR()
{
}
