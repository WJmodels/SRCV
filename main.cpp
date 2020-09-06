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
#include <fstream>
#include <sstream>
#include "SVM.h"
#include "CImg.h"
#include "Adaboost.h"
#include "CutImg.h"
#include "OCR.h"
#include "HorizontalBracket.h"
#include "KNN.h"
using namespace cimg_library;
using namespace cv;
using namespace std;


int main() {

	
	OCR ocr;

	HorizontalBracket horizontal;
	CutImg cut;
	ofstream fout("brackets.txt");

	while (true) {
		cout << "1.SVM 2.AdaBoost 3.KNN 4.horizontal integration bracket extraction 5.vertical integration bracket extraction 6.1H chemical shift extraction 7.13C chemical shift extraction  8.knn test" << endl;
		char c;
		cin >> c;
		if (c == '1') {
			SVM svm;
			svm.SVMTrain("trainset");
			svm.SVMTest(string("testset"));
		}
		else if (c == '2') {
			Adaboost adaboost;
			adaboost.AdaboostTrain("trainset");
			adaboost.AdaboostTest("testset");
		}
		else if (c == '3') {
			KNN knn;
			knn.KNNTrain("trainset");
			knn.KNNTest("testset");
		}
		else if (c == '4') {
			// 对水平括号处理
			for (int i = 0; i <=29 ; i++) {
				//string filepath = "TT035/CC -  (" + to_string(i) + ").png";
				string filepath = "examples-input/" + to_string(i) + "H.png";

				 //string filepath = "TT035/CC -  (2).png";
				Mat image = imread(filepath, IMREAD_COLOR);
				vector<string> numbers;
				vector<pair<int, int> > bracket_pos;
				// 检测刻度尺，得到起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
				vector<double> scale = cut.ScaleDetect(image, ocr);
				// 检测水平括号的数字以及对应的括号像素位置
				//cout << "$$$$$$$$$$$$$" << endl;
				horizontal.detectHorizontalBracketAndNum(image, ocr, numbers, bracket_pos);
				cout << "integral---------------------------------------------------" << endl;
				fout << "integral---------------------------------------------------" << endl;
				// 将括号的像素位置转换为实际坐标
				cout << filepath << endl;
				fout << filepath << endl;
			//	for (int k = 0; k < scale.size(); k++) {
			//	cout << "scale+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << endl;
			//		cout << "( " << scale[k] << " )" << endl;
			//		cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << endl;

			//	}
			//	for (int k = 0; k < bracket_pos.size() ; k++) {
			//		cout << "bracket+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << endl;
			//		cout << "( " << bracket_pos[k].first << ", " << bracket_pos[k].second<< " )" << endl;
			//		cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << endl;
					
			//	}


				vector<pair<double, double>> actual_bracket = horizontal.convertToActualBracket(scale, bracket_pos);
				for (int k = 0; k < actual_bracket.size() && k < numbers.size(); k++) {
					cout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
					fout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
				}
				time_t time1 = time(0);
				cout << "time1 = " << time1 << endl;
				fout << "---------------------------------------------------" << endl << endl;
				cout << "---------------------------------------------------" << endl << endl;
			}
		}
		else if (c == '5') {
			// 对垂直括号处理
			for (int i = 1; i <= 88; i += 2) {
				string filepath = "Ol9b/ol9b (" + to_string(i) + ").png";
				cout << filepath << endl;
				// string filepath = "TT035/CC -  (2).png";
				Mat image = imread(filepath, IMREAD_COLOR);
				// 检测刻度尺，得到起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
				vector<double> scale = cut.ScaleDetect(image, ocr);
				// 检测垂直括号的位置
				vector<pair<int, int> > bracket_pos = horizontal.detectVerticalBracket(image);
				// 得到垂直括号数字
				vector<string> numbers = horizontal.detectVerticalBracketNum(image, ocr);
				fout << "---------------------------------------------------" << endl;
				fout << filepath << endl;
				// 将括号的像素位置转换为实际坐标
				vector<pair<double, double>> actual_bracket = horizontal.convertToActualBracket(scale, bracket_pos);
				for (int k = 0; k < actual_bracket.size() && k < numbers.size(); k++) {
					cout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
					fout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
				}
				time_t time1 = time(0);
				//cout << "time1 = " << time1 << endl;
				fout << "---------------------------------------------------" << endl;
				cout << "---------------------------------------------------" << endl << endl;
			}
		}
		else if (c == '6') {
			// 对H谱上方数据处理
			for (int i = 0; i <= 29; i++) {
				fout << endl;
				cout << endl;
				fout << "shift--------------------------------------------------" << endl;
				cout << "shift---------------------------------------------------" << endl;
				string filepath = "examples-input/" + to_string(i) + "H.png";
				cout << filepath << endl;
				fout << filepath << endl;
				// string filepath = "TT035/CC -  (2).png";
				Mat image = imread(filepath, IMREAD_COLOR);
				
				vector<string> numbers;
				vector<pair<int, int> > bracket_pos;
				// 检测刻度尺，得到起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
				vector<double> scale = cut.ScaleDetect(image, ocr);
				// 检测水平括号的数字以及对应的括号像素位置
				horizontal.detectCNTRNum(image, ocr, numbers, bracket_pos);
				for (int i = 0; i < numbers.size(); i++) {
					cout << numbers[i] << endl;
					fout << numbers[i] << endl;
				}
				// 将括号的像素位置转换为实际坐标
				
				vector<pair<double, double>> actual_bracket = horizontal.convertToActualBracket(scale, bracket_pos);
				for (int k = 0; k < actual_bracket.size() && k < numbers.size(); k++) {
					cout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
					fout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
				}
				time_t time1 = time(0);
				cout << "time1 = " << time1 << endl;
				fout << "---------------------------------------------------" << endl << endl;
				cout << "---------------------------------------------------" << endl << endl;
			}
		}
		else if (c == '7') {
			// 对C谱数据处理
			for (int i = 0; i <= 29; i++) {
				fout << endl;
				cout << endl;
				string filepath = "examples-input/" + to_string(i) + "C.png";
				cout << filepath << endl;
				// string filepath = "TT035/CC -  (2).png";
				Mat image = imread(filepath, IMREAD_COLOR);

				vector<string> numbers;
				vector<pair<int, int> > bracket_pos;
				// 检测刻度尺，得到起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
				vector<double> scale = cut.ScaleDetect(image, ocr);
				// 检测水平括号的数字以及对应的括号像素位置
				horizontal.detectWNTRNum(image, ocr, numbers, bracket_pos);
				fout << "shift---------------------------------------------------" << endl;
				fout << filepath << endl;
				for (int i = 0; i < numbers.size(); i++) {
				//	cout << numbers[i] << endl;
					fout << numbers[i] << endl;
				}
				// 将括号的像素位置转换为实际坐标
				//fout << filepath << endl;
				vector<pair<double, double>> actual_bracket = horizontal.convertToActualBracket(scale, bracket_pos);
				for (int k = 0; k < actual_bracket.size() && k < numbers.size(); k++) {
					cout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
					fout << "( " << numbers[k] << ", " << actual_bracket[k].first << ", " << actual_bracket[k].second << " )" << endl;
				}
				time_t time1 = time(0);
				cout << "time1 = " << time1 << endl;
				fout << "---------------------------------------------------" << endl << endl;
				cout << "---------------------------------------------------" << endl << endl;
			}
		}
		else if (c == '8') {
		KNN knn;
		for (int i = 1; i <= 5; i++)
		{	
			String str1;
			String str2;
			
			str1 = "trainset" + std::to_string(i);
			str2 = "testset" + std::to_string(i);
			//cout << str1;
			knn.KNNTrain(str1);
			knn.KNNTest(str2);
		}
		
		}
	}

	
}