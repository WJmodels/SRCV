#include "Adaboost.h"


void Adaboost::AdaboostTrain(string trainpath) {
	//获取训练数据
	Mat trainingData;
	Mat trainingImages;
	vector<int> trainingLabels;
	fileop.getTrainSet(trainpath, trainingImages, trainingLabels);
	Mat(trainingImages).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	cout << "Adaboost开始训练！" << endl;
	for (int k = 0; k < 10; k++) {
		cout << "Adaboost开始训练！"<<k<< endl;
		vector<int> temp_trainingLabels;
		for (int i = 0; i < trainingLabels.size(); i++) {
			if (trainingLabels[i] == k) {
				temp_trainingLabels.push_back(1);
			}
			else {
				temp_trainingLabels.push_back(-1);
			}
		}
		Mat classes;
		Mat(temp_trainingLabels).copyTo(classes);
		adaboost[k]->train(trainingData, cv::ml::ROW_SAMPLE, classes);

	}

	//adaboost->train(trainingData, cv::ml::ROW_SAMPLE, classes);
	//adaboost->save("svm.xml");
	cout << "Adaboost train done！" << endl;
}

void Adaboost::AdaboostTest(string testpath) {
	vector<string> filenames;
	fileop.getFiles(testpath, filenames);
	string save_path = "adaboosttestset/";
	int wholesize = filenames.size();
	int correctsize = 0;
	for (int i = 0; i < filenames.size(); i++) {
		int num;
		for (int k = 0; k < filenames[i].length(); k++) {
			if (filenames[i][k] == '_') {
				num = filenames[i][k + 1] - '0';
				break;
			}
		}
		Mat inMat = imread(filenames[i]);
		resize(inMat, inMat, Size(12, 18));
		Mat p = inMat.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		cout << "Adaboost开始测试！" << endl;
		for (int k = 0; k < 10; k++) {
			int result = (int)adaboost[k]->predict(p);
			if (result == 1) {
				string savename = save_path + to_string(i + 1) + "_" + to_string(k) + ".png";
				imwrite(savename, inMat);
				result = k;
				if (result == num) {
					correctsize++;
				}
				else {
					cout << "错误识别的图片：" << filenames[i] << endl;
				}

				break;
			}
		}
	}
	cout << "正确率为：" << double(correctsize) / double(wholesize) << endl;


}


Adaboost::Adaboost()
{
	for (int i = 0; i < 10; i++) {
		adaboost[i] = cv::ml::Boost::create();
		adaboost[i]->setBoostType(cv::ml::Boost::DISCRETE);
		adaboost[i]->setWeakCount(100);
		adaboost[i]->setWeightTrimRate(0.95);
		adaboost[i]->setMaxDepth(5);
		adaboost[i]->setUseSurrogates(false);
	}
	
	

	
}


Adaboost::~Adaboost()
{
}
