#include "KNN.h"


void KNN::KNNTrain(string trainpath) {
	//获取训练数据
	Mat classes;
	Mat trainingData;
	Mat trainingImages;
	vector<int> trainingLabels;
	fileop.getTrainSet(trainpath, trainingImages, trainingLabels);
	Mat(trainingImages).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);
	knn->setDefaultK(4);
	knn->train(trainingData, cv::ml::ROW_SAMPLE, classes);
	knn->save("svm.xml");
	cout << "KNN train done！" << endl;
}


int KNN::KNNTest(Mat inMat) {
	resize(inMat, inMat, Size(12, 18));
	Mat p = inMat.reshape(1, 1);
	p.convertTo(p, CV_32FC1);
	int result = (int)knn->predict(p);
	return result;
}





void KNN::KNNTest(string trainpath) {
	vector<string> filenames;
	fileop.getFiles(trainpath, filenames);
	string save_path = "knntestset/";
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
		//cout << "111"<< filenames[i];
		Mat inMat = imread(filenames[i]);
		resize(inMat, inMat, Size(12, 18));
		Mat p = inMat.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		int result = (int)knn->predict(p);
		string savename = save_path + to_string(i + 1) + "_" + to_string(result) + ".png";
		imwrite(savename, inMat);
		if (result == num) {
			correctsize++;
		}
		else {
			cout << "错误识别的图片：" << filenames[i] << endl;
		}
	}
	cout << "正确率为：" << double(correctsize) / double(wholesize) << endl;
}


KNN::KNN()
{
	knn = cv::ml::KNearest::create();
}


KNN::~KNN()
{
}
