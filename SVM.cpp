#include "SVM.h"


void SVM::SVMTrain(string trainpath) {
	//��ȡѵ������
	Mat classes;
	Mat trainingData;
	Mat trainingImages;
	vector<int> trainingLabels;
	cout << "SVM111 initialization" << endl;
	fileop.getTrainSet(trainpath,trainingImages, trainingLabels);
	
	Mat(trainingImages).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);
	svm->train(trainingData, cv::ml::ROW_SAMPLE, classes);
	svm->save("svm.xml");
	cout << "SVM train done��" << endl;

}


int SVM::SVMTest(Mat inMat) {
	resize(inMat, inMat, Size(12, 18));
	Mat p = inMat.reshape(1, 1);
	p.convertTo(p, CV_32FC1);
	int result = (int)svm->predict(p);
	return result;
}

void SVM::SVMTest(string testpath) {
	vector<string> filenames;
	fileop.getFiles(testpath, filenames);
	string save_path = "svmtestset/";
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
		int result = (int)svm->predict(p);
		if (result == num) {
			correctsize++;
		}
		else {
			cout << "����ʶ���ͼƬ��" <<  filenames[i] << endl;
		}
	}
	cout << "��ȷ��Ϊ��" << double(correctsize) / double(wholesize) << endl;

}



SVM::SVM()
{
	// ��ʼ��
	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setDegree(0);
	svm->setGamma(1);
	svm->setCoef0(0);
	svm->setC(1);
	svm->setNu(0);
	svm->setP(0);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-2));
}


SVM::~SVM()
{
}
