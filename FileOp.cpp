#include "FileOp.h"

void FileOp::getFiles(string path, vector<string>& files)
{
	//�ļ����  
	long long hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int FileOp::getLabel(string filename) {
	int pos;
	//cout << filename << endl;
	for (int i = 0; i < filename.length(); i++) {
		if (filename[i] == '_') {
			pos = i + 1;
			break;
		}
	}
	int label = filename[pos] - '0';
	return label;
}

void FileOp::getTrainSet(string trainsetpath, Mat & trainingImages, vector<int> & trainingLabels) {
	string filePath = trainsetpath;
	vector<string> files;
	getFiles(filePath, files);
	for (int i = 0; i < files.size(); i++) {
		Mat  SrcImage = imread(files[i].c_str());
		resize(SrcImage, SrcImage, Size(12, 18));
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(getLabel(files[i]));
	}
}


FileOp::FileOp()
{
}


FileOp::~FileOp()
{
}
