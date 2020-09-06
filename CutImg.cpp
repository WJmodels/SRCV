#include "CutImg.h"





// 输入图片，转换为二值图片,背景是0，前景是1
Mat CutImg::ConvertToBinImage(Mat image, int MaxValue, int BinaryType) {
	Mat image_bin, gray;
	// 转为灰度图
	cvtColor(image, gray, COLOR_BGR2GRAY);
	// 中值滤波
	//medianBlur(gray, gray, 3);
	// 自适应阈值
	adaptiveThreshold(gray, image_bin, MaxValue, ADAPTIVE_THRESH_MEAN_C, BinaryType, 7, 10);
	return image_bin;
}



int CutImg::icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	// connected component analysis (4-component)  
	// use two-pass algorithm  
	// 1. first pass: label each foreground pixel with a label  
	// 2. second pass: visit each labeled pixel and merge neighbor labels  
	//   
	// foreground pixel: _binImg(x,y) = 1  
	// background pixel: _binImg(x,y) = 0  


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return 0;
	}

	// 1. first pass  

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2  
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0  
	labelSet.push_back(1);   // foreground: 1  

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _lableImg.ptr<int>(i - 1);
		int* data_curRow = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				int leftupPixel = data_preRow[j - 1];
				int rightupPixel = data_preRow[j + 1];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}
				if (leftupPixel > 1) {
					neighborLabels.push_back(leftupPixel);
				}
				if (rightupPixel > 1) {
					neighborLabels.push_back(rightupPixel);
				}
				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // assign to a new label  
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// save equivalence  
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	int maxLabel = 0;

	// update equivalent labels  
	// assigned with the smallest label in each equivalent label set  
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;

	}


	// 2. second pass  
	for (int i = 0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
			if (pixelLabel > maxLabel) {
				maxLabel = pixelLabel;
			}
		}
	}

	return maxLabel;
}


cv::Scalar CutImg::icvprGetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void CutImg::icvprLabelColor(const cv::Mat & _labelImg, cv::Mat & _colorLabelImg)
{
	if (_labelImg.empty() ||
		_labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = _labelImg.rows;
	int cols = _labelImg.cols;

	_colorLabelImg.release();
	_colorLabelImg.create(rows, cols, CV_8UC3);
	_colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)_labelImg.ptr<int>(i);
		uchar* data_dst = _colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = icvprGetRandomColor();
				}
				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}



// 按照行坐标升序排列
struct cmp_x {
	bool operator()(pair<int, int> a, pair<int, int> b) {
		return (a.first < b.first);
	}
} cmp_x1;

// 按照列坐标升序排列
struct cmp_y {
	bool operator() (pair<int, int> a, pair<int, int> b) {
		return (a.second < b.second);
	}
} cmp_y1;

// Point按照行坐标升序排列
struct point_cmp_row {
	bool operator ()(pair<Point, Point> a, pair<Point, Point> b) {
		return (a.first.y < b.first.y);
	}
} point_cmp_row1;

struct point_cmp_col {
	bool operator()(pair<Point, Point> a, pair<Point, Point> b) {
		return (a.first.x < b.first.x);
	}
} point_cmp_col1;




// 输入原图像，切割标尺图片
void CutImg::CutNum(Mat image, string save_filename) {
	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);
	int scale_row_start, scale_row_end, scale_col_start, scale_col_end;
	// connected component labeling  
	cv::Mat labelImg;
	int maxLabel = icvprCcaByTwoPass(binImage, labelImg);
	vector<vector<pair<int, int> > > labels;
	labels.resize(maxLabel + 1);
	for (int i = 0; i < labelImg.rows; i++) {
		for (int j = 0; j < labelImg.cols; j++) {
			labels[labelImg.at<int>(i, j)].push_back(pair<int, int>(i, j));
		}
	}

	
	// show result  
	cv::Mat grayImg;
	labelImg *= 10;
	labelImg.convertTo(grayImg, CV_8UC1);

	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	

	for (int i = 2; i < labels.size(); i++) {

		// 清除噪点
		if (labels[i].size() < 5) {
			for (int k = 0; k < labels[i].size(); k++) {
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[0] = 0;
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[1] = 0;
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[2] = 0;

				//				rectangle(colorLabelImg, Point(labels[i][k].second, labels[i][k].first), Point(labels[i][k].second, labels[i][k].first), Scalar(0, 0, 0));
								// cout << labels[i][k].first << " " << labels[i][k].second << endl;
			}
			continue;
		}

		else if (labels[i].size() < 500) {
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小标尺位置
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x1);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y1);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x2 - x1 >= 10 && x2 - x1 <= 30 && y2 - y1 >= 1600) {
			cout << "Mark out the scale." << endl;
			scale_row_start = x1;
			scale_row_end = x2;
			scale_col_start = y1;
			scale_col_end = y2;
			cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
			//rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
		}
	}

	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 0; i < labels.size(); i++) {
		if (labels[i].size() > 300 || labels[i].size() < 7) {
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小标尺位置
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x1);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y1);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x1 > scale_row_end && x1 < scale_row_end + 70) {
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		}
	}

	// 按照列坐标从左到右的顺序进行排序
	std::sort(NumSet.begin(), NumSet.end(), point_cmp_col1);
	vector<Mat> num1, num2;
	double num_first, num_second;



	for (int i = 0; i < NumSet.size(); i++) {
		// 切割数字
		Mat temp = grayImage(Rect(NumSet[i].first.x, NumSet[i].first.y, NumSet[i].second.x - NumSet[i].first.x + 1, NumSet[i].second.y - NumSet[i].first.y + 1));
		string _save_filename = save_filename + std::to_string(i) + ".bmp";
		imwrite(_save_filename, temp);
	}

}

// 输入原图像，得到标尺信息，返回起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
vector<double> CutImg::ScaleDetect(Mat image, OCR& ocr) {
	//cout << "hello " << endl;
	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	//cout << "hello1 " << endl;
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);
	//cout << "hello2 " << endl;
	int scale_row_start, scale_row_end, scale_col_start, scale_col_end;
	// connected component labeling  
	cv::Mat labelImg;
	int maxLabel = icvprCcaByTwoPass(binImage, labelImg);
	vector<vector<pair<int, int> > > labels;
	labels.resize(maxLabel + 1);
	for (int i = 0; i < labelImg.rows; i++) {
		for (int j = 0; j < labelImg.cols; j++) {
			labels[labelImg.at<int>(i, j)].push_back(pair<int, int>(i, j));
		}
	}

	// show result  
	cv::Mat grayImg;
	labelImg *= 10;
	labelImg.convertTo(grayImg, CV_8UC1);

	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);

	for (int i = 2; i < labels.size(); i++) {

		// 清除噪点
		if (labels[i].size() < 5) {
			for (int k = 0; k < labels[i].size(); k++) {
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[0] = 0;
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[1] = 0;
				colorLabelImg.at<Vec3b>(labels[i][k].first, labels[i][k].second)[2] = 0;

				//				rectangle(colorLabelImg, Point(labels[i][k].second, labels[i][k].first), Point(labels[i][k].second, labels[i][k].first), Scalar(0, 0, 0));
								// cout << labels[i][k].first << " " << labels[i][k].second << endl;
			}
			continue;
		}

		else if (labels[i].size() < 500) {
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小标尺位置
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x1);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y1);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		//if (x2 - x1 >= 10 && x2 - x1 <= 30 && y2 - y1 >= 1600) {
		if (x2 - x1 <= 30 && y2 - y1 >= 1600) {
			cout << "Mark out the scale." << endl;
			scale_row_start = x1;
			scale_row_end = x2;
			scale_col_start = y1;
			scale_col_end = y2;
			//cout << "___________" << x2 << " " << y2 << endl;
			// rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
		}
	}

	// 刻度尺的刻度
	vector<int> scale_col_pos;


	// 扫描线所在的行
	int scan_line = scale_row_start + (scale_row_end - scale_row_start) * 2 / 3;
	cout << "scan line" << scan_line << endl;
	for (int i = scale_col_start; i <= scale_col_end; i++) {
		if (colorLabelImg.at<Vec3b>(scan_line, i)[0] > 0
			|| colorLabelImg.at<Vec3b>(scan_line, i)[1] > 0
			|| colorLabelImg.at<Vec3b>(scan_line, i)[2] > 0
			) {
			// 取中间位置
			int cnt = 0;
			for (int k = i; k <= scale_col_end; k++) {
				if (colorLabelImg.at<Vec3b>(scan_line, k)[0] > 0
					|| colorLabelImg.at<Vec3b>(scan_line, k)[1] > 0
					|| colorLabelImg.at<Vec3b>(scan_line, k)[2] > 0
					) {
					cnt++;
				}
				else {
					// cout << i + cnt / 2 << endl;
					scale_col_pos.push_back(i + cnt / 2);
					i += cnt;
					break;
				}
			}
			i += 3;
		}
	}


	int unit_pixels = (scale_col_pos.back() - scale_col_pos.front()) / (scale_col_pos.size() - 1);

	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 0; i < labels.size(); i++) {
		if (labels[i].size() > 300 || labels[i].size() < 3) {
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小标尺位置
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x1);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y1);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x1 > scale_row_end && x1 < scale_row_end + 50) {
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
			//cout << "X1:" << y1 << "Y1:" << x1 << "X2" << y2 << "Y2" << x2 << endl;
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
		}
		//if ( x1 > scale_row_end  && x1 < scale_row_end + 70 && x2 - x1 >= 13 && x2 - x1 <= 20 && y2 - y1 >= 9 && y2 - y1 <= 14) {
			//NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		//}
		//else if (x1 > scale_row_end && x1 < scale_row_end + 70 && x2 - x1 >= 3 && x2 - x1 <= 7 && y2 - y1 >= 2 && y2 - y1 <= 5) {
			//NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		//}
	}

	// 按照列坐标从左到右的顺序进行排序
	std::sort(NumSet.begin(), NumSet.end(), point_cmp_col1);
	vector<Mat> num1, num2;
	string num_first, num_second;
	for (int i = 3; i < 6; i++) {
		// 切割数字
		Mat temp = grayImage(Rect(NumSet[i].first.x, NumSet[i].first.y, NumSet[i].second.x - NumSet[i].first.x + 1, NumSet[i].second.y - NumSet[i].first.y + 1));
		imwrite("temp.png", temp);
		temp = imread("temp.png");
		num1.push_back(temp);
		rectangle(colorLabelImg, NumSet[i].first, NumSet[i].second, Scalar(0, 0, 255));
	}
	// 用OCR识别数字
	num_first = ocr.DetectNum(num1);
	for (int i = 6; i < 9; i++) {
		// 切割数字
		Mat temp = grayImage(Rect(NumSet[i].first.x, NumSet[i].first.y, NumSet[i].second.x - NumSet[i].first.x + 1, NumSet[i].second.y - NumSet[i].first.y + 1));
		imwrite("temp.png", temp);
		temp = imread("temp.png");
		num2.push_back(temp);
		rectangle(colorLabelImg, NumSet[i].first, NumSet[i].second, Scalar(0, 0, 255));
	}
	// 用OCR识别数字
	num_second = ocr.DetectNum(num2);

	// 计算刻度1，刻度2，记录起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
	int col_start = scale_col_pos[1];

	double col_start_num, col_end_num;

	stringstream ss;
	ss << num_first;
	ss >> col_start_num;
	ss.clear();

	ss << num_second;
	ss >> col_end_num;


	double unit = col_end_num - col_start_num;
	//cout << "unit " << unit << endl;
	//cout << "unit_pixels " << unit_pixels << endl;
	//cout << col_start_num << " " << col_end_num << endl;

	imwrite("color.bmp", colorLabelImg);

	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	// colored_img.display("rectangle");

	vector<double> return_values;
	return_values.push_back(col_start);
	return_values.push_back(col_start_num);
	return_values.push_back(unit_pixels);
	return_values.push_back(unit);
	return return_values;
}



CutImg::CutImg()
{	
	//cout << "cut初始化"<< endl;
	svm.SVMTrain("testset");

}


CutImg::~CutImg()
{
}
