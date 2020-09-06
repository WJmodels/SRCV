#include "HorizontalBracket.h"

// 按照行坐标升序排列
bool cmp_x2(pair<int, int> a, pair<int, int> b) {
	return (a.first < b.first);
}

// 按照列坐标升序排列
bool cmp_y2(pair<int, int> a, pair<int, int> b) {
	return (a.second < b.second);
}

// Point按照行坐标降序排列
bool point_cmp_row2(pair<Point, Point> a, pair<Point, Point> b) {
	return (a.first.y > b.first.y);
}

// Point按照列坐标升序排列
bool point_cmp_col2(pair<Point, Point> a, pair<Point, Point> b) {
	return (a.first.x < b.first.x);
}

HorizontalBracket::HorizontalBracket()
{
	//cout << "Horizonta初始化" << endl;
}


HorizontalBracket::~HorizontalBracket()
{
}

// 输入图片，转换为二值图片,背景是0，前景是1
Mat HorizontalBracket::ConvertToBinImage(Mat image, int MaxValue, int BinaryType) {
	Mat image_bin, gray;
	// 转为灰度图
	cvtColor(image, gray, COLOR_BGR2GRAY);
	// 中值滤波
	// medianBlur(gray, gray, 3);
	// 自适应阈值
	adaptiveThreshold(gray, image_bin, MaxValue, ADAPTIVE_THRESH_MEAN_C, BinaryType, 7, 10);
	return image_bin;
}



int HorizontalBracket::icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{

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


cv::Scalar HorizontalBracket::icvprGetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void HorizontalBracket::icvprLabelColor(const cv::Mat & _labelImg, cv::Mat & _colorLabelImg)
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


// 检测水平括号以及括号内的数字
void HorizontalBracket::detectHorizontalBracketAndNum(Mat image, OCR &ocr, vector<string> &numbers, vector<pair<int, int> > &bracket_pos) {

	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);

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
	imwrite("result.bmp", grayImg);
	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 2; i < labels.size(); i++) {

		if (labels[i].size() > 700 || labels[i].size() < 5) {
			labels[i].clear();
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小区分数字块
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x2);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y2);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x1 >= 1280 && x1 <= 1430& x2 - x1 >= 5 && y2 - y1 >= 12 && y2 - y1 <= 200) {
			//rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		}


	}


	// 按照行坐标从下到上的顺序进行排序

	std::sort(NumSet.begin(), NumSet.end(), point_cmp_row2);
	// 水平括号的基准位置
	int horizontalbase = NumSet.back().first.y;

	//cout << "firstYY:" << horizontalbase << endl;
	// 括号位置集合
	vector<pair<Point, Point> > brackets;
	// 括号数字集合
	vector<pair<Point, Point> > bracketnumbers;
	for (int i = 0; i < NumSet.size(); i++) {
		if (NumSet[i].first.y >= horizontalbase - 5 && NumSet[i].first.y <= horizontalbase + 5) {
			brackets.push_back(NumSet[i]);
		}
		else {
			bracketnumbers.push_back(NumSet[i]);
		}
	}
	


	// 对括号位置集合进行列排序，按照从左到右的顺序
	std::sort(brackets.begin(), brackets.end(), point_cmp_col2);
	// 对括号数字位置进行列排序，按照从左到右的顺序
	std::sort(bracketnumbers.begin(), bracketnumbers.end(), point_cmp_col2);

	std::cout << "pixel coordinate of bracket" << endl;

	//for (int i = 0; i < bracketnumbers.size(); i++) {
	//	cout << "XX:" << bracketnumbers[i].first.x << "   YY:" << bracketnumbers[i].first.y << endl;
	//				}
//	cout << "brackets.size:" << brackets.size()<< endl;
	// 准确率100%
	// 对每个水平括号的1/5处进行行扫描，检测左右位置
	for (int i = 0; i < brackets.size(); i++) {

		// 扫描线
		int bracket_base = brackets[i].first.y + 3;

		Vec3b basecolor = colorLabelImg.at<Vec3b>(brackets[i].second.y, (brackets[i].first.x + brackets[i].second.x) / 2);
		// 基准色
		for (int k = brackets[i].first.y; k <= brackets[i].second.y; k++) {
			if (basecolor != Vec3b(0, 0, 0)) {
				break;
			}
			basecolor = colorLabelImg.at<Vec3b>(k, (brackets[i].first.x + brackets[i].second.x) / 2);
		}
		vector<int> temp_brackets;
		for (int c = brackets[i].first.x; c <= brackets[i].second.x; c++) {
			if (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
				temp_brackets.push_back(c);
				while (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
					c++;
				}
			}
		}
		for (int k = 0; k < temp_brackets.size() - 1; k++) {
			// 如果距离过近
			if (temp_brackets[k + 1] - temp_brackets[k] <= 4) {
				continue;
			}
			// 如果有重复
			if (!bracket_pos.empty() && pair<int, int>(temp_brackets[k], temp_brackets[k + 1]) == bracket_pos.back()) {
				continue;
			}
			std::cout << temp_brackets[k] << " " << temp_brackets[k + 1] << endl;
			bracket_pos.push_back(pair<int, int>(temp_brackets[k], temp_brackets[k + 1]));
		}
	}




	// 打印括号位置，检查，全部OK!!!
	/*
	for (int i = 0; i < brackets.size(); i++) {
		cout << brackets[i].first.x << " " << brackets[i].first.y << endl;
	}
	*/
	std::cout << "recognition results of integral:" << endl;
	vector<int> lenk;
	int wp = 0;
	for (int i = 1; i < bracketnumbers.size(); i++)
	{

		if (((bracketnumbers[i - 1].first.x - bracketnumbers[i].first.x) < 5) && ((bracketnumbers[i - 1].first.x - bracketnumbers[i].first.x) > -5))
		{
			
			wp++;
			
		}
		else
		{
			lenk.push_back(wp);
			//cout << "haoxue" << wp << "haoxuesi" << bracketnumbers[i].first.x << endl;
			wp = 0;
		}
	}
	lenk.push_back(wp);
	int count1 = 0;
	int count2 = 0;


	vector<pair<Point, Point> > Singlenum;
	for (int i = 0; i < bracketnumbers.size(); i++) {
		Singlenum.push_back(bracketnumbers[i]);
		// 每3个截取一个数字
		count2 = lenk[count1];
		if (Singlenum.size() == count2+1) {
			// 按行降序排列
			//sort(Singlenum.begin(), Singlenum.end(), point_cmp_col2);
			//sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			vector<Mat> num_mat;
			for (int k = 0; k < (count2 + 1); k++) {
				// 切割数字
				Mat temp = grayImage(Rect(Singlenum[k].first.x, Singlenum[k].first.y, Singlenum[k].second.x - Singlenum[k].first.x + 1, Singlenum[k].second.y - Singlenum[k].first.y + 1));
				// 顺时针旋转90度
				transpose(temp, temp);
				flip(temp, temp, 1);
				//imshow("temp", temp);
				//waitKey(0);
				imwrite("temp.png", temp);
				temp = imread("temp.png");
				num_mat.push_back(temp);

			}
			string detectnum = ocr.DetectNum(num_mat);
			int iPos = detectnum.length() - 2;
			string numstr = detectnum.substr(0, iPos) + "." + detectnum.substr(iPos, 2);
			std::cout << numstr << endl;
			numbers.push_back(numstr);

			Singlenum.clear();
			count2 = 0;
			count1++;
		}
	}

	// cout << (numbers.size() == bracket_pos.size() ? "true" : "false") << endl;




	imwrite("color.bmp", colorLabelImg);

	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	//colored_img.display("rectangle");
}
void HorizontalBracket::detectCNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos) {

	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);

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
	imwrite("result.bmp", grayImg);
	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 2; i < labels.size(); i++) {

		if (labels[i].size() > 400 || labels[i].size() < 5) {
			labels[i].clear();
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小区分数字块
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x2);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y2);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		//cout << "xxx:" << x1 << "|:" << x2 << "yyy:" << y1 << "|:" << y2 << endl;


		if (x1 >= 20 && x1 <= 125 && x2 - x1 >= 5 && y2 - y1 >= 10 && y2 - y1 <= 150) {
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
			//cout <<"X1:" << y1 << "Y1:" << x1<< "   X2:" <<y2<< "Y2:" << x2<< endl;
		}


	}


	// 按照行坐标从下到上的顺序进行排序
	//std::sort(NumSet.begin(), NumSet.end(), point_cmp_row2);
	std::sort(NumSet.begin(), NumSet.end(), point_cmp_col2);
	

	


	// 水平括号的基准位置
	int horizontalbase = NumSet.back().first.y;
	// 括号位置集合
	vector<pair<Point, Point> > brackets;
	// 括号数字集合
	vector<pair<Point, Point> > bracketnumbers;
	for (int i = 0; i < NumSet.size(); i++) {
		
			bracketnumbers.push_back(NumSet[i]);
			
		
	}
/*
	// 对括号位置集合进行列排序，按照从左到右的顺序
	std::sort(brackets.begin(), brackets.end(), point_cmp_col2);
	// 对括号数字位置进行列排序，按照从左到右的顺序
	std::sort(bracketnumbers.begin(), bracketnumbers.end(), point_cmp_col2);

	std::cout << "括号像素位置：" << endl;

	// 准确率100%
	// 对每个水平括号的1/5处进行行扫描，检测左右位置
	for (int i = 0; i < brackets.size(); i++) {

		// 扫描线
		int bracket_base = brackets[i].first.y + 3;

		Vec3b basecolor = colorLabelImg.at<Vec3b>(brackets[i].second.y, (brackets[i].first.x + brackets[i].second.x) / 2);
		// 基准色
		for (int k = brackets[i].first.y; k <= brackets[i].second.y; k++) {
			if (basecolor != Vec3b(0, 0, 0)) {
				break;
			}
			basecolor = colorLabelImg.at<Vec3b>(k, (brackets[i].first.x + brackets[i].second.x) / 2);
		}
		vector<int> temp_brackets;
		for (int c = brackets[i].first.x; c <= brackets[i].second.x; c++) {
			if (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
				temp_brackets.push_back(c);
				while (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
					c++;
				}
			}
		}
		for (int k = 0; k < temp_brackets.size() - 1; k++) {
			// 如果距离过近
			if (temp_brackets[k + 1] - temp_brackets[k] <= 4) {
				continue;
			}
			// 如果有重复
			if (!bracket_pos.empty() && pair<int, int>(temp_brackets[k], temp_brackets[k + 1]) == bracket_pos.back()) {
				continue;
			}
			std::cout << temp_brackets[k] << " " << temp_brackets[k + 1] << endl;
			bracket_pos.push_back(pair<int, int>(temp_brackets[k], temp_brackets[k + 1]));
		}
	}


	*/

	// 打印括号位置，检查，全部OK!!!
	/*
	for (int i = 0; i < brackets.size(); i++) {
		cout << brackets[i].first.x << " " << brackets[i].first.y << endl;
	}
	*/
	std::cout << "recognition results of integral" << endl;

	vector<pair<Point, Point> > Singlenum;
	for (int i = 0; i < bracketnumbers.size(); i++) {
		Singlenum.push_back(bracketnumbers[i]);
		// 每3个截取一个数字
		if (Singlenum.size() == 3) {
			// 按行降序排列
			//sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			sort(Singlenum.begin(), Singlenum.end(), point_cmp_col2);
			sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			vector<Mat> num_mat;
			for (int k = 0; k < 3; k++) {
				// 切割数字
				Mat temp = grayImage(Rect(Singlenum[k].first.x, Singlenum[k].first.y, Singlenum[k].second.x - Singlenum[k].first.x + 1, Singlenum[k].second.y - Singlenum[k].first.y + 1));
				// 顺时针旋转90度
				//cout << "x1:" << Singlenum[k].first.x << "x2:" << Singlenum[k].first.x + 1 << "y1" << Singlenum[k].first.y <<"y2 " << Singlenum[k].first.y + 1 << endl;
				transpose(temp, temp);
				flip(temp, temp, 1);
				//imshow("temp", temp);
				//waitKey(0);
				
			   //std::string str = "temp " + std::to_string(i)+ ".png ";
				imwrite("temp.png", temp);
				//imwrite(str, temp);
				temp = imread("temp.png");
				num_mat.push_back(temp);

			}
			string detectnum = ocr.DetectNum(num_mat);
			string numstr = detectnum.substr(0, 1) + "." + detectnum.substr(1, 2);

			numbers.push_back(numstr);

			Singlenum.clear();
		}
	}

	// cout << (numbers.size() == bracket_pos.size() ? "true" : "false") << endl;




	imwrite("color.bmp", colorLabelImg);

	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	//colored_img.display("rectangle");
}
void HorizontalBracket::detectWNTRNum(Mat image, OCR& ocr, vector<string>& numbers, vector<pair<int, int> >& bracket_pos) {

	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);

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
	imwrite("result.bmp", grayImg);
	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 2; i < labels.size(); i++) {

		if (labels[i].size() > 400 || labels[i].size() < 5) {
			labels[i].clear();
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小区分数字块
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x2);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y2);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		//cout << "xxx:" << x1 << "|:" << x2 << "yyy:" << y1 << "|:" << y2 << endl;


		if (x1 >= 110 && x1 <= 196 && x2 - x1 >= 4 && y2 - y1 >= 10 && y2 - y1 <= 150) {
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
			//cout << "xxx:" << x1 << "|:" << y1 << "yyy:" << x2 << "|:" << y2 << endl;
		}


	}


	// 按照行坐标从下到上的顺序进行排序
	//std::sort(NumSet.begin(), NumSet.end(), point_cmp_row2);
	std::sort(NumSet.begin(), NumSet.end(), point_cmp_col2);
	//std::sort(NumSet.begin(), NumSet.end(), point_cmp_row2);






	// 水平括号的基准位置
	int horizontalbase = NumSet.back().first.y;
	// 括号位置集合
	vector<pair<Point, Point> > brackets;
	// 括号数字集合
	vector<pair<Point, Point> > bracketnumbers;
	for (int i = 0; i < NumSet.size(); i++) {

		bracketnumbers.push_back(NumSet[i]);
	//	cout << "X1: " << bracketnumbers[i].first.x<< " X2:" << bracketnumbers[i].second.x << "  Y1:" << bracketnumbers[i].first.y << "  Y2" << bracketnumbers[i].second.y << endl;

	}

	//cout << "XXXYY1: " << bracketnumbers.size();

	/*
		// 对括号位置集合进行列排序，按照从左到右的顺序
		std::sort(brackets.begin(), brackets.end(), point_cmp_col2);
		// 对括号数字位置进行列排序，按照从左到右的顺序
		std::sort(bracketnumbers.begin(), bracketnumbers.end(), point_cmp_col2);

		std::cout << "括号像素位置：" << endl;

		// 准确率100%
		// 对每个水平括号的1/5处进行行扫描，检测左右位置
		for (int i = 0; i < brackets.size(); i++) {

			// 扫描线
			int bracket_base = brackets[i].first.y + 3;

			Vec3b basecolor = colorLabelImg.at<Vec3b>(brackets[i].second.y, (brackets[i].first.x + brackets[i].second.x) / 2);
			// 基准色
			for (int k = brackets[i].first.y; k <= brackets[i].second.y; k++) {
				if (basecolor != Vec3b(0, 0, 0)) {
					break;
				}
				basecolor = colorLabelImg.at<Vec3b>(k, (brackets[i].first.x + brackets[i].second.x) / 2);
			}
			vector<int> temp_brackets;
			for (int c = brackets[i].first.x; c <= brackets[i].second.x; c++) {
				if (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
					temp_brackets.push_back(c);
					while (colorLabelImg.at<Vec3b>(bracket_base, c) == basecolor) {
						c++;
					}
				}
			}
			for (int k = 0; k < temp_brackets.size() - 1; k++) {
				// 如果距离过近
				if (temp_brackets[k + 1] - temp_brackets[k] <= 4) {
					continue;
				}
				// 如果有重复
				if (!bracket_pos.empty() && pair<int, int>(temp_brackets[k], temp_brackets[k + 1]) == bracket_pos.back()) {
					continue;
				}
				std::cout << temp_brackets[k] << " " << temp_brackets[k + 1] << endl;
				bracket_pos.push_back(pair<int, int>(temp_brackets[k], temp_brackets[k + 1]));
			}
		}


		*/

		// 打印括号位置，检查，全部OK!!!
		/*
		for (int i = 0; i < brackets.size(); i++) {
			cout << brackets[i].first.x << " " << brackets[i].first.y << endl;
		}
		*/
	
	std::cout << "recognition results of integral" << endl;
	vector<int> lenk;
	int wp = 0;
	for (int i = 1; i < bracketnumbers.size(); i++)
	{
		
		if (((bracketnumbers[i - 1].first.x - bracketnumbers[i].first.x) < 5) && ((bracketnumbers[i - 1].first.x - bracketnumbers[i].first.x) > -5))
		{
			//cout << "haoxuesi:" << i<<endl;
			wp++;
			//cout << "kuanghaoxue" << wp << endl;
		}
		else
		{
			lenk.push_back(wp);
			//cout << "haoxue" << wp << "haoxuesi" << bracketnumbers[i].first.x << endl;
			wp = 0;
		}
	}
	lenk.push_back(wp);
	//cout << "haoxue" << wp << "haoxuesi"  << endl;
	vector<pair<Point, Point> > Singlenum;
	int count1 = 0;
	int count2 = 0;
	for (int i = 0; i < bracketnumbers.size(); i++) {
		Singlenum.push_back(bracketnumbers[i]);
		// 每n个截取一个数字
		
		
		count2 = lenk[count1];

		if (Singlenum.size() == (count2+1)) {
			// 按行降序排列
			//sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			sort(Singlenum.begin(), Singlenum.end(), point_cmp_col2);
			sort(Singlenum.begin(), Singlenum.end(), point_cmp_row2);
			vector<Mat> num_mat;
			for (int k = 0; k < (count2 + 1); k++) {
				// 切割数字
				Mat temp = grayImage(Rect(Singlenum[k].first.x, Singlenum[k].first.y, Singlenum[k].second.x - Singlenum[k].first.x + 1, Singlenum[k].second.y - Singlenum[k].first.y + 1));
				// 顺时针旋转90度
				//cout << "x1:" << Singlenum[k].first.x << "x2:" << Singlenum[k].first.x + 1 << "y1" << Singlenum[k].first.y <<"y2 " << Singlenum[k].first.y + 1 << endl;
				transpose(temp, temp);
				flip(temp, temp, 1);
			//  imshow("temp", temp);
			//	waitKey(0);

			   //std::string str = "temp " + std::to_string(i)+ ".png ";
				imwrite("temp.png", temp);
				//imwrite(str, temp);
				temp = imread("temp.png");
				num_mat.push_back(temp);

			}
			string detectnum = ocr.DetectNum(num_mat);
			string numstr = detectnum.substr(0, count2-1) + "." + detectnum.substr(count2-1, count2);
			std::cout << numstr << endl;
			numbers.push_back(numstr);

			Singlenum.clear();
			count2 = 0;
			count1++;
		}
	}

	// cout << (numbers.size() == bracket_pos.size() ? "true" : "false") << endl;




	imwrite("color.bmp", colorLabelImg);

	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	//colored_img.display("rectangle");
}


// scale:起始的像素纵坐标col_start和数字col_start_num，单位长度的像素大小unit_pixels,以及单位长度unit
vector<pair<double, double>> HorizontalBracket::convertToActualBracket(vector<double>scale, vector<pair<int, int> > bracket_pos) {
	vector<pair<double, double> > actual_bracket_pos;
	for (int i = 0; i < bracket_pos.size(); i++) {
		double left = bracket_pos[i].first;
		double right = bracket_pos[i].second;
		left = (double(left - scale[0]) / scale[2]) * scale[3] + scale[1];
		right = (double(right - scale[0]) / scale[2]) * scale[3] + scale[1];
		actual_bracket_pos.push_back(pair<double, double>(left, right));
	}
	return actual_bracket_pos;
}



vector<pair<int, int> > HorizontalBracket::detectVerticalBracket(Mat image) {
	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);

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
	imwrite("result.bmp", grayImg);
	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 2; i < labels.size(); i++) {

		if (labels[i].size() > 700 || labels[i].size() < 5) {
			labels[i].clear();
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小区分括号
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x2);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y2);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x1 >= 900 && x2 < 1360 && x2 - x1 >= 7 && x2 - x1 <= 12 && y2 - y1 >= 11 && y2 - y1 <= 18) {
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		}
		
	}

	sort(NumSet.begin(), NumSet.end(), point_cmp_col2);
	vector<pair<int, int> > brackets;
	bool isLeft = false;
	int left_bracket_pos;
	for (int i = 0; i < NumSet.size(); i++) {
		if (NumSet[i].second.y - NumSet[i].first.y >= 20) {
			isLeft = !isLeft;
			// 当前是左括号
			if (isLeft == true) {
				left_bracket_pos = NumSet[i].second.x;
			}
			// 当前是右括号
			else {
				brackets.push_back(pair<int, int>(left_bracket_pos, NumSet[i].first.x));
				// 如果当前是右括号，并且右边是数字，说明该括号同时又是左括号
				if (i + 1 < NumSet.size() && NumSet[i + 1].second.y - NumSet[i + 1].first.y < 20) {
					isLeft = !isLeft;
					left_bracket_pos = NumSet[i].second.x;
				}
			}
		}
	}
	imwrite("color.bmp", colorLabelImg);
	std::cout << "pixel coordinate of bracket" << endl;
	for (int i = 0; i < brackets.size(); i++) {
		std::cout << brackets[i].first << " " << brackets[i].second << endl;
	}
	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	//colored_img.display("rectangle");
	return brackets;

}




// 检测垂直括号图像括号内的数字
vector<string> HorizontalBracket::detectVerticalBracketNum(Mat image, OCR &ocr) {
	vector<string> numbers;

	Mat grayImage = ConvertToBinImage(image, 255, THRESH_BINARY);
	Mat binImage = ConvertToBinImage(image, 1, THRESH_BINARY_INV);

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
	imwrite("result.bmp", grayImg);
	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	// 记录切割的数字的坐标，point格式
	vector<pair<Point, Point> > NumSet;

	for (int i = 2; i < labels.size(); i++) {

		if (labels[i].size() > 200 || labels[i].size() < 5) {
			labels[i].clear();
			continue;
		}
		vector<pair<int, int> > sort_by_row = labels[i];
		vector<pair<int, int> > sort_by_col = labels[i];
		// 根据方块的大小区分数字块
		std::sort(sort_by_row.begin(), sort_by_row.end(), cmp_x2);
		std::sort(sort_by_col.begin(), sort_by_col.end(), cmp_y2);
		int x1 = sort_by_row.front().first;
		int y1 = sort_by_col.front().second;
		int x2 = sort_by_row.back().first;
		int y2 = sort_by_col.back().second;
		if (x1 >= 900 && x2 < 1360 && x2 - x1 >= 7 && x2 - x1 <= 12 && y2 - y1 >= 11 && y2 - y1 <= 18) {
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		}
		/*
		else if (x1 >= 900 && x2 < 1360 && x2 - x1 >= 2 && x2 - x1 <= 7 && y2 - y1 >= 2 && y2 - y1 <= 5 && labels[i].size() > 8) {
			rectangle(colorLabelImg, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255));
			NumSet.push_back(pair<Point, Point>(Point(y1, x1), Point(y2, x2)));
		}
		*/
	}


	// 按照列坐标从左到右的顺序进行排序
	std::sort(NumSet.begin(), NumSet.end(), point_cmp_col2);

	std::cout << "recognition results of integral" << endl;

	vector<pair<Point, Point> > num;
	for (int i = 0; i < NumSet.size(); i++) {
		num.push_back(NumSet[i]);
		// 每3个截取一个数字
		if (num.size() == 3) {
			// 按行降序排列
			sort(num.begin(), num.end(), point_cmp_row2);
			vector<Mat> num_mat;
			for (int k = 0; k < 3; k++) {
				// 切割数字
				Mat temp = grayImage(Rect(num[k].first.x, num[k].first.y, num[k].second.x - num[k].first.x + 1, num[k].second.y - num[k].first.y + 1));
				// 顺时针旋转90度
				transpose(temp, temp);
				flip(temp, temp, 1);
				//imshow("temp", temp);
				//waitKey(0);
				imwrite("temp.png", temp);
				temp = imread("temp.png");
				num_mat.push_back(temp);

			}
			string detectnum = ocr.DetectNum(num_mat);
			string numstr = detectnum.substr(0, 1) + "." + detectnum.substr(1, 2);
			numbers.push_back(numstr);
			std::cout << numstr << endl;
			num.clear();
		}
	}

	imwrite("color.bmp", colorLabelImg);

	CImg<unsigned char> colored_img;
	colored_img.load_bmp("color.bmp");
	//colored_img.display("rectangle");
	return numbers;
}



