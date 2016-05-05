// barcodeDetection.cpp : �������̨Ӧ�ó������ڵ㡣
//

// mser.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <io.h>
#include <cstring>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;




//����Բ���ʵ�ֵ
const float PI = 3.1415926;

#define min(a, b) (a < b ? a : b)

/*****************************************************************************\
*                 getFiles()��������ȡ��·���������ļ�                           *
\*****************************************************************************/
//��ȡ��·���������ļ�
void getFiles(string path, vector<string>& files)
{
	//�ļ����
	long hFile = 0;

	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼������
			//������ǣ������б�
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);

	}
}

//��תͼ�����ݲ��䣬�ߴ���Ӧ���
IplImage* rotateImage2(IplImage* img, int degree)
{
	double angle = degree  * CV_PI / 180.;
	double a = sin(angle), b = cos(angle);
	int width = img->width, height = img->height;
	//��ת�����ͼ�ߴ�
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), img->depth, img->nChannels);
	cvZero(img_rotate);
	//��֤ԭͼ��������Ƕ���ת����С�ߴ�
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;
	int tempX = (tempLength + 1) / 2 - width / 2;
	int tempY = (tempLength + 1) / 2 - height / 2;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth, img->nChannels);
	cvZero(temp);
	//��ԭͼ���Ƶ���ʱͼ��tmp����
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));
	cvCopy(img, temp, NULL);
	cvResetImageROI(temp);
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = b;
	m[1] = a;
	m[3] = -m[1];
	m[4] = m[0];
	// ����ת��������ͼ���м�  
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;
	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(temp, img_rotate, &M);
	cvReleaseImage(&temp);
	return img_rotate;
}


//��תͼ�����ݲ��䣬�ߴ���Ӧ���  
IplImage* rotateImage1(IplImage* img, int degree){
	double angle = degree  * CV_PI / 180.; // ����    
	double a = sin(angle), b = cos(angle);
	int width = img->width;
	int height = img->height;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//��ת����map  
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]  
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]  
	float map[6];
	CvMat map_matrix = cvMat(2, 3, CV_32F, map);
	// ��ת����  
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 3);
	//��ͼ��������任  
	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�  
	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.  
	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��  
	cvWarpAffine(img, img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
	return img_rotate;
}



/************************************************************************/
/* ��ɫ�ռ�ת��	RGB TO HSI                                              */
/************************************************************************/
int rgb2hsi(Mat &image, Mat &hsi){
	if (!image.data){
		cout << "Miss Data" << endl;
		return -1;
	}
	int nl = image.rows;
	int nc = image.cols;
	if (image.isContinuous()){
		nc = nc*nl;
		nl = 1;
	}
	for (int i = 0; i < nl; i++){
		uchar *src = image.ptr<uchar>(i);
		uchar *dst = hsi.ptr<uchar>(i);
		for (int j = 0; j < nc; j++){
			float b = src[j * 3] / 255.0;
			float g = src[j * 3 + 1] / 255.0;
			float r = src[j * 3 + 2] / 255.0;
			float num = (float)(0.5*((r - g) + (r - b)));
			float den = (float)sqrt((r - g)*(r - g) + (r - b)*(g - b));
			float H, S, I;
			if (den == 0){	//��ĸ����Ϊ0
				H = 0;
			}
			else{
				double theta = acos(num / den);
				if (b <= g)
					H = theta / (PI * 2);
				else
					H = (2 * PI - theta) / (2 * PI);
			}
			double minRGB = min(min(r, g), b);
			den = r + g + b;
			if (den == 0)	//��ĸ����Ϊ0
				S = 0;
			else
				S = 1 - 3 * minRGB / den;
			I = den / 3.0;
			//��S������H���������䵽[0,255]�����Ա�����ʾ;
			//һ��H������[0,2pi]֮�䣬S��[0,1]֮��
			dst[3 * j] = H * 255;
			dst[3 * j + 1] = S * 255;
			dst[3 * j + 2] = I * 255;
		}
	}
	return 0;
}

/************************************************************************/
/*    ɫ�ʿռ�ת��                                                       */
/************************************************************************/
int rgb2cmyk(Mat &image, Mat &cmyk){
	if (!image.data){
		cout << "Miss Data" << endl;
		return -1;
	}
	int nl = image.rows;	//����
	int nc = image.cols;	//����
	if (image.isContinuous()){	//û�ж���������
		nc = nc*nl;
		nl = 1;					//It is now a 1D array
	}
	//��������ͼ�񣬱�ѭ��ִֻ��1��
	for (int i = 0; i < nl; i++){
		uchar *data = image.ptr<uchar>(i);
		uchar *dataCMYK = cmyk.ptr<uchar>(i);
		for (int j = 0; j < nc; j++){
			uchar b = data[3 * j];
			uchar g = data[3 * j + 1];
			uchar r = data[3 * j + 2];
			uchar c = 255 - r;
			uchar m = 255 - g;
			uchar y = 255 - b;
			uchar k = min(min(c, m), y);
			dataCMYK[4 * j] = c - k;
			dataCMYK[4 * j + 1] = m - k;
			dataCMYK[4 * j + 2] = y - k;
			dataCMYK[4 * j + 3] = k;
		}
	}
	return 0;
}


/************************************************************************/
/* ���ֱ��ͼ                                                            */
/************************************************************************/
enum
{
	HIST_ROW,
	HIST_COL
};

Mat get_histogram(Mat image, int hist_type)
{
	CV_Assert(image.type() == CV_8U);
	cv::Mat hist;

	if (hist_type == HIST_ROW)
	{
		hist == cv::Mat::zeros(image.rows, 1, CV_32S);
		for (int t = 0; t < image.rows; ++t)
		{
			cv::Scalar sum = cv::sum(image.row(t));
			hist.at<uint>(t, 0) = sum[0];
		}
	}
	else if (hist_type == HIST_COL)
	{
		hist = cv::Mat::zeros(image.cols, 1, CV_32S);
		for (int c = 0; c < image.cols; ++c)
		{
			cv::Scalar sum = cv::sum(image.col(c));
			hist.at<uint>(c, 0) = sum[0];
		}
	}

	return hist;
}


/***********************************************************************************\
*                                main����                                           *
\***********************************************************************************/

int main(int argc, char* argv[])
{

	//������Ҫ��ȡ��ͼƬ�ļ�Ŀ¼
	//char * filePath = "C:\\Users\\Administrator\\Desktop\\�����\\1d_barcode_extended\\1d_barcode_extended\\resized_25%\\JPEGImages";
	char * filePath = "C:\\Users\\zhaoqiang\\Desktop\\msertest\\normal";
	//char * filePath = "C:\\Users\\zhaoqiang\\Desktop\\msertest\\test";
	//char * filePath = "C:\\Users\\zhaoqiang\\Desktop\\angleNo";
	vector<string> files;
	//��ȡ��·���������ļ�
	getFiles(filePath, files);

	char str[80];
	int size = files.size();

	//for (int i = 0; i < size; i++)
	//{
	//	cout << files[i].c_str() << endl;
	//}

	Mat src;

	for (int i = 0; i < 10; ++i)
	{
		src = imread(files[i].c_str());

		//����ͼ����
		Size src_size = src.size();

		//�����հ�ͼ������ʾ�м��1��2�����˽��
		Mat immiSrc(src_size, CV_8UC3, Scalar(255, 255, 255));
		Mat immiSrc2(src_size, CV_8UC3, Scalar(255, 255, 255));
		Mat immiSrc3(src_size, CV_8UC3, Scalar(255, 255, 255));

		//ת����ɫ�ռ�ΪHSI�ռ䣬
		//Mat hsiImage;
		//vector<Mat> vecHsi;
		//hsiImage.create(src.rows, src.cols, CV_8UC3);
		//rgb2hsi(src, hsiImage);
		//split(hsiImage, vecHsi);
		//Mat cmykImage, hsvImage;
		//vector<Mat> vecHsv, vecCmyk;
		//cmykImage.create(src.rows, src.cols, CV_8UC4);
		//cvtColor(src, hsvImage, CV_BGR2HSV);
		//rgb2cmyk(src, cmykImage);

		//ofstream fout;
		//fout.open("1.txt");



		//ת��Ϊ�Ҷ�ͼ
		Mat textImg;
		cvtColor(src, textImg, CV_BGR2GRAY);


		//����MSER��ͬʱȷ����������
		Ptr<MSER> ms = MSER::create(10, 30, 14400, 0.25, 0.2);
		//Ptr<MSER> ms = MSER::create(7, 20, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5);

		//���������������ص㼯
		vector<vector<Point> > regions;
		vector<Rect> bboxes;

		ms->detectRegions(textImg, regions, bboxes);
		//ms->detectRegions(src, regions, bboxes);

		//����RotatedRect����,�洢��ϵľ���
		vector<RotatedRect> rorect;

		//����RotatedRect���󣬴洢��1��2��filterd��ľ���
		vector<RotatedRect> filterroRect1;
		vector<RotatedRect> filterroRect2;

		//rorect.resize(bboxes.size());

		//����һ��Mat���������洢Ҫ���о��������
		Mat clustrMat;

		//�ڻҶ�ͼ�����þ��λ������
		for (int j = 0; j < bboxes.size(); j++)
		{

			rorect.push_back(fitEllipse(regions[j]));

			//����4���������ڻ�������
			Point2f vertices[4];
			Point vertice[4];
			rorect[j].points(vertices);

			/************************************************************************/
			/*			�������ε��д���                                              */
			/************************************************************************/
			//�����д��߳���
			float length = 30;
			//������һ������
			Point2f p2;

			//�ԽǶ�angle�����ж��Ƿ����90�����廭�д��߽Ƕ�angle2
			//float angle2 = 0;
			//if (rorect[j].angle > 90)
			//{
			//	angle2 = rorect[j].angle + 90.0 - 180.0;
			//}
			//else
			//{
			//	angle2 = rorect[j].angle + 90.0;
			//}

			p2.x = (float)std::round(rorect[j].center.x + length * cos(rorect[j].angle * CV_PI / 180.0));
			p2.y = (float)std::round(rorect[j].center.y + length * sin(rorect[j].angle * CV_PI / 180.0));




			/*************************************************************************************************\
			*	����1����ʹ��fillConvexPoly()	    �����forѭ����Ϊ���ų���height��width֮��С��10��ͬʱ��������		  *
			\*************************************************************************************************/

			//if (rorect[j].size.height / rorect[j].size.width > 10)
			//{
			//	for (int k = 0; k < 4; ++k)
			//	{
			//		vertice[k] = vertices[k];
			//	}
			//	//�������һ����ɫ
			//	//Scalar color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
			//	//������
			//	fillConvexPoly(immiSrc, vertice, 4, CV_RGB(0, 255, 0));
			//	//�����д���
			//	line(immiSrc, rorect[j].center, p2, CV_RGB(0, 0, 255));
			//}
			//else if (rorect[j].size.width / rorect[j].size.height > 10)
			//{
			//	for (int k = 0; k < 4; ++k)
			//	{
			//		vertice[k] = vertices[k];
			//	}
			//	//�������һ����ɫ
			//	//Scalar color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
			//	//������
			//	fillConvexPoly(immiSrc, vertice, 4, CV_RGB(0, 255, 0));
			//	//�����д���
			//	line(immiSrc, rorect[j].center, p2, CV_RGB(0, 0, 255));
			//}




			/************************************************************************/
			/*	����2����	�������λ������αߵķ����������β������д���                 */
			/************************************************************************/

			if (rorect[j].size.height / rorect[j].size.width > 10)
			{
				//�ѵ�һ��filter�Ľ������
				filterroRect1.push_back(rorect[j]);
				//���λ����εı���
				for (int k = 0; k < 4; ++k)
				{
					cv::line(immiSrc, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
				}
				//�����д���
				cv::line(immiSrc, rorect[j].center, p2, CV_RGB(0, 0, 255));
			}
			else if (rorect[j].size.width / rorect[j].size.height > 10)
			{
				//�ѵ�һ��filter�Ľ������
				filterroRect1.push_back(rorect[j]);
				//���λ������ε�����
				for (int k = 0; k < 4; ++k)
				{
					cv::line(immiSrc, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
				}
				//�����д���
				cv::line(immiSrc, rorect[j].center, p2, CV_RGB(0, 0, 255));
			}



			//���height��widthС���򽻻�ֵ,����angle+90��
			//if (rorect[j].size.height < rorect[j].size.width)
			//{

			//	double temp	= rorect[j].size.height;
			//	rorect[j].size.height = rorect[j].size.width;
			//	rorect[j].size.width = temp;
			//	rorect[j].angle += PI / 2;
			//}


			//����Բ��ϳ�MSER����
			//ellipse(src, fitEllipse(regions[j]), CV_RGB(0, 255, 0));
			//ellipse(src, fitEllipse(regions[j]), CV_RGB(rand() % 255, rand() % 255, rand() % 255), -1);
			//rectangle(src, bboxes[i], CV_RGB(255, 0, 0));
		}




		/************************************************************************/
		/* ʧ�ܵ�K��ֵ����                                                        */
		/************************************************************************/
		////�����ά�������ڴ洢Ҫ���������
		//int sampleCount = rorect.size();
		//vector<Point2f> points;
		//Mat labels, centers;

		//for (int m = 0; m < rorect.size(); ++m)
		//{
		//	/************************************************************************/
		//	/* ��ȡ�д��ߵĽǶȺ;�����Ϣ�����ھ���                                     */
		//	/************************************************************************/
		//	if (rorect[m].size.height / rorect[m].size.width > 10)
		//	{
		//		Point2f tempPoint = 0;
		//		tempPoint.x = rorect[m].angle;
		//		//���Ƕ�ת��Ϊ����
		//		float arc = rorect[m].angle * CV_PI / 180.0;
		//		tempPoint.y = int(abs(rorect[m].center.y - arc * rorect[m].center.x) / sqrt(arc * arc + 1));
		//		//�����ݵ����Ҫ���������
		//		points.push_back(tempPoint);

		//	}
		//	else if (rorect[m].size.width / rorect[m].size.height > 10)
		//	{
		//		Point2f tempPoint = 0;
		//		tempPoint.x = rorect[m].angle;
		//		//���Ƕ�ת��Ϊ����
		//		float arc = rorect[m].angle * CV_PI / 180.0;
		//		tempPoint.y = int(abs(rorect[m].center.y - arc * rorect[m].center.x) / sqrt(arc * arc + 1));
		//		//�����ݵ����Ҫ���������
		//		points.push_back(tempPoint);
		//	}

		//}

		////ʹ��K��ֵ���෽�����о���
		//int clusterCount = 3;
		//kmeans(points, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

		////���������������
		//int labelKind[3] = { 0 };

		//for (int k = 0; k < labels.rows; ++k)
		//{
		//	int labelValue = labels.at<int>(k, 0);
		//	switch (labelValue)
		//	{
		//	case 0:
		//		labelKind[0]++;
		//		break;
		//	case 1:
		//		labelKind[1]++;
		//		break;
		//	case 2:
		//		labelKind[2]++;
		//		break;
		//	//case 3:
		//	//	labelKind[3]++;
		//	//	break;
		//	//case 4:
		//	//	labelKind[4]++;
		//	//	break;
		//	//case 5:
		//	//	labelKind[5]++;
		//	//	break;
		//	//case 6:
		//	//	labelKind[6]++;
		//	//	break;
		//	//case 7:
		//	//	labelKind[7]++;
		//	//	break;
		//	}
		//}

		//int maxLabel = 0, maxKind = 0;

		//for (int m = 0; m < clusterCount; ++m)
		//{
		//	if (labelKind[m] > maxKind)
		//	{
		//		maxKind = labelKind[m];
		//		maxLabel = m;
		//	}
		//}

		//for (int m = 0; m < points.size(); ++m)
		//{
		//	//����4���������ڻ�������
		//	Point2f vertices[4];
		//	Point vertice[4];
		//	filterroRect1[m].points(vertices);

		//	if (labels.at<int>(m, 0) == maxLabel)
		//	{
		//		for (int k = 0; k < 4; ++k)
		//		{
		//			cv::line(immiSrc3, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
		//		}
		//	}
		//}







		/*****************************************************************************************\
		*				���ýǶ���Ϣ���й���														  *
		\*****************************************************************************************/
		////����һ���Ƕ�ֱ��ͼ
		//int angleMatrix[181] = { 0 };
		//for (int k = 0; k < filterroRect1.size(); ++k)
		//{
		//	//ͳ�ƽǶ�ֱ��ͼ
		//	++angleMatrix[int(filterroRect1[k].angle)];
		//}

		////����������Ƕ�
		//int mainAngle = 0;
		//for (int k2 = 0; k2 < 181; ++k2)
		//{
		//	if (angleMatrix[k2] > mainAngle)
		//		mainAngle = angleMatrix[k2];
		//}

		//for (int k3 = 0; k3 < filterroRect1.size(); ++k3)
		//{
		//	//����4���������ڻ�������
		//	Point2f vertices[4];
		//	Point vertice[4];
		//	filterroRect1[k3].points(vertices);

		//	if (int(filterroRect1[k3].angle) >= mainAngle - 2 && int(filterroRect1[k3].angle) <= mainAngle + 2)
		//	{
		//		filterroRect2.push_back(filterroRect1[k3]);
		//		for (int k = 0; k < 4; ++k)
		//		{
		//			vertice[k] = vertices[k];
		//		}
		//		//�������һ����ɫ
		//		//Scalar color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
		//		//������
		//		fillConvexPoly(immiSrc2, vertice, 4, CV_RGB(0, 255, 0));
		//	}
		//}

		//real_2d_array

		/*****************************************************************************************\
		*			ʹ��hough�任��һ��filter����													  *
		\*****************************************************************************************/
		//����hough�ۻ�����
		int houghMatrix[181][1000] = { 0 };

		//�������߶ν���Hough�����ۼ�
		for (int k = 0; k < filterroRect1.size(); ++k)
		{
			//���Ƕ�ת��Ϊ����
			float arc = filterroRect1[k].angle * CV_PI / 180.0;
			//�����߶ε���0��0���ľ���
			int dist = 0;

			//�ж�X�����Ƿ�Ϊ����������
			if (filterroRect1[k].center.x < 0)
			{
				continue;
			}
			//�ж�angle�Ƿ�Ϊ0��180
			if (int(filterroRect1[k].angle) == 0 || int(filterroRect1[k].angle) == 180)
			{
				dist = int(filterroRect1[k].center.y);
				++houghMatrix[int(filterroRect1[k].angle)][dist];
			}
			else if (int(filterroRect1[k].angle) == 90)
			{
				dist = int(filterroRect1[k].center.x);
				++houghMatrix[int(filterroRect1[k].angle)][dist];
			}
			else
			{
				dist = int(abs(filterroRect1[k].center.y - arc * filterroRect1[k].center.x) / sqrt(arc * arc + 1.0));
				++houghMatrix[int(filterroRect1[k].angle)][dist];
			}
		}

		//�����ֵ���ĽǶ�ֵ�ͼ���
		int mainAngle = 0;
		int mainDist = 0;
		int maxHoughValue = 0;

		//�ҳ���ֵ���ĽǶ�ֵ
		for (int m = 0; m < 181; ++m)
		{
			for (int n = 0; n < 1000; ++n)
			{
				if (houghMatrix[m][n] > maxHoughValue)
				{
					maxHoughValue = houghMatrix[m][n];
					mainAngle = m;
					mainDist = n;
				}
			}
		}

		//�ҳ�filterroRect�еķ�ֵ>=5�ľ���
		for (int k2 = 0; k2 < filterroRect1.size(); ++k2)
		{

			//����4���������ڻ�������
			Point2f vertices[4];
			Point vertice[4];
			filterroRect1[k2].points(vertices);
			//���Ƕ�ת��Ϊ����
			float arc = filterroRect1[k2].angle * CV_PI / 180.0;

			//�жϽǶ�����Ƿ���󣬹�������ȥ����
			if (abs(filterroRect1[k2].angle - mainAngle) > 90)
			{
				int miniAngle = filterroRect1[k2].angle < mainAngle ? filterroRect1[k2].angle : mainAngle;
				int maxiAngle = filterroRect1[k2].angle > mainAngle ? filterroRect1[k2].angle : mainAngle;
				if (miniAngle + 180 - maxiAngle > 5)
				{
					continue;
				}
			}
			else
			{
				if (abs(filterroRect1[k2].angle - mainAngle) > 5)
				{
					continue;
				}
			}

			//�����������ĳ��Ϳ��ֵ��С������ȥ
			if (filterroRect1[k2].size.width < 3 && filterroRect1[k2].size.height < 15)
			{
				continue;
			}
			//�������˺�ľ���
			int dist = 0;
			if (int(filterroRect1[k2].angle) == 0 || int(filterroRect1[k2].angle == 180))
			{
				dist = int(filterroRect1[k2].center.y);

				if (houghMatrix[int(filterroRect1[k2].angle)][dist] >= 2)
				{
					//�ѵ�һ��filter�Ľ������
					filterroRect2.push_back(filterroRect1[k2]);
					//���λ����εı���
					for (int k = 0; k < 4; ++k)
					{
						//line(immiSrc2, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
						vertice[k] = vertices[k];
					}
					//������
					fillConvexPoly(immiSrc2, vertice, 4, CV_RGB(0, 255, 0));

					//if (houghMatrix[int(filterroRect1[k2].angle)][dist] > maxHoughValue)
					//{
					//	maxHoughValue = houghMatrix[int(filterroRect1[k2].angle)][dist];
					//	maxAngle = filterroRect1[k2].angle;
					//}

				}
			}
			else if (int(filterroRect1[k2].angle) == 90)
			{
				dist = int(filterroRect1[k2].center.x);

				if (houghMatrix[int(filterroRect1[k2].angle)][dist] >= 2)
				{
					//�ѵ�һ��filter�Ľ������
					filterroRect2.push_back(filterroRect1[k2]);
					//���λ����εı���
					for (int k = 0; k < 4; ++k)
					{
						//line(immiSrc2, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
						vertice[k] = vertices[k];
					}
					//������
					fillConvexPoly(immiSrc2, vertice, 4, CV_RGB(0, 255, 0));

				}
			}
			else
			{
				dist = int(abs(filterroRect1[k2].center.y - arc * filterroRect1[k2].center.x) / sqrt(arc * arc + 1.0));

				if (houghMatrix[int(filterroRect1[k2].angle)][dist] >= 2)
				{
					//�ѵ�һ��filter�Ľ������
					filterroRect2.push_back(filterroRect1[k2]);
					//���λ����εı���
					for (int k = 0; k < 4; ++k)
					{
						//line(immiSrc2, vertices[k], vertices[(k + 1) % 4], CV_RGB(0, 255, 0));
						vertice[k] = vertices[k];
					}
					//������
					fillConvexPoly(immiSrc2, vertice, 4, CV_RGB(0, 255, 0));

					//fout << dist << "\t" << filterroRect1[k2].size.height << endl;
				}
			}
		}



		/************************************************************************/
		/* ���Ͳ�����������                                                       */
		/************************************************************************/
		//Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));

		////dilate(immiSrc2, immiSrc3, element);
		////erode(immiSrc2, immiSrc3, element);
		//morphologyEx(immiSrc2, immiSrc3, MORPH_RECT, element);




		/************************************************************************/
		/* ��תͼ��ʹ������ˮƽ��ֱ                                              */
		/************************************************************************/
		Mat srcCopy = src.clone();
		IplImage * src_img = &(IplImage(srcCopy));
		IplImage * ipl_img = &(IplImage(immiSrc2));

		////�ж����angle
		//int maxAngle = mainAngle;
		//if (maxAngle >= 90)
		//{
		//	maxAngle = 180 - maxAngle;
		//}
		//��������������תͼ
		IplImage * temp_img = rotateImage1(ipl_img, mainAngle);
		Mat temp_mat = cvarrToMat(temp_img);
		Mat temp_mat2(temp_mat.size(), CV_8UC3, Scalar(255, 255, 255));

		//���ԭͼ�����תͼ
		IplImage * temp_src = rotateImage1(src_img, mainAngle);
		Mat temp_srcMat = cvarrToMat(temp_src);





		//row_hist = get_histogram();



		/************************************************************************/
		/* ��������ͶӰֱ��ͼ��ѡ�������󽻼�                                       */
		/************************************************************************/
		int rowNum = temp_mat.rows;
		int colNum = temp_mat.cols;
		int rowBin[1000] = { 0 };
		int colBin[1000] = { 0 };

		//���н���ֱ��ͼͶӰ
		for (int m = 0; m < rowNum; ++m)
		{
			for (int n = 0; n < colNum; ++n)
			{
				if (temp_mat.at<Vec3b>(m, n)[0] == 0 && temp_mat.at<Vec3b>(m, n)[1] == 255)
				{
					rowBin[m] += 1;
				}
			}
		}

		//���н���ֱ��ͼͶӰ
		for (int m = 0; m < colNum; ++m)
		{
			for (int n = 0; n < rowNum; ++n)
			{
				if (temp_mat.at<Vec3b>(n, m)[0] == 0 && temp_mat.at<Vec3b>(n, m)[1] == 255)
				{
					colBin[m] += 1;
				}
			}
		}
		//�ҳ�������������ϡ����ϡ����¡������ĸ���
		int lowXvalue = 0, lowYvalue = 0, highXvalue = 0, highYvalue = 0;

		for (int k = 0; k < rowNum; ++k)
		{
			if (rowBin[k] > 1)
			{
				lowXvalue = k;
				break;
			}
		}

		for (int k = rowNum - 1; k >= 0; --k)
		{
			if (rowBin[k] > 1)
			{
				highXvalue = k;
				break;
			}
		}

		for (int k = 0; k < colNum; ++k)
		{
			if (colBin[k] > 1)
			{
				lowYvalue = k;
				break;
			}
		}

		for (int k = colNum - 1; k >= 0; --k)
		{
			if (colBin[k] > 1)
			{
				highYvalue = k;
				break;
			}
		}
		//�������
		cv::line(temp_srcMat, Point(lowYvalue, lowXvalue), Point(lowYvalue, highXvalue), Scalar(0, 255, 0), 2);
		cv::line(temp_srcMat, Point(lowYvalue, lowXvalue), Point(highYvalue, lowXvalue), Scalar(0, 255, 0), 2);
		cv::line(temp_srcMat, Point(lowYvalue, highXvalue), Point(highYvalue, highXvalue), Scalar(0, 255, 0), 2);
		cv::line(temp_srcMat, Point(highYvalue, lowXvalue), Point(highYvalue, highXvalue), Scalar(0, 255, 0), 2);

		//�ٰѻ���ܺ��ͼ��������ת����
		IplImage * ipl_result = &(IplImage(temp_srcMat));
		IplImage * ipl_tempresu = rotateImage1(ipl_result, -mainAngle);
		Mat result_img = cvarrToMat(ipl_tempresu);

		/***************************************************************************************\
		*					���Ի������������													*
		\***************************************************************************************/
		//�������������ĸ����㡣
		Point2f px_min = (0, 0);
		Point2f px_max = (0, 0);
		Point2f py_min = (0, 0);
		Point2f py_max = (0, 0);
		//������������ĸ�����
		for (int k3 = 0; k3 < filterroRect2.size(); ++k3)
		{
			//����4���������ڻ�������
			Point2f vertices[4];
			filterroRect1[k3].points(vertices);
			for (int k4 = 0; k4 < 4; ++k4)
			{
				if (vertices[k4].x < px_min.x)
				{
					px_min = vertices[k4];
				}
				if (vertices[k4].y < py_min.y)
				{
					py_min = vertices[k4];
				}
				if (vertices[k4].x > px_max.x)
				{
					px_max = vertices[k4];
				}
				if (vertices[k4].y > py_max.y)
				{
					py_max = vertices[k4];
				}
			}
		}
		//�����������ı߿�
		Point2f center = ((px_max.x + px_min.x) / 2, (py_max.y + py_min.y) / 2);




		/*****************************************\
		*         ��ÿ������������ʾ��д��           *
		\*****************************************/
		stringstream stream;
		string str2 = "";
		stream << i;
		stream >> str2;
		//ÿ�����ڵ���ʾ��
		string srcName = "src" + str2;
		string mserName = "mser" + str2;
		string houghName = "hough" + str2;
		string rotateImage = "rotate" + str2;
		string rotateImage2 = rotateImage;
		string dilateImage = "dilate" + str2;
		string resultImage = "result" + str2;
		string cluster1 = "cluster" + str2;
		//string hsiName		= "hsiImage" + str2;
		//string hsvName		= "hsvIamge" + str2;
		//string cmykName		= "cmykImage" + str2;
		//stream.clear();
		//char * ch = new char[30];
		//stream << rotateImage;
		//stream >> ch;

		//for (int i = 0; i < 4; ++i)
		//{

		//	string splitName = "split" + str2 + i;
		//	imshow(splitName, vecCmyk[i]);
		//}


		/************************************************************************/
		/*                        ʵ�ֱ���ͼƬ��д�뵽�ļ���                       */
		/************************************************************************/
		//Ҫд���ͼƬ��
		//string saveName = "C:\\Users\\zhaoqiang\\Desktop\\mserresult\\filteredBlock3\\" + str2 + ".jpg";
		//string saveName = "C:\\Users\\zhaoqiang\\Desktop\\testresult\\ab\\" + str2 + ".jpg";
		//��ʾͼ��
		//namedWindow(srcName);
		//namedWindow(mserName);
		//namedWindow(houghName);
		//cvNamedWindow(ch);
		//namedWindow(rotateImage);
		//namedWindow(rotateImage2);
		//namedWindow(resultImage);
		//imshow(srcName, src);
		cv::imshow(mserName, immiSrc);
		//cv::imshow(hsiName, hsiImage);
		//cv::imshow(hsvName, hsvImage);
		//cv::imshow(cmykName, cmykImage);
		cv::imshow(houghName, immiSrc2);
		//cv::imshow(dilateImage, immiSrc3);
		//imshow(cluster1, immiSrc3);
		//imshow(rotateImage, temp_mat);
		//imshow(rotateImage2, temp_srcMat);
		//cv::imshow(resultImage, result_img);
		//cvShowImage(ch, temp_img);
		//����ͼƬ��д�뵽�ļ���
		//imwrite(saveName, temp_srcMat);

		//cvReleaseImage(&temp_img);
		//cvReleaseImage(&ptr_ipl);
		//temp_img = nullptr;
		//ipl_img = nullptr;

	}

	waitKey(0);


	//Mat image(200, 200, CV_8UC3, Scalar(0));
	//RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 50), 60);

	//Point2f vertice[4];
	//rRect.points(vertice);
	//for (int i = 0; i < 4; i++)
	//	line(image, vertice[i], vertice[(i + 1) % 4], Scalar(0, 255, 0));

	//Rect brect = rRect.boundingRect();
	//rectangle(image, brect, Scalar(255, 0, 0));

	//imshow("rectangles", image);
	//waitKey(0);



	return 0;
}
















//
//int sliderPos = 70;
//
//Mat image;
//
//void processImage(int, void*);
//
//int main(int argc, char** argv)
//{
//	const char* filename = argc == 2 ? argv[1] : (char*)"stuff.jpg";
//	image = imread(filename, 0);
//	if (image.empty())
//	{
//		cout << "Couldn't open image " << filename << "\nUsage: fitellipse <image_name>\n";
//		return 0;
//	}
//
//	imshow("source", image);
//	namedWindow("result", 1);
//
//	// Create toolbars. HighGUI use.
//	createTrackbar("threshold", "result", &sliderPos, 255, processImage);
//	processImage(0, 0);
//
//	// Wait for a key stroke; the same function arranges events processing
//	waitKey();
//	return 0;
//}
//
//// Define trackbar callback functon. This function find contours,
//// draw it and approximate it by ellipses.
//void processImage(int /*h*/, void*)
//{
//	vector<vector<Point> > contours;
//	Mat bimage = image >= sliderPos;
//
//	findContours(bimage, contours, RETR_LIST, CHAIN_APPROX_NONE);
//
//	Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);
//
//	for (size_t i = 0; i < contours.size(); i++)
//	{
//		size_t count = contours[i].size();
//		if (count < 6)
//			continue;
//
//		Mat pointsf;
//		Mat(contours[i]).convertTo(pointsf, CV_32F);
//		RotatedRect box = fitEllipse(pointsf);
//
//		if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
//			continue;
//		drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);
//
//		ellipse(cimage, box, Scalar(0, 0, 255), 1, LINE_AA);
//		ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0, 255, 255), 1, LINE_AA);
//		Point2f vtx[4];
//		box.points(vtx);
//		for (int j = 0; j < 4; j++)
//			line(cimage, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
//	}
//
//	imshow("result", cimage);
//}
