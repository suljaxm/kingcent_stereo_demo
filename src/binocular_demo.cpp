#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include "error.h"

using namespace std;
using namespace cv;


VideoCapture capCam;

ERROR_CODE Initialize(int deviceID){

	capCam.open(deviceID);
	if (!capCam.isOpened()){
		return CAMERA_OPEN_FAILED;
	}

	capCam.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	capCam.set(CV_CAP_PROP_FPS, 60);
	capCam.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capCam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	return INITIALIZING;
}

void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap);
void insertDepth32f(cv::Mat& depth);

int  main()
{
	Mat frame;
	Mat image_left, image_right, frame_l, frame_r;

	// VideoCapture cap(1);
	Initialize(1);
    
    /*
	KRX*/

	int width = capCam.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capCam.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frameRate = capCam.get(CV_CAP_PROP_FPS);
	int totalFrames = capCam.get(CV_CAP_PROP_FRAME_COUNT);

	// Create a Rect box, which belongs to the class in "cv". 
	// The four parameters are x, y, width, height. 

	Rect left_rect(0, 0, width / 2, height);  
	Rect right_rect(width / 2, 0, width / 2, height);

	cout << "width=" << width << endl;
	cout << "height=" << height << endl;
	cout << "totalFrames=" << totalFrames << endl;
	cout << "frameRate=" << frameRate << endl;
	

	Size imageSize(width / 2, height);
	Mat cameraMatrix[2], distCoeffs[2];

	Mat R, T, E, F;


	FileStorage fs("../parameters/intrinsics.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["M1"] >> cameraMatrix[0];
		fs["D1"] >> distCoeffs[0];
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1];

		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	cout << "cameraMatrix[0]:" << cameraMatrix[0] << endl;
	cout << "cameraMatrix[1]:" << cameraMatrix[1] << endl;

	cout << "distCoeffs[0]:" << distCoeffs[0] << endl;
	cout << "distCoeffs[1]:" << distCoeffs[1] << endl;

	Mat R1, R2, P1, P2, Q; //outside parameter
	Rect validRoi[2];

	fs.open("../parameters/extrinsics.yml", FileStorage::READ); //read outside parameter
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["R2"] >> R2;
		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	cout << "start stereoRectify" << endl;
	stereoRectify(
		cameraMatrix[0], //internal parameter matrix
		distCoeffs[0],   //distortion parameter
		cameraMatrix[1], 
		distCoeffs[1], 
		imageSize,  //image size
		R,  // Rotation matrix
		T,  // Translation matrix
		R1, // left rotaion correction parameters
		R2, // right rotaion correction parameters
		P1, // left translation correction parameters
		P2, // right translation correction parameters
		Q, // depth correction parameters
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]); 
	
	cout << "end stereoRectify" << endl;
	cout << "validRoi[0]:" << validRoi[0] << endl;
	cout << "validRoi[1]:" << validRoi[1] << endl;

	cout << R << endl;
	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
	cout << "isVerticalStereo:" << isVerticalStereo << endl;
	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;
	while (1)
	{
		capCam >> frame;
		frame_l = Mat(frame, left_rect).clone();
		frame_r = Mat(frame, right_rect).clone();

		if (frame_l.empty() || frame_r.empty())
			continue;

		// left image
		remap(frame_l, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
		cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		// right image
		remap(frame_r, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;  // set display image rection

		imgLeft = canvasPart1(vroi).clone();
		imgRight = canvasPart2(vroi).clone();


		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl; return -1;
		}

		imshow("imgLeft", imgLeft);
		imshow("imgRight", imgRight);

		int minDisparity = 0;  		// 最小的视差值
		int numDisparities = ((imgLeft.cols / 8) + 15) & -16;; 	// 视差范围，即最大视差值和最小视差值之差，必须是16的倍数 ((imgSize.width / 8) + 15) & -16
		int blockSize = 3; 			// 匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
		int cn = imgLeft.channels(); 
		int P1 = 8*cn*blockSize*blockSize; 	// 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize
		int P2 = 4*P2; 		        // P2=4*P1
		int disp12MaxDiff = 1; 		// 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查
		int preFiterCap = 32;		// 预滤波图像像素的截断值。该算法首先计算每个像素的x导数，并通过[-preFilterCap，preFilterCap]间隔剪切其值。结果值被传递给Birchfield-Tomasi像素成本函数。
		int uniquenessRatio = 10;   // 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
		int speckleWindowSize = 100;
		int speckleRange = 32;		// 视差变化阈值
		int mode = StereoSGBM::MODE_SGBM;

		static Ptr<StereoSGBM> sgbm = StereoSGBM::create(
		minDisparity,
		numDisparities,
		blockSize,
		P1,
		P2,
		disp12MaxDiff,
		preFiterCap,
		uniquenessRatio,
		speckleWindowSize,
		speckleRange,
		mode);

		//-- And create the image in which we will save our disparities
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);
		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / (numDisparities*16.0)); //将16位符号整形的视差矩阵转换为8位无符号整形矩阵

		cv::imshow("disparity", sgbmDisp8U);

		Mat  sgbmDisparityShow;
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		Mat depth = Mat(imgLeft.rows, imgLeft.cols, CV_16UC1);
		disp2Depth(sgbmDisparityShow, depth);

		cv::imshow("depth", depth); 

		// insertDepth32f(depth);  // error?
		// cv::imshow("depth2", depth);
		char c = (char)waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
	return 0;
}

/*
函数作用：视差图转深度图
输入：
　　dispMap ----视差图，8位单通道，CV_8UC1
输出：
　　depthMap ----深度图，16位无符号单通道，CV_16UC1
*/
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap)
{
    int type = dispMap.type();

	double baseline = 1.0394043770848877e+02; // 基线距离100 mm
	double fx = 4.8285420143582115e+02;
    if (type == CV_8U)
    {
        int height = dispMap.rows;
        int width = dispMap.cols;

        uchar* dispData = (uchar*)dispMap.data;
        ushort* depthData = (ushort*)depthMap.data;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int id = i*width + j;
                if (!dispData[id]){
					// depthData[id] = 0;//
					continue;  //防止0除
				}  
                depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]) );
            }
        }
    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}

void insertDepth32f(cv::Mat &depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
	cout << "ok>" << endl;
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
	cout << "1 " << endl;
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
		cout << "3" << endl;
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        // cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}