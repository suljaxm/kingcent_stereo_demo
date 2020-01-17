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

		Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

		//-- And create the image in which we will save our disparities
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow, sgnm[3];
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		int pix = sgbmDisparityShow.at<Vec3b>(170, 150)[1];//
		double alpha = 0.35;
		double intercept = 24.75; //
		int distance = pix * alpha + intercept;//
		cout << distance << "cm" << endl;

		split(sgbmDisparityShow, sgnm);
		// imshow("sgbmDisparity", sgbmDisparityShow);//
		imshow("sgbm", sgnm[1]); // Grayscale display

		char c = (char)waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
	return 0;
}