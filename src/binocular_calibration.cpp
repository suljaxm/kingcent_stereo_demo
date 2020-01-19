// create by yjx 2020/01/11

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
using namespace std;
using namespace cv;

vector<vector<Point2f> > corners_l_array, corners_r_array;

int array_index = 0;

bool ChessboardStable(vector<Point2f>corners_l, vector<Point2f>corners_r);

int main(){

	Mat frame;
	Mat image_left, image_right, frame_l, frame_r;
	VideoCapture cap(1);

	/*
	KRX*/
	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	cap.set(CV_CAP_PROP_FPS, 60);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frameRate = cap.get(CV_CAP_PROP_FPS);
	int totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);

	// Create a Rect box, which belongs to the class in "cv". 
	// The four parameters are x, y, width, height. 
	Rect left_rect(0, 0, width / 2, height);  
	Rect right_rect(width / 2, 0, width / 2, height);

	cout << "width=" << width << endl;
	cout << "height=" << height << endl;
	cout << "totalFrames=" << totalFrames << endl;
	cout << "frameRate=" << frameRate << endl;
	
	if (!cap.isOpened())
		exit(0);

	Size boardSize(9, 6);
	const float squareSize = 26.f;  //26mm

	vector<vector<Point2f> > imagePoints_l;
	vector<vector<Point2f> > imagePoints_r;

	int nimages = 0;
	int img_num = 60;
	while (true)
	{
		cap >> frame;  
		image_left = Mat(frame, left_rect).clone();
		image_right = Mat(frame, right_rect).clone();

		bool found_l = false, found_r = false;
		vector<Point2f> corners_l, corners_r;

		found_l = findChessboardCorners(image_left, boardSize, corners_l, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		found_r = findChessboardCorners(image_right, boardSize, corners_r, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found_l && found_r && ChessboardStable(corners_l, corners_r)) 
		{
			Mat viewGray;
			cvtColor(image_left, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, corners_l, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			cvtColor(image_right, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, corners_r, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

			imagePoints_l.push_back(corners_l);
			imagePoints_r.push_back(corners_r);
			++nimages;
			image_left += 100;
			image_right += 100;

			drawChessboardCorners(image_left, boardSize, corners_l, found_l);
			drawChessboardCorners(image_right, boardSize, corners_r, found_r);

			imshow("Left Camera", image_left);
			imshow("Right Camera", image_right);

			char c = (char)waitKey(500);
			if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
				exit(-1);

			if (nimages >= 30)
				break;
		}
		else
		{
			drawChessboardCorners(image_left, boardSize, corners_l, found_l);
			drawChessboardCorners(image_right, boardSize, corners_r, found_r);

			//putText(image_left, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));  //show poind id in images
			//putText(image_right, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));
			imshow("Left Camera", image_left);
			imshow("Right Camera", image_right);
			//cout << image_left.size() << endl;
			
			//string image_name = to_string(img_num) +  ".png";
			//FILE* points_txt = fopen(point_path.c_str(), "w");  //save chessboardpoints

			char key = waitKey(1);
			if (key == 27 || key == 'q' || key == 'Q') //Allow ESC to quit
				break;
			if (key == 's' || key == 'S') //Allow S to save data
			{
				//imwrite(image_name, frame);
				//img_num = img_num + 20;
			}
		}
	}
	if (nimages < 20) { cout << "Not enough" << endl; return -1; }

	vector<vector<Point2f> > imagePoints[2] = { imagePoints_l, imagePoints_r };
	vector<vector<Point3f> > objectPoints;
	objectPoints.resize(nimages);

	for (int i = 0; i < nimages; i++)
	{
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}
	cout << "Running stereo calibration ..." << endl;

	//Size imageSize(320, 240);
	Size imageSize(width / 2, height);
	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints_l, imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints_r, imageSize, 0);

	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imagePoints_l, imagePoints_r,
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F, 
		CV_CALIB_FIX_ASPECT_RATIO + 
		CV_CALIB_ZERO_TANGENT_DIST + 
		CV_CALIB_USE_INTRINSIC_GUESS + 
		CV_CALIB_SAME_FOCAL_LENGTH + 
		CV_CALIB_RATIONAL_MODEL +
		CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	cout << "done with RMS error=" << rms << endl;

	double err = 0;
	int npoints = 0;

	//Calculate the polar vector
	vector<Vec3f> lines[2]; //polar
	for (int i = 0; i < nimages; i++)
	{

		int npt = (int)imagePoints_l[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(imagePoints_l[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]);
		computeCorrespondEpilines(imgpt[0], 0 + 1, F, lines[0]);

		imgpt[1] = Mat(imagePoints_r[i]); 
		undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]); 
		computeCorrespondEpilines(imgpt[1], 1 + 1, F, lines[1]); 

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;

	// save intrinsics
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);  
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";


	Mat R1, R2, P1, P2, Q; // outside parameter
	Rect validRoi[2];

	stereoRectify(
		cameraMatrix[0],  // internal parameter matrix
		distCoeffs[0],    // distortion parameter
		cameraMatrix[1], 
		distCoeffs[1], 
		imageSize,  //image size
		R,  // Rotation matrix
		T,  // Translation matrix
		R1, // left rotaion correction parameters
		R2, // right rotaion correction parameters
		P1, // left translation correction parameters
		P2, // right translation correction parameters
		Q,  // depth correction parameters
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]); 

	// save extrinsics
	fs.open("extrinsics.yml", FileStorage::WRITE); // outside parameter file
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

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

	destroyAllWindows();

	Mat imgLeft, imgRight;

	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(0);
	//sbm->setNumDisparities(64);
	sbm->setTextureThreshold(10);
	sbm->setDisp12MaxDiff(-1);
	sbm->setPreFilterCap(31);
	sbm->setUniquenessRatio(25);
	sbm->setSpeckleRange(32);
	sbm->setSpeckleWindowSize(100);


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);


	Mat rimg, cimg;
	Mat Mask;
	while (1)
	{
		cap >> frame;
		frame_l = Mat(frame, left_rect).clone();
		frame_r = Mat(frame, right_rect).clone();

		if (frame_l.empty() || frame_r.empty())
			continue;

		remap(frame_l, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_r, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;

		imgLeft = canvasPart1(vroi).clone();
		imgRight = canvasPart2(vroi).clone();

		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);


		cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
		cvtColor(imgRight, imgRight, CV_BGR2GRAY);


		//-- And create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl; return -1;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);


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
		imshow("bmDisparity", disparityShow);//
		imshow("sgbmDisparity", sgbmDisparityShow);//
		imshow("sgbm", sgnm[1]); // Grayscale display
		imshow("rectified", canvas);//
		char c = (char)waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}


	return 0;
}

bool ChessboardStable(vector<Point2f>corners_l, vector<Point2f>corners_r) {
	if (corners_l_array.size() < 10) {
		corners_l_array.push_back(corners_l);
		corners_r_array.push_back(corners_r);
		return false;
	}
	else {
		corners_l_array[array_index % 10] = corners_l;
		corners_r_array[array_index % 10] = corners_r;
		array_index++;
		double error = 0.0;
		for (int i = 0; i < corners_l_array.size(); i++) {
			for (int j = 0; j < corners_l_array[i].size(); j++) {
				error += abs(corners_l[j].x - corners_l_array[i][j].x) + abs(corners_l[j].y - corners_l_array[i][j].y);
				error += abs(corners_r[j].x - corners_r_array[i][j].x) + abs(corners_r[j].y - corners_r_array[i][j].y);
			}
		}
		cout << "error= " << error << endl;
		if (error < 1000)
		{
			corners_l_array.clear();
			corners_r_array.clear();
			array_index = 0;
			return true;
		}
		else
			return false;
	}
}