#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace std;
using namespace cv::xfeatures2d;

//Import of all CUDA and OpenCV Related Header Files

void showChoices(); //Declaration of Main Menu Function

// Function for SURF Object Detection and Matching

int Surf() {
	Mat h_object_image = imread("C:/Users/umash/OneDrive/Desktop/Imageres/object1.jpg", 0); // Object to be found
	Mat h_scene_image = imread("C:/Users/umash/OneDrive/Desktop/Imageres/scene1.jpg", 0); // Scene to be analyzed

	//Declaration of Image in Matrix form for GPU

	cuda::GpuMat d_object_image; 
	cuda::GpuMat d_scene_image;

	//Vectorizing the Keypoints in the scene

	cuda::GpuMat d_keypoints_scene, d_keypoints_object;
	vector< KeyPoint > h_keypoints_scene, h_keypoints_object;
	cuda::GpuMat d_descriptors_scene, d_descriptors_object;

	// Uploading image to the GPU Matrices

	d_object_image.upload(h_object_image);
	d_scene_image.upload(h_scene_image);

	//Time Initialization before SURF

	int64 work_begin = cv::getTickCount();

	// Calling the Surf function and begin matching of vector feature points
	cuda::SURF_CUDA surf(100);
	surf(d_object_image, cuda::GpuMat(), d_keypoints_object, d_descriptors_object);
	surf(d_scene_image, cuda::GpuMat(), d_keypoints_scene, d_descriptors_scene);

	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector< vector< DMatch> > d_matches;
	matcher->knnMatch(d_descriptors_object, d_descriptors_scene, d_matches, 2);


	surf.downloadKeypoints(d_keypoints_scene, h_keypoints_scene);
	surf.downloadKeypoints(d_keypoints_object, h_keypoints_object);


	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(h_keypoints_object.size() - 1, d_matches.size()); k++)
	{
		if ((d_matches[k][0].distance < 0.6 * (d_matches[k][1].distance)) &&
			((int)d_matches[k].size() <= 2 && (int)d_matches[k].size() > 0))
		{
			good_matches.push_back(d_matches[k][0]);
		}
	}
	std::cout << "size:" << good_matches.size();
	Mat h_image_result;
	drawMatches(h_object_image, h_keypoints_object, h_scene_image, h_keypoints_scene,
		good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::DEFAULT);


	std::vector<Point2f> object;
	std::vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
	}
	Mat Homo = findHomography(object, scene, RANSAC);
	std::vector<Point2f> corners(4);
	std::vector<Point2f> scene_corners(4);
	corners[0] = Point(0, 0);
	corners[1] = Point(h_object_image.cols, 0);
	corners[2] = Point(h_object_image.cols, h_object_image.rows);
	corners[3] = Point(0, h_object_image.rows);
	perspectiveTransform(corners, scene_corners, Homo);
	line(h_image_result, scene_corners[0] + Point2f(h_object_image.cols, 0), scene_corners[1] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[1] + Point2f(h_object_image.cols, 0), scene_corners[2] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[2] + Point2f(h_object_image.cols, 0), scene_corners[3] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	line(h_image_result, scene_corners[3] + Point2f(h_object_image.cols, 0), scene_corners[0] + Point2f(h_object_image.cols, 0), Scalar(255, 0, 0), 4);
	
	// End time after Surf
	int64 delta = cv::getTickCount() - work_begin;
	double freq = cv::getTickFrequency(); // Frequency Calculation
	double work_fps = freq / delta; //FPS Calculation
	imshow("Good Matches & Object detection", h_image_result); // Image Display
	std::cout << "Time: " << (1 / work_fps) << std::endl;
	std::cout << "FPS: " << work_fps << std::endl;
	waitKey(0);


	return 0;
}

// Foreground Movement Detection using Thresholding and MoG

int mog() {
	VideoCapture cap("C:/Users/umash/OneDrive/Desktop/Imageres/abc.avi"); // Video Initialization
	if (!cap.isOpened())
	{
		cerr << "can not open camera or video file" << endl;
		return -1;
	}
	Mat frame; // Declare frame as Matrix Input
	cap.read(frame);  // Read the frames from video
	GpuMat d_frame;
	d_frame.upload(frame);
	Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG(); //Declaration of Backround Subtractor MoG
	GpuMat d_fgmask, d_fgimage, d_bgimage; //Declaration of Masks
	Mat h_fgmask, h_fgimage, h_bgimage;
	mog->apply(d_frame, d_fgmask, 0.01); //Applying the Backround Masks
	while (1)
	{
		cap.read(frame);
		if (frame.empty())
			break;
		d_frame.upload(frame); // Loading the Video frames
		int64 start = cv::getTickCount(); //Begin start time
		mog->apply(d_frame, d_fgmask, 0.01); //Applying the Masks
		mog->getBackgroundImage(d_bgimage); //Retrieval of Backround Images
		
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start); // FPS Calculation per frame
		std::cout << "FPS : " << fps << std::endl;
		std::cout << "Time : " << 1/fps << std::endl;

		// Display of Image and Masks
		d_fgimage.create(d_frame.size(), d_frame.type());
		d_fgimage.setTo(Scalar::all(0));
		d_frame.copyTo(d_fgimage, d_fgmask);
		d_fgmask.download(h_fgmask);
		d_fgimage.download(h_fgimage);
		d_bgimage.download(h_bgimage);
		imshow("image", frame);
		imshow("foreground mask", h_fgmask);
		imshow("foreground image", h_fgimage);
		imshow("mean background image", h_bgimage);
		if (waitKey(1) == 'q')
			break;
	}

	return 0;
}

// FPS Comparison of GPU Thresholding for different thresholds on 4K Image
int GPU() {
	cv::Mat src = cv::imread("C:/Users/umash/OneDrive/Desktop/Imageres/6000x4000.jpg", 0); //Load Image
	cv::Mat result_host1, result_host2, result_host3, result_host4, result_host5;

	int64 work_begin = cv::getTickCount(); //Begin start time

	//Passing the image through 5 different types of Images
	cv::threshold(src, result_host1, 128.0, 255.0, cv::THRESH_BINARY);
	cv::threshold(src, result_host2, 128.0, 255.0, cv::THRESH_BINARY_INV);
	cv::threshold(src, result_host3, 128.0, 255.0, cv::THRESH_TRUNC);
	cv::threshold(src, result_host4, 128.0, 255.0, cv::THRESH_TOZERO);
	cv::threshold(src, result_host5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

	int64 delta = cv::getTickCount() - work_begin; // End time

	double freq = cv::getTickFrequency(); //Frequency
	double work_fps = freq / delta; // FPS Calculation
	std::cout << "Performance of Thresholding on GPU: " << std::endl;
	std::cout << "Time: " << (1 / work_fps) << std::endl;
	std::cout << "FPS: " << work_fps << std::endl;
	return 0;
}
 // FPS Comparison of CPU Thresholding
int CPU()
{
	cv::Mat h_img1 = cv::imread("C:/Users/umash/OneDrive/Desktop/Imageres/6000x4000.jpg", 0);
	cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_result5, d_img1;
	//Measure initial time ticks
	int64 work_begin = cv::getTickCount();
	d_img1.upload(h_img1);
	cv::cuda::threshold(d_img1, d_result1, 128.0, 255.0, cv::THRESH_BINARY);
	cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
	cv::cuda::threshold(d_img1, d_result3, 128.0, 255.0, cv::THRESH_TRUNC);
	cv::cuda::threshold(d_img1, d_result4, 128.0, 255.0, cv::THRESH_TOZERO);
	cv::cuda::threshold(d_img1, d_result5, 128.0, 255.0, cv::THRESH_TOZERO_INV);

	cv::Mat h_result1, h_result2, h_result3, h_result4, h_result5;
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	d_result3.download(h_result3);
	d_result4.download(h_result4);
	d_result5.download(h_result5);

	
	
	//Measure difference in time ticks
	int64 delta = cv::getTickCount() - work_begin;
	double freq = cv::getTickFrequency();
	//Measure frames per second
	double work_fps = freq / delta;
	std::cout << "Performance of Thresholding on CPU: " << std::endl;
	std::cout << "Time: " << (1 / work_fps) << std::endl;
	std::cout << "FPS: " << work_fps << std::endl;
	return 0;
}
// Implementation of Menu for the different CUDA Operations
int main()
{

	int choice;
	VideoCapture cap(0);
	do
	{
		showChoices();
		cin >> choice;
		switch (choice)
		{
			//Comparison of different image resolutions by passing through GPU Thresholding
		case 1:
			try
			{ 
				//Options for different Image Resolutions
				cout << "Enter Image Resolution:" << endl;
				cout << "640x480 (jpg)" << endl;
				cout << "1392x1024 (png)" << endl;
				cout << "1600x1200 (jpg)" << endl;
				cout << "2080x1542 (jpg)" << endl;
				cout << "6000x4000 4K (jpg)" << endl;

				std::string file;
				std::string path = "C:/Users/umash/OneDrive/Desktop/Imageres/";
				cin >> file;
				std::string filefin = path.append(file);
				cout << filefin;
				cv::Mat src_host = cv::imread(filefin, 0);
				cv::cuda::GpuMat dst, src;
				src.upload(src_host);
				//Start time before thresholding
				int64 work_begin = cv::getTickCount();
				cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
				int64 delta = cv::getTickCount() - work_begin; // End time

				double freq = cv::getTickFrequency();// Frequency
				double work_fps = freq / delta;// FPS

				cv::Mat result_host;
				dst.download(result_host);
				std::cout << "Performance of Thresholding on GPU: " << std::endl;
				std::cout << "Time: " << (1 / work_fps) << std::endl;
				std::cout << "FPS: " << work_fps << std::endl;
				cv::imshow("Result", result_host);
				cv::waitKey();
			}
			catch (const cv::Exception & ex)
			{
				std::cout << "Error: " << ex.what() << std::endl;
			}
			break;

			//Coloured Object Detection
		case 2:

			if (!cap.isOpened())
			{
				cout << "Cannot open the web cam" << endl;
				return -1;
			}
			while (true)
			{
				Mat frame;

				bool flag = cap.read(frame); // Reading of frames from Camera
				if (!flag)
				{
					cout << "Cannot read a frame from webcam" << endl;
					break;
				}

				// Variable declarations for Image Matrix and Threshold filters and BGR to HSV
				cuda::GpuMat d_frame, d_frame_hsv, d_intermediate, d_result;
				cuda::GpuMat d_frame_shsv[3];
				cuda::GpuMat d_thresc[3];
				Mat h_result;
				d_frame.upload(frame);

				int64 work_begin = cv::getTickCount(); //Start time
				cuda::cvtColor(d_frame, d_frame_hsv, COLOR_BGR2HSV);


				cuda::split(d_frame_hsv, d_frame_shsv);

				//Thresholding of Frames
				cuda::threshold(d_frame_shsv[0], d_thresc[0], 16, 45, THRESH_BINARY);
				cuda::threshold(d_frame_shsv[1], d_thresc[1], 50, 255, THRESH_BINARY);
				cuda::threshold(d_frame_shsv[2], d_thresc[2], 50, 255, THRESH_BINARY);

				//Bitwise And of Image and Threshold Filters
				cv::cuda::bitwise_and(d_thresc[0], d_thresc[1], d_intermediate);
				cv::cuda::bitwise_and(d_intermediate, d_thresc[2], d_result);
				int64 delta = cv::getTickCount() - work_begin;//End time
				double freq = cv::getTickFrequency();//Frequency
				double work_fps = freq / delta;// FPS

				d_result.download(h_result);
				cv::imshow("Thresholded Image", h_result);
				cv::imshow("Original", frame);
				std::cout << "Time: " << (1 / work_fps) << std::endl;
				std::cout << "FPS: " << work_fps << std::endl;

				if (waitKey(1) == 'q')
				{
					break;
				}
			}
			break;
		case 3:
			GPU();
			break;
		case 4:
			CPU();
			break;
		case 5:
			mog();
			break;

		case 6:
			Surf();
			break;

		case 7:
			break;
		default:
			cout << "Invalid input" << endl;
		}
	} while (choice != 5);

	return 0;
}

void showChoices() // Final Menu Display
{
	cout << "-------------------------------------------------" << endl;
	cout << "Image Processing Operations Using CUDA and OpenCV" << endl;
	cout << "-------------------------------------------------" << endl;
	cout << "1: Thresholding Comparison " << endl;
	cout << "2: Object Detection" << endl;
	cout << "3: GPU Performance" << endl;
	cout << "4: CPU Performance" << endl;
	cout << "5: Backgroud MoG " << endl;
	cout << "6: Surf Object " << endl;
	cout << "7: Exit " << endl;
	cout << "Enter your choice :";
}



