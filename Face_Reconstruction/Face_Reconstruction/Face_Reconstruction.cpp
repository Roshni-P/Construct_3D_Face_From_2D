// Face_Reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//





#include <opencv2/opencv.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream> 
#include "FaceConstruction.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	FaceConstruction objFace;
	objFace.Reconstruct();
	waitKey(0);

	return 0;
}
