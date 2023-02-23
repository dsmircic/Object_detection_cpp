#pragma once
#ifndef MYLIB_CONSTANTS_H
#define MYLIB_CONSTANTS_H
#include <opencv2/opencv.hpp>

namespace font
{
	const float FONT_SCALE = 0.7;
	const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
	const int THICKNESS = 0.5;
}

namespace detection_constants
{
	const float INPUT_WIDTH = 640.0;
	const float INPUT_HEIGHT = 640.0;
	const float SCORE_THRESHOLD = 0.5;
	const float NMS_THRESHOLD = 0.45;
	const float CONFIDENCE_THRESHOLD = 0.45;
}

namespace color
{
	cv::Scalar BLACK = cv::Scalar(117, 117, 117);
	cv::Scalar BLUE = cv::Scalar(255, 178, 50);
	cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
	cv::Scalar RED = cv::Scalar(0, 0, 255);
}

namespace base
{
	const int base = 1;
}

#endif