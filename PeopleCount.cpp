#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>


void displayFaces(cv::UMat, std::vector<cv::Rect>, std::string);
void captureVideo();

/** @function main */
int main(int argc, char** argv)
{
	captureVideo();

	return 0;
}

//Display the faces of a frame
void displayFaces(cv::UMat frame, std::vector<cv::Rect> bodies, std::string windowName = "Window") {
	for (size_t i = 0; i < bodies.size(); i++) {
		cv::Point p1(bodies[i].x, bodies[i].y);
		cv::Point p2(bodies[i].x + bodies[i].width, bodies[i].y + bodies[i].height);

		rectangle(frame, p1, p2, cv::Scalar(255, 255, 255), 4, 8, 0);
		}

	cv::namedWindow((windowName), cv::WINDOW_AUTOSIZE);
	imshow((windowName), frame);

}


//Capture video from webcamera
void captureVideo() {
	//Video capture
	cv::VideoCapture capture(0);
	cv::UMat frame;

	cv::CascadeClassifier cascade;
	cascade.load("data/haarcascades/haarcascade_upperbody.xml");

	//Vector of bodies.
	std::vector<cv::Rect> bodies;

	//Something to make the counter accurate
	int counter = 0;
	int occurrences = 15;
	int lastNum = 0;

	if (capture.isOpened()) {
		while (true) {
			capture.read(frame);

			//Apply the wanted classifier to the frame
			if (!frame.empty()) {
				/*
				cv::cvtColor(frame, frame, CV_BGR2GRAY);
				cv::equalizeHist(frame, frame);
				*/

				cascade.detectMultiScale(frame, bodies, 1.1, 2, 0, cv::Size(40, 40));
				
				displayFaces(frame, bodies);
			}
			else {
				printf(" --(!) No captured frame -- Break!"); break;
			}

			//Change the number of people in the image if it is consistant for some iterations.
			if (bodies.size() != lastNum) {
				counter++;
				if (counter == occurrences) {
					counter = 0;
					lastNum = bodies.size();

					std::cout << bodies.size() << std::endl;
				}
			}

			int c = cv::waitKey(15);
			if ((char)c == 'c') {
				break;
			}
		}
	}

	std::cout << "Closing the camera" << std::endl;
	capture.release();
	cv::destroyAllWindows();
}
