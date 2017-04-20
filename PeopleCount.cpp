#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cvstd.hpp"

#include <iostream>
#include <time.h>
#include <string>


void displayObjects(cv::UMat&, std::vector<cv::Rect>&);
void captureVideo();

/** @function main */
int main(int argc, char** argv)
{
	for (int i = 0; i < 10; i++) {
		captureVideo();
	}

	return 0;
}

//Display the faces of a frame
void displayObjects(cv::UMat& frame, std::vector<cv::Rect>& bodies) {
	for (size_t i = 0; i < bodies.size(); i++) {
		cv::Point p1(bodies[i].x, bodies[i].y);
		cv::Point p2(bodies[i].x + bodies[i].width, bodies[i].y + bodies[i].height);

		rectangle(frame, p1, p2, cv::Scalar(255, 255, 255), 4, 8, 0);
		}

	cv::namedWindow(("Window"), cv::WINDOW_AUTOSIZE);
	imshow(("Window"), frame);

}

/*
Realy good for two seconds occurrences.
cv::VideoCapture capture("../../../video/2.mp4");
cascade.load("../../data/haarcascades/haarcascade_mcs_upperbody.xml");
cascade.detectMultiScale(frame, bodies, 1.1, 3, 0, cv::Size(48, 48));

*/
//Capture video from webcamera
void captureVideo() {
	//Video capture
	//cv::VideoCapture capture("../../../video/2.mp4");
	cv::VideoCapture capture("../../../video/2.mp4");

	cv::UMat frame;

	cv::CascadeClassifier cascade;
	
	cascade.load("../../data/haarcascades/haarcascade_mcs_upperbody.xml");
	//cascade.load("../../data/haarcascades/haarcascade_frontalface_alt2.xml");
	//cascade.load("../../data/lbpcascades/lbpcascade_frontalface.xml");

	//cascade.load("../../data/case.xml");

	//Vector of bodies.
	std::vector<cv::Rect> bodies;

	//Something to make the counter accurate
	int counter = 0;
	int occurrences = 10;
	int lastNum = 0;


	if (capture.isOpened()) {
		int iterations = 0;
		int fps = 0;
		
		time_t second;
		time(&second);

		while (true) {
			if (time(time_t()) > second) {
				fps = iterations;
				std::cout << fps << " fps" << std::endl;
				
				iterations = 0;
				time(&second);
			}

			//count the iterations. (Amount of frames)
			iterations++;

			//Load captured frame into "frame"
			capture.read(frame);

			//Apply the wanted classifier to the frame
			if (!frame.empty()) {
				
				//Color modification(Make grayscale and equalize on histogram)
				//cv::cvtColor(frame, frame, CV_BGR2GRAY);
				//cv::equalizeHist(frame, frame);
				
				//Resize the frame, for lighter processing.
				/*
				cv::resize(frame, frame, cv::Size(640, 480));
				/*/
				cv::resize(frame, frame, cv::Size(1024, 576));
				//*/
				
				
				//Object detection
				cascade.detectMultiScale(frame, bodies, 1.1, 3, 0, cv::Size(55, 50));
				cv::putText(frame, "FPS: " + std::to_string(fps), cvPoint(30,30), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
				cv::putText(frame, "Persons: " + std::to_string(lastNum), cvPoint(30, 55), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

				//Display objects
				displayObjects(frame, bodies);
			}
			else {
				printf(" --(!) No captured frame -- Break! Or no more video...");
				break;
			}

			//Change the number of people in the image if it is consistant for some iterations.
			if (bodies.size() != lastNum) {
				counter++;
				if (counter == occurrences) {
					counter = 0;
					occurrences = fps*2;
					lastNum = bodies.size();
					
					std::cout << lastNum << std::endl;
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