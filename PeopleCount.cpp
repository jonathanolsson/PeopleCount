#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cvstd.hpp"

#include <iostream>
#include <time.h>
#include <string>


void displayObjects(cv::UMat&, std::vector<cv::Rect>&, const int& r = 0, const cv::Scalar& colour = cv::Scalar(0,0,0));
void captureVideo(const std::string& filename = "");

const cv::String keys =
{
	"{help h | | print this message}"
	"{video v | | directory to video}"
	"{display d | | display videostream}"
};

bool DISPLAY = false;

/** @function main */
int main(int argc, char** argv)
{
	//Command line user interface
	cv::CommandLineParser parser(argc, argv, keys);
	
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	if (parser.has("display")) {
		std::cout << "Displaying video \n";
		DISPLAY = true;
	}

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	if (parser.has("v")) {

		cv::String filename = parser.get<std::string>("v");
		captureVideo(filename);
	}
	else {
		captureVideo();
	}



	return 0;
}

//Display the faces of a frame
void displayObjects(cv::UMat& frame, std::vector<cv::Rect>& bodies, const cv::Scalar& colour) {
	for (size_t i = 0; i < bodies.size(); i++) {
		cv::Point p1(bodies[i].x, bodies[i].y);
		cv::Point p2(bodies[i].x + bodies[i].width, bodies[i].y + bodies[i].height);

		rectangle(frame, p1, p2, colour, 4, 8, 0);
		}

	cv::namedWindow(("Window"), cv::WINDOW_AUTOSIZE);
	imshow(("Window"), frame);

}

/*
Realy good for two seconds occurrences.
cv::VideoCapture capture("../../../video/2.mp4");
firstCascade.load("../../data/haarcascades/haarcascade_mcs_upperbody.xml");
firstCascade.detectMultiScale(frame, bodies, 1.1, 3, 0, cv::Size(48, 48));

*/
//Capture video from webcamera
void captureVideo(const std::string& filename) {
	//Video capture
	//cv::VideoCapture capture("../../../video/2.mp4");
	cv::VideoCapture capture;

	if (filename.empty()) {
		capture.open(0);
	}
	else {
		capture.open(filename);
	}

	cv::UMat frame;

	cv::CascadeClassifier firstCascade;
	cv::CascadeClassifier secondCascade;

	firstCascade.load("../../data/haarcascades/haarcascade_upperbody.xml");
	//secondCascade.load("../../data/haarcascades/haarcascade_frontalface_alt2.xml");

	//secondCascade.load("../../data/haarcascades/haarcascade_upperbody.xml");
	//secondCascade.load("../../data/lbpcascades/lbpcascade_profileface.xml");

	//firstCascade.load("../../data/haarcascades/haarcascade_frontalface_alt2.xml");
	secondCascade.load("../../data/lbpcascades/lbpcascade_frontalface.xml");

	//firstCascade.load("../../data/case.xml");

	//Vector of bodies.
	std::vector<cv::Rect> bodies1;
	std::vector<cv::Rect> bodies2;
	std::vector<cv::Rect> Total;

	//Something to make the counter accurate
	int counter = 0;
	int occurrences = 10;
	int lastNum = 0;
	int amount = 0;

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
				//cv::resize(frame, frame, cv::Size(1024, 576));
				//*/
				cv::resize(frame, frame, cv::Size(640, 360));
				
				//Object detection
				//firstCascade.detectMultiScale(frame, bodies, 1.1, 3, 0, cv::Size(55, 50));
				//firstCascade.detectMultiScale(frame, bodies1, 1.05, 1, 0, cv::Size(88, 80), cv::Size(300, 300));
				//firstCascade.detectMultiScale(frame, bodies1, 1.1, 2, 0, cv::Size(60, 60), cv::Size(150, 150));
				
				//Description:	Upperbody detection is primarily for pedestrians. These parameters (frame, bodies1, 1.005, 1, 0, cv::Size(77, 70), cv::Size(231, 210)) sets
				//				scale to very low, to detect object even if they are close to each other in the image
				//				min neighbor to 1. This as the cascade is accurate and do not produce a vast amount of false positives. A 2 will make the classifier find a lower number of possitives.
				//				0 flags
				//				MinSize set quite high, to make the process faster it is asumed that no object smaller than 77, 70 exists in the feed. These will be ignored in that case.
				//				MaxSize set to 231, 210 as object larger than this should not be counted.
				//
				//Motivation:	These parameters will give good result as the scaling steps are small. This to detect as many object as possible. After testing to set the minSize to a smaller value, the relizesation that this is unimportant occured
				//				as this valut describes the minimum size that should be detected. A smaller value would detect smaller object, not make the function more efficent and accurate. The scale, however, does this. 
				firstCascade.detectMultiScale(frame, bodies1, 1.05, 1, 0, cv::Size(77, 70), cv::Size(231, 210));

				secondCascade.detectMultiScale(frame, bodies2, 1.1, 2, 0 , cv::Size(30, 30));

				if (DISPLAY) {
					//Put info on frame(Frames processed per second and the occupancy)
					cv::putText(frame, "FPS: " + std::to_string(fps), cvPoint(30, 30), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
					cv::putText(frame, "Persons: " + std::to_string(lastNum), cvPoint(30, 55), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

					//Display objects
					displayObjects(frame, bodies1, cv::Scalar(100, 200, 0));
					displayObjects(frame, bodies2, cv::Scalar(255, 255, 255));

				}

			}
			else {
				printf(" --(!) No captured frame -- Break! Or no more video...");
				break;
			}

			if (!bodies1.empty()) {
				std::cout << "x: " << bodies1[0].x << "y: " << bodies1[0].y << "width: " << bodies1[0].width << "height: " << bodies1[0].height << std::endl;
			}
			/*
			//Emense loop to stop repeated counting.
			for (int i = 0; i < bodies1.size(); i++) {
				for (int j = 0; j < bodies2.size(); j++) {
					if ((bodies1[i].x < bodies2[j].x < bodies1[i].x + bodies1[i].width) ||
						(bodies1[i].y < bodies2[j].y < bodies1[i].y + bodies1[i].height) ||
						(bodies1[i].x < bodies2[j].x + bodies2[j].width < bodies1[i].x + bodies1[i].width) ||
						(bodies1[i].x < bodies2[j].y + bodies2[j].height < bodies1[i].x + bodies1[i].width)) {
						//Do not count.

					}
					else {
						//Do count
						amount++;
					}

				}
			}
			*/

			//Change the number of people in the image if it is consistant for some iterations.
			if (bodies1.size() + bodies2.size() != lastNum) {
				counter++;
				if (counter == occurrences) {
					counter = 0;
					occurrences = fps*2;
					lastNum = bodies1.size() + bodies2.size();
					
					std::cout << lastNum/2 << std::endl;
				}
			}

			int c = cv::waitKey(15);
			if ((char)c == 'c') {
				break;
			}
		}

		std::cout << "Closing the camera" << std::endl;
		capture.release();
		cv::destroyAllWindows();
	}
	return;
}