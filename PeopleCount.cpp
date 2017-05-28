#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cvstd.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <ctime>
#include <string>

//Function declerations
void displayObjects(cv::UMat&, std::vector<cv::Rect>&, const cv::Scalar& colour = cv::Scalar(0,0,0));

void findArguments(const std::string& filename);
void testFPS(const std::string& filename);

void captureVideo(const std::string& filename = "");
void captureVideoTestFPS(const std::string& filename, std::ofstream& outFile, int& minNeighbour, double& scale, cv::Size& minSize, const std::string& cascadeName);
void captureVideoFindArguments(const std::string& filename, std::ofstream& outFile, int& minNeighbour, double& scale, cv::Size& minSize, const std::string& cascadeName);


//Keys for the interface(What arguments to listen for)
const cv::String keys =
{
	"{help h | | print this message}"
	"{video v | | directory to video, used as -v=<video-file>}"
	"{display d | | display the processed video-stream}"
	"{findArgs a| | find best arguments for \"detectMultiscale\" function}"
	"{FPS f | | run FPS test on the application. The result is saved in /fps}"
};

bool DISPLAY = false;

/** @function main */
int main(int argc, char** argv)
{
	//Command line user interface
	cv::CommandLineParser parser(argc, argv, keys);
	
	//If the commandline has "help", print the help.
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	//If the commandline has "display", show the video.
	if (parser.has("display")) {
		std::cout << "Displaying video \n";
		DISPLAY = true;
	}

	//If the commandline have errors.
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	//If the commandline has "v", run program on video.
	if (parser.has("video")) {
		if (parser.has("findArgs")){
			std::cout << "run test...\n";
			cv::String filename = parser.get<std::string>("video");
			findArguments(filename);
		}
		if (parser.has("FPS")) {
			std::cout << "run test for fps...\n";
			cv::String filename = parser.get<std::string>("video");
			testFPS(filename);
		}
		else {
			cv::String filename = parser.get<std::string>("video");
			captureVideo(filename);
		}
	}
	else {
		captureVideo();
	}

	return 0;
}

//Function to iterate through a couple of settings to find the most accurate one.
void findArguments(const std::string& filename) {
	std::ofstream haarFront;
	haarFront.open("haar_front.txt", std::ofstream::app);
	
	std::ofstream lbpFront;
	lbpFront.open("lbp_front.txt", std::ofstream::app);
	
	std::ofstream haarUpperbody;
	haarUpperbody.open("haar_upperbody.txt", std::ofstream::app);
	
	std::ofstream haarUpperbodyMcs;
	haarUpperbodyMcs.open("haar_upperbody_mcs.txt", std::ofstream::app);
	
	//MinNeighbour
	for (int minNeighbour = 1; minNeighbour < 4; minNeighbour++) {
		//Scale
		for (double scale = 1.05; scale <= 1.1; scale = scale + 0.01) {
			//MinSize(Start from 35,35 as no object smaller than that should be counted and up to 80,80 as the smallest object should not be larger than that.)
			for (cv::Size minSize = cv::Size(25, 25); minSize.width < 50; minSize.width++ && minSize.height++) {
				std::cout << "run test haarFront \n" << minNeighbour << "\t" << scale  << "\t" << minSize << "\n";
				captureVideoFindArguments(filename, haarFront, minNeighbour, scale, minSize, "../../data/haarcascades/haarcascade_frontalface_alt2.xml");
				
				std::cout << "run test lbpFront\n" << minNeighbour << "\t" << scale << "\t" << minSize << "\n";
				captureVideoFindArguments(filename, lbpFront, minNeighbour, scale, minSize, "../../data/lbpcascades/lbpcascade_frontalface.xml");
				
				std::cout << "run test haarUpperbody\n" << minNeighbour << "\t" << scale << "\t" << minSize << "\n";
				captureVideoFindArguments(filename, haarUpperbody, minNeighbour, scale, minSize, "../../data/haarcascades/haarcascade_upperbody.xml");
				
				std::cout << "run test haarUpperbodyMcs\n" << minNeighbour << "\t" << scale << "\t" << minSize << "\n";
				captureVideoFindArguments(filename, haarUpperbodyMcs, minNeighbour, scale, minSize, "../../data/haarcascades/haarcascade_mcs_upperbody.xml");
			}
		}
	}

	haarFront.close();
	lbpFront.close();
	haarUpperbody.close();
	haarUpperbodyMcs.close();
}

//Test the FPS of all the models, with their optimal arguments.
//In the case of FPS = 1, the true FPS can differ. That is a limitation in the loop as the fps is measured based on number of frames on one second. However, the loop contains a calculation that takes more than a second and therefore the fps is calculated wrong.
void testFPS(const std::string& filename) {
	//Initial values..
	int minNeighbour = 1;
	double scale = 1.1;
	cv::Size minSize = cv::Size(33,33);

	std::ofstream haarFront;
	haarFront.open("fps/haar_front.txt");

	std::cout << "run fps-test haarFront \n";
	captureVideoTestFPS(filename, haarFront, minNeighbour, scale, minSize, "../../data/haarcascades/haarcascade_frontalface_alt2.xml");
	haarFront.close();
	

	std::ofstream lbpFront;
	lbpFront.open("fps/lbp_front.txt");
	
	std::cout << "run fps-test lbpFront \n";
	captureVideoTestFPS(filename, lbpFront, minNeighbour = 3, scale = 1.1, minSize = cv::Size(28, 28), "../../data/lbpcascades/lbpcascade_frontalface.xml");
	lbpFront.close();

	std::ofstream haarUpperbody;
	haarUpperbody.open("fps/haar_upperbody.txt");

	std::cout << "run fps-test haarUpperbody \n";
	captureVideoTestFPS(filename, haarUpperbody, minNeighbour = 1, scale = 1.09, minSize = cv::Size(29, 29), "../../data/haarcascades/haarcascade_upperbody.xml");
	haarUpperbody.close();


	std::ofstream haarUpperbodyMcs;
	haarUpperbodyMcs.open("fps/haar_upperbody_mcs.txt");

	std::cout << "run fps-test haarUpperbodyMcs \n";
	captureVideoTestFPS(filename, haarUpperbodyMcs, minNeighbour = 1, scale = 1.08, minSize = cv::Size(47, 47), "../../data/haarcascades/haarcascade_mcs_upperbody.xml");
	haarUpperbodyMcs.close();
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
//Capture video from webcamera or video.
void captureVideo(const std::string& filename) {
	
	//Video capture
	cv::VideoCapture capture(filename);
	
	if (filename.empty()) {
		capture.open(0);
	}
	else {
		capture.open(filename);
	}

	cv::UMat frame;

	cv::CascadeClassifier cascade;
	cascade.load("../../data/haarcascades/haarcascade_frontalface_alt2.xml");
	//cascade.load("../../data/haarcascades/haarcascade_mcs_upperbody.xml");

	//Vector of bodies.
	std::vector<cv::Rect> bodies;

	if (capture.isOpened()) {
		// Variables for the timer. Counter is the amount of iterations done where every number is new. occurences is the counted amount of people in the frame. 
		int counter = 0;
		int occurrences = 0;

		double fps = 0;
		int totalIterations = 0;

		std::clock_t start;
		std::clock_t totalTime = std::clock();

		while (true) {
			start = std::clock();

			totalIterations++;

			//Load captured frame into "frame"
			capture.read(frame);

			//Apply the wanted classifier to the frame
			if (!frame.empty()) {
				//Detect objects in frame
				//Upperbody:
				//cascade.detectMultiScale(frame, bodies, 1.08, 1, 0, cv::Size(47, 47), cv::Size(250, 250));

				//HaarFace:
				cascade.detectMultiScale(frame, bodies, 1.1, 1, 0, cv::Size(33, 33), cv::Size(200, 200));
				if (DISPLAY) {
					//Put info on frame(Frames processed per second and the occupancy)
					cv::putText(frame, "FPS: " + std::to_string(fps), cvPoint(30, 30), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
					cv::putText(frame, "Persons: " + std::to_string(occurrences), cvPoint(30, 55), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
					cv::putText(frame, "Frame: " + std::to_string(totalIterations), cvPoint(30, 80), CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
					
					//Display objects
					displayObjects(frame, bodies, cv::Scalar(100, 200, 0));
				}
			}
			else {
				std::cout << " --(!) No more captured frame -- Break! \n";
				break;
			}
			//Change the number of people in the image if it is consistant for two seconds of iterations.
			if (bodies.size() != occurrences) {
				counter++;
				if (counter >= fps * 2) {
					counter = 0;
					occurrences = bodies.size();
					std::cout << "Occurences: " << occurrences << " People\n";
					std::cout << "Time: " <<  (std::clock() - totalTime) / (int)CLOCKS_PER_SEC << " Seconds in\n";
					std::cout << fps << " fps\n\n";
				}
			}
			fps = 1.0 / ((std::clock() - start) / (double)CLOCKS_PER_SEC);
			
			int c = cv::waitKey(15);
			if ((char)c == 'c') {
				break;
			}
		}
		std::cout << "\nTotal Time:\n" << (std::clock() - totalTime) / (int)CLOCKS_PER_SEC;
		std::cout << "\nTotal Frames:\n" << totalIterations;

		std::cout << "Closing the capture" << std::endl;
		capture.release();

		cv::destroyAllWindows();
	}
	return;
}

//Capture video for testing the FPS.
void captureVideoTestFPS(const std::string& filename, std::ofstream& outFile, int& minNeighbour, double& scale, cv::Size& minSize, const std::string& cascadeName) {
	//Video capture
	//cv::VideoCapture capture("../../../video/2.mp4");
	cv::VideoCapture capture(filename);
	cv::UMat frame;

	cv::CascadeClassifier cascade;
	cascade.load(cascadeName);

	//Vector of bodies.
	std::vector<cv::Rect> bodies;

	if (capture.isOpened()) {
		outFile << "New fpstest\n";

		// Variables for the timer. Counter is the amount of iterations done where every number is new. occurences is the counted amount of people in the frame. 
		int counter = 0;
		int occurrences = 0;

		double fps = 0;
		int totalIterations = 0; 

		std::clock_t start;
		std::clock_t totalTime = std::clock();

		while (true) {
			start = std::clock();

			totalIterations++;

			//Load captured frame into "frame"
			capture.read(frame);

			//Apply the wanted classifier to the frame
			if (!frame.empty()) {
				cascade.detectMultiScale(frame, bodies, scale, minNeighbour, 0, minSize, cv::Size(250, 250));

			}
			else {
				std::cout << " --(!) No more captured frame -- Break! \n";
				break;
			}
			//Change the number of people in the image if it is consistant for two seconds of iterations.
			if (bodies.size() != occurrences) {
				counter++;
				if (counter >= fps * 2) {
					counter = 0;
					occurrences = bodies.size();
				}
			}
			fps = 1.0 / ((std::clock() - start) / (double)CLOCKS_PER_SEC);
			outFile << fps << "\n";

		}
		outFile << "\n" << "Total Time:\n" << (std::clock() - totalTime) / (int)CLOCKS_PER_SEC;
		outFile << "\n" << "Total Frames:\n" << totalIterations;

		std::cout << "Closing the capture" << std::endl;
		capture.release();
	}
	return;
}

//Capture video for finding arguments. 
void captureVideoFindArguments(const std::string& filename, std::ofstream& outFile, int& minNeighbour, double& scale, cv::Size& minSize, const std::string& cascadeName) {
	//Video capture
	//cv::VideoCapture capture("../../../video/2.mp4");
	cv::VideoCapture capture(filename);
	cv::UMat frame;

	cv::CascadeClassifier cascade;
	cascade.load(cascadeName);

	outFile << "New arguments:\t" << minNeighbour << "\t" << scale << "\t" << minSize << "\n";

	//Vector of bodies.
	std::vector<cv::Rect> bodies;

	if (capture.isOpened()) {
		//Total amount of iterations.
		int totalIteration = 0;

		// Variables for the timer. Counter is the amount of iterations done where every number is new. occurences is the counted amount of people in the frame. 
		int counter = 0;
		int occurrences = 0;
		
		//Both iteration and fps is needed as fps only will be changed once a second. This means that the value is quite static and can be compared. 
		int fpsIteration = 0;
		int fps = 0;

		std::clock_t start;
		start = std::clock();

		int lastSecond = 0;
		int time = 0;

		while (true) {
			int time = (std::clock() - start) / (int)CLOCKS_PER_SEC;
			totalIteration++;

			//On each second, write the current fps to prompt.
			if (time > lastSecond) {
				fps = fpsIteration;
				std::cout << fps << " fps" << std::endl;
				
				fpsIteration = 0;
				lastSecond = time;
			}

			//On each fifth frame(or iteration), print amount of people to file.(this will give numbers that will be easy to compare, as it will allways be the same frame on the same iteration.)
			if (totalIteration % 5 == 0) {
				outFile << totalIteration << "\t" << occurrences << "\n";
			}

			//count the iterations. (Amount of frames)
			fpsIteration++;

			//Load captured frame into "frame"
			capture.read(frame);

			//Apply the wanted classifier to the frame
			if (!frame.empty()) {

				//Color modification(Make grayscale and equalize on histogram)
				//cv::cvtColor(frame, frame, CV_BGR2GRAY);
				//cv::equalizeHist(frame, frame);
				//cv::resize(frame, frame, cv::Size(640, 360));
				cascade.detectMultiScale(frame, bodies, scale, minNeighbour, 0, minSize, cv::Size(250, 250));

				//secondCascade.detectMultiScale(frame, bodies2, 1.1, 2, 0 , cv::Size(30, 30));
			}
			else {
				std::cout << " --(!) No more captured frame -- Break! \n";
				break;
			}
			//Change the number of people in the image if it is consistant for two seconds of iterations.
			if (bodies.size() != occurrences) {
				counter++;
				if (counter >= fps * 2) {
					counter = 0;
					occurrences = bodies.size();
				}
			}

			if (10 < time && fps < 8) {
				outFile << "Too Slow..\n";
				std::cout << "Too Slow..\n";
				
				capture.release();
				return;
			}

		}
		std::cout << "Closing the capture" << std::endl;
		capture.release();
	}
	return;
}