#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>



int main(void){
	// création d'une matrice(col,ligne,pixeltype(unsigned 8 bits 3 channels), valeur initiale)
	cv::Mat img(512,512, CV_8UC3, cv::Scalar(0));

	cv::putText(img,						// 
				"Hello World!", 
				cv::Point(10,img.rows/2), 
				cv::FONT_HERSHEY_DUPLEX, 
				1.0, 
				CV_RGB(0,255,255), 
				2);
	cv::imshow("Hello!", img);
	cv::waitKey();




	return 0;
}
