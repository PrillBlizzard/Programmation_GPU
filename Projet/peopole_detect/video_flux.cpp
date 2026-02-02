#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <iomanip>

 #include <vpi/OpenCVInterop.hpp>
  
 #include <vpi/Array.h>
 #include <vpi/Image.h>
 #include <vpi/ImageFormat.h>
 #include <vpi/Pyramid.h>
 #include <vpi/Status.h>
 #include <vpi/Stream.h>
 #include <vpi/algo/BackgroundSubtractor.h>
 #include <vpi/algo/ConvertImageFormat.h>

#define CHECK_STATUS(STMT)                                    \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);

class Detector
{
    enum Mode { Default, Daimler } m;
    cv::HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    std::string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    std::vector<cv::Rect> detect(cv::InputArray img)
    {
        // Run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        std::vector<cv::Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(), 1.05, 2, true);
        return found;
    }
    void adjustRect(cv::Rect & r) const
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};

int main()
{
    // Ouvre la caméra par défaut (souvent /dev/video0)
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Vérifie que la caméra est bien ouverte
    if (!cap.isOpened())
    {
        std::cerr << "Erreur : impossible d'ouvrir la caméra" << std::endl;
        return -1;
    }

// ---------- Function with VPI ---------- //

    //créer le stream pour process (ok)
    VPIBackend backend;
    backend = VPI_BACKEND_CUDA;
    VPIStream stream = NULL;
    CHECK_STATUS(vpiStreamCreate(backend, &stream) );

    //créer le background substarctor
    VPIPayload payload   = NULL;
    CHECK_STATUS( vpiCreateBackgroundSubtractor(backend,640,480,VPI_IMAGE_FORMAT_BGR8,&payload) );

    //créer les images résutlantes (foreground / background) (ok)
    VPIImage fgmask; 
    VPIImage bgimage;
    CHECK_STATUS(vpiImageCreate(640, 480, VPI_IMAGE_FORMAT_U8, 0, &fgmask));
    CHECK_STATUS(vpiImageCreate(640, 480, VPI_IMAGE_FORMAT_BGR8, 0, &bgimage));

// --------- Function with VPI ----------//

    cv::Mat frame;
    cv::Mat fgFrame = cv::Mat::ones(480,640,CV_8U);
    cv::Mat bgFrame = cv::Mat::ones(480,640,CV_8U);
    VPIImageData fgdata;
    VPIImageData bgdata;
    VPIImage vpiFrame = NULL;

    bool bob= true;
    while (true)
    {
        // Capture une image
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Image vide reçue" << std::endl;
            break;
        }

        // ---------- Suite Backgound remover ---------- //

        // Transforme la frame d'openCV en frame VPI (ok)
        CHECK_STATUS( vpiImageCreateWrapperOpenCVMat(frame, 0, &vpiFrame) ) ;

        //créaction des paramètres du background remover (ok)
        VPIBackgroundSubtractorParams params;
        CHECK_STATUS(vpiInitBackgroundSubtractorParams(&params));
        params.learningRate = 0.01;

        //"submit" bg remover (ok)
        CHECK_STATUS(vpiSubmitBackgroundSubtractor(stream, backend, payload, vpiFrame, fgmask, bgimage, &params));

        //wait for stream to finish (ok)
        CHECK_STATUS(vpiStreamSync(stream));

        //show the bg and the fg img (ok)

        
        CHECK_STATUS(vpiImageLockData(fgmask, VPI_LOCK_READ,VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,&fgdata));
        CHECK_STATUS( vpiImageDataExportOpenCVMat(fgdata, &fgFrame) );
        vpiImageUnlock(fgmask);
        cv::imshow("fg window",fgFrame);

        
  
        vpiImageLockData(bgimage, VPI_LOCK_READ,VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &bgdata);
        vpiImageDataExportOpenCVMat(bgdata, &bgFrame);
        vpiImageUnlock(bgimage);
        cv::imshow("bg window",bgFrame);

        // ---------- Suite Backgound remover ---------- //

        // Affiche l'image
        cv::imshow("Flux camera", frame);

        // Quitte si on appuie sur 'q'
        if (cv::waitKey(1) == 'q')
            break;
        
    }

    // std::cout<<frame.rows<< "\n";
    // std::cout<<frame.cols<< "\n";

    // Libération
    cap.release();
    cv::destroyAllWindows();

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);
  
    vpiImageDestroy(vpiFrame);
    vpiImageDestroy(fgmask);
    vpiImageDestroy(bgimage);

    return 0;
}
