 #include <opencv2/core/version.hpp>
 #if CV_MAJOR_VERSION >= 3
 #    include <opencv2/imgcodecs.hpp>
 #    include <opencv2/videoio.hpp>
 #    include <opencv2/opencv.hpp>

 #else
 #    include <opencv2/highgui/highgui.hpp>
 #endif
  
 #include <opencv2/imgproc/imgproc.hpp>
 #include <vpi/OpenCVInterop.hpp>
  
 #include <vpi/Array.h>
 #include <vpi/Image.h>
 #include <vpi/ImageFormat.h>
 #include <vpi/Pyramid.h>
 #include <vpi/Status.h>
 #include <vpi/Stream.h>
 #include <vpi/algo/BackgroundSubtractor.h>
 #include <vpi/algo/ConvertImageFormat.h>
  
 #include <iostream>
 #include <sstream>
  
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
  
 int main(int argc, char *argv[])
 {
     // OpenCV image that will be wrapped by a VPIImage.
     // Define it here so that it's destroyed *after* wrapper is destroyed
     cv::Mat cvCurFrame;
  
     // VPI objects that will be used
     VPIStream stream     = NULL;
     VPIImage imgCurFrame = NULL;
     VPIImage bgimage     = NULL;
     VPIImage fgmask      = NULL;
     VPIPayload payload   = NULL;
  
     int retval = 0;
  
     try
     {
         if (argc != 3)
         {
             throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda> <input_video>");
         }
  
         // Parse input parameters
         std::string strBackend    = argv[1];
         std::string strInputVideo = argv[2];
  
         VPIBackend backend;
         if (strBackend == "cpu")
         {
             backend = VPI_BACKEND_CPU;
         }
         else if (strBackend == "cuda")
         {
             backend = VPI_BACKEND_CUDA;
         }
         else
         {
             throw std::runtime_error("Backend '" + strBackend + "' not recognized.");
         }
  
         // Load the input video
         cv::VideoCapture invid("/dev/video0");
         if (!invid.isOpen())
         {
             throw std::runtime_error("Can't open '" + strInputVideo + "'");
         }
  
 #if CV_MAJOR_VERSION >= 3
         int32_t width  = invid.get(cv::CAP_PROP_FRAME_WIDTH);
         int32_t height = invid.get(cv::CAP_PROP_FRAME_HEIGHT);
  
         int fourcc                 = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
         double fps                 = invid.get(cv::CAP_PROP_FPS);
         std::string extOutputVideo = ".mp4";
 #else
         int32_t width  = invid.get(CV_CAP_PROP_FRAME_WIDTH);
         int32_t height = invid.get(CV_CAP_PROP_FRAME_HEIGHT);
  
         // MP4 support with OpenCV-2.4 has issues, we'll use
         // avi/mpeg instead.
         //int fourcc                 = CV_FOURCC('M', 'P', 'E', 'G');
         double fps                 = invid.get(CV_CAP_PROP_FPS);
         std::string extOutputVideo = ".avi";
 #endif
  
         // Create the stream where processing will happen. We'll use user-provided backend.
         CHECK_STATUS(vpiStreamCreate(backend, &stream));
  
         // Create background subtractor payload to be executed on the given backend
         // OpenCV delivers us BGR8 images, so the algorithm is configured to accept that.
         CHECK_STATUS(vpiCreateBackgroundSubtractor(backend, width, height, VPI_IMAGE_FORMAT_BGR8, &payload));
  
         // Create foreground image
         CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &fgmask));
  
         // Create background image
         CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, 0, &bgimage));
  
        
  
         // Fetch a new frame until video ends
         int idxFrame = 1;
  
         while (invid.read(cvCurFrame))
         {
             printf("Processing frame %d\n", idxFrame++);
             // Wrap frame into a VPIImage
             if (imgCurFrame == NULL)
             {
                 CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvCurFrame, 0, &imgCurFrame));
             }
             else
             {
                 CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFrame, cvCurFrame));
             }
  
             VPIBackgroundSubtractorParams params;
             CHECK_STATUS(vpiInitBackgroundSubtractorParams(&params));
             params.learningRate = 0.01;
  
             CHECK_STATUS(
                 vpiSubmitBackgroundSubtractor(stream, backend, payload, imgCurFrame, fgmask, bgimage, &params));
  
             // Wait for processing to finish.
             CHECK_STATUS(vpiStreamSync(stream));
  
             {
                 // Now add it to the output video stream
                 VPIImageData imgdata;
                 CHECK_STATUS(vpiImageLock(fgmask, VPI_LOCK_READ));
  
                 cv::Mat outFrame;
                 CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata, &outFrame));
                
                 

                 CHECK_STATUS(vpiImageUnlock(fgmask));
                 if (cv::waitKey(1) == 'q')
                break;
             }
  
             {
                 VPIImageData bgdata;
                 CHECK_STATUS(vpiImageLock(bgimage, VPI_LOCK_READ));
  
                 cv::Mat outFrame;
                 CHECK_STATUS(vpiImageDataExportOpenCVMat(bgdata, &outFrame));
  
                 
  
                 CHECK_STATUS(vpiImageUnlock(bgimage));
             }
         }
     }
     catch (std::exception &e)
     {
         std::cerr << e.what() << std::endl;
         retval = 1;
     }
  
     // Destroy all resources used
     vpiStreamDestroy(stream);
     vpiPayloadDestroy(payload);
  
     vpiImageDestroy(imgCurFrame);
     vpiImageDestroy(fgmask);
     vpiImageDestroy(bgimage);
  
     return retval;
 }