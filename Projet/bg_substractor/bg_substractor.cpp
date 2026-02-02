#include <opencv2/opencv.hpp>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/BackgroundSubtractor.h>

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
    } while (0)

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <cpu|cuda>\n";
        return -1;
    }

    /* ---------------- Backend selection ---------------- */
    VPIBackend backend;
    std::string strBackend = argv[1];

    if (strBackend == "cpu")
        backend = VPI_BACKEND_CPU;
    else if (strBackend == "cuda")
        backend = VPI_BACKEND_CUDA;
    else
    {
        std::cerr << "Invalid backend (cpu or cuda)\n";
        return -1;
    }

    /* ---------------- Open camera ---------------- */
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened())
    {
        std::cerr << "Error: cannot open camera\n";
        return -1;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    /* ---------------- VPI objects ---------------- */
    VPIStream stream = nullptr;
    VPIImage imgCurFrame = nullptr;
    VPIImage fgmask = nullptr;
    VPIImage bgimage = nullptr;
    VPIPayload payload = nullptr;

    try
    {
        CHECK_STATUS(vpiStreamCreate(backend, &stream));

        CHECK_STATUS(
            vpiCreateBackgroundSubtractor(
                backend, width, height, VPI_IMAGE_FORMAT_BGR8, &payload));

        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &fgmask));
        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, 0, &bgimage));

        cv::Mat frame;

        /* Grab first frame BEFORE creating payload */
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Failed to grab first frame\n";
            return -1;
        }

        int width  = frame.cols;
        int height = frame.rows;

        /* Now create VPI objects */
        CHECK_STATUS(vpiCreateBackgroundSubtractor(
            backend, width, height, VPI_IMAGE_FORMAT_BGR8, &payload));

        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &fgmask));
        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, 0, &bgimage));


        std::cout << "Press 'q' to quit\n";

        while (true)
        {
            cap >> frame;
            if (frame.empty())
                break;

            if (!frame.isContinuous())
                frame = frame.clone();

            if (frame.type() != CV_8UC3)
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

            if (imgCurFrame)
            {
                vpiImageDestroy(imgCurFrame);
                imgCurFrame = nullptr;
            }

            CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frame, 0, &imgCurFrame));


            /* Wrap OpenCV frame into VPIImage */
            if (imgCurFrame == nullptr)
                CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(frame, 0, &imgCurFrame));
            else
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFrame, frame));

            VPIBackgroundSubtractorParams params;
            CHECK_STATUS(vpiInitBackgroundSubtractorParams(&params));
            params.learningRate = 0.01f;

            CHECK_STATUS(
                vpiSubmitBackgroundSubtractor(
                    stream, backend, payload,
                    imgCurFrame, fgmask, bgimage, &params));

            CHECK_STATUS(vpiStreamSync(stream));

            /* ----------- Display foreground mask ----------- */
            cv::Mat fgMat;
            {
                VPIImageData data;
                CHECK_STATUS(vpiImageLock(fgmask, VPI_LOCK_READ));
                CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &fgMat));
                CHECK_STATUS(vpiImageUnlock(fgmask));
            }

            /* ----------- Display background image ----------- */
            cv::Mat bgMat;
            {
                VPIImageData data;
                CHECK_STATUS(vpiImageLock(bgimage, VPI_LOCK_READ));
                CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &bgMat));
                CHECK_STATUS(vpiImageUnlock(bgimage));
            }

            cv::imshow("Camera", frame);
            cv::imshow("Foreground Mask", fgMat);
            cv::imshow("Background Model", bgMat);

            if (cv::waitKey(1) == 'q')
                break;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    /* ---------------- Cleanup ---------------- */
    vpiImageDestroy(imgCurFrame);
    vpiImageDestroy(fgmask);
    vpiImageDestroy(bgimage);
    vpiPayloadDestroy(payload);
    vpiStreamDestroy(stream);

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
