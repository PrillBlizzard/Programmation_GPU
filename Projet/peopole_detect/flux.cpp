#include <opencv2/opencv.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Context.h>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/algo/BackgroundSubtractor.h>
#include <iostream>

int main() {
    // Initialisation de la caméra
    cv::VideoCapture cap(0); // 0 = première caméra
    if(!cap.isOpened()){
        std::cerr << "Erreur : impossible d'ouvrir la caméra" << std::endl;
        return -1;
    }

    // Initialisation de VPI
    VPIContext ctx;
    vpiCheck(vpiContextCreate(0, &ctx)); // 0 = CPU, ou VPI_BACKEND_CUDA si GPU NVIDIA

    // Création du modèle de soustraction de fond
    VPIBackground background;
    VPIImageFormat fmt = VPI_IMAGE_FORMAT_U8;
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    vpiCheck(vpiBackgroundCreate(ctx, width, height, fmt, &background));

    cv::Mat frame, grayFrame, fgMask;

    while(true){
        cap >> frame;
        if(frame.empty()) break;

        // Conversion en niveaux de gris
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Conversion OpenCV -> VPI
        VPIImage vpiFrame;
        vpiCheck(vpiImageCreateOpenCVMatWrapper(grayFrame, 0, &vpiFrame));

        VPIImage vpiFGMask;
        vpiCheck(vpiImageCreate(width, height, fmt, 0, &vpiFGMask));

        // Soustraction de fond
        //vpiCheck(vpiBackgroundCompute(ctx, background, vpiFrame, vpiFGMask, nullptr));

        // Conversion VPI -> OpenCV
        cv::Mat fgMaskCV;
        vpiCheck(vpiImageCreateOpenCVMatWrapper(fgMaskCV, vpiFGMask));
        fgMask = fgMaskCV.clone();

        // Affichage
        cv::imshow("Original", frame);
        cv::imshow("Foreground Mask", fgMask);

        // Nettoyage des images VPI temporaires
        vpiImageDestroy(vpiFrame);
        vpiImageDestroy(vpiFGMask);

        if(cv::waitKey(1) == 27) break; // Échapper avec ESC
    }

    // Libération
    vpiBackgroundDestroy(background);
    vpiContextDestroy(ctx);
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
