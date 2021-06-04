#ifndef LAPNET_DETECTOR_H
#define LAPNET_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <iostream>

#include "LibTorch_Detector.h"

#define PI 3.1415926535898

using namespace std;
using namespace cv;

class LapNet_Detector : public LibTorch_Detector
{
public:
    LapNet_Detector();
    ~LapNet_Detector();

    void pre_progress() override;

    void post_progress() override;
};

#endif // LAPNET_DETECTOR_H
