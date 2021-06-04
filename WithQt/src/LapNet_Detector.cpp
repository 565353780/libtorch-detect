#include "LapNet_Detector.h"

LapNet_Detector::LapNet_Detector()
{

}

LapNet_Detector::~LapNet_Detector()
{

}

void LapNet_Detector::pre_progress()
{
    Mat input_image;

    if(image_.rows != net_input_height_ || image_.cols != net_input_width_)
    {
        resize(image_, input_image, Size(net_input_width_, net_input_height_));
    }
    else
    {
        input_image = image_;
    }

    image_tensor_ = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols, 3}, torch::kByte);

    image_tensor_ = image_tensor_.permute({0, 3, 1, 2});
    image_tensor_ = image_tensor_.toType(torch::kFloat);
//        image_tensor_ = image_tensor_.div(255);

    image_tensor_ = image_tensor_.to(torch::kCUDA);
}

void LapNet_Detector::post_progress()
{
    image_tensor_ = image_tensor_.squeeze(0);
    image_tensor_ = image_tensor_.detach();
    image_tensor_ = image_tensor_.to(torch::kU8);
    image_tensor_ = image_tensor_.to(torch::kCPU);
    image_tensor_ = image_tensor_[1];

    output_ = Mat(Size(net_input_width_, net_input_height_), CV_8UC1, image_tensor_.data_ptr());

    threshold(output_, output_, 0, 255, THRESH_BINARY_INV);

    if(image_.rows != net_input_height_ || image_.cols != net_input_width_)
    {
        resize(output_, output_, image_.size());
    }
}
