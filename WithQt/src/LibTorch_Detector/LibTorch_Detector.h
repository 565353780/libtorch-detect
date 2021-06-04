#ifndef LIBTORCH_DETECTOR_H
#define LIBTORCH_DETECTOR_H

#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

enum MsgLevel
{
    NoMsg = 0,
    Info = 1,
    All = 2
};

class LibTorch_Detector
{
public:
    LibTorch_Detector();
    ~LibTorch_Detector();

    bool init(int net_input_width,
              int net_input_height,
              string model_path,
              MsgLevel msg_level=Info,
              bool show_result=false,
              int screen_width=1440,
              int screen_height=900,
              bool save_result=false,
              string save_path="");
    void test();

    vector<string> split(const string& str, const string& delim);

    Mat hstack(vector<Mat> image_vec);
    Mat vstack(vector<Mat> image_vec);

    void merge_result();

    bool detect_image(Mat image);
    bool detect_image(string image_path);
    bool detect_video(string video_path);

    void set_msg_level(MsgLevel msg_level)
    {
        msg_level_ = msg_level;
    }
    void set_net_input_size(int net_input_width, int net_input_height)
    {
        net_input_width_ = net_input_width;
        net_input_height_ = net_input_height;
    }
    void set_net_input_size(Size net_input_size)
    {
        set_net_input_size(net_input_size.width, net_input_size.height);
    }
    void set_result(Mat result)
    {
        result_ = result;
    }

    Mat get_image()
    {
        return image_;
    }
    Mat get_output()
    {
        return output_;
    }
    Mat get_result()
    {
        return result_;
    }
    Mat get_merge_image()
    {
        if(!is_merged_)
        {
            update_stack();
            merge_result();
        }
        return merge_image_;
    }

    bool get_is_v_stack()
    {
        return is_v_stack_;
    }
    int get_screen_width()
    {
        return screen_width_;
    }
    int get_screen_height()
    {
        return screen_height_;
    }
    string get_cv_window_title()
    {
        return cv_window_title_;
    }

protected:
    virtual void pre_progress();

    virtual void post_progress();

    virtual bool after_post_progress();

private:
    void update_stack(int rows, int cols);
    void update_stack();

    bool detect();
    bool detect_image();
    bool detect_video();

public:
    MsgLevel msg_level_;

    bool is_v_stack_;
    bool show_result_;
    bool save_result_;
    bool is_merged_;

    vector<string> net_type_vec_;

    string model_path_;
    string model_name_;
    string image_path_;
    string image_name_;
    string video_path_;
    string video_name_;
    string cv_window_title_;
    string save_path_;

    torch::jit::Module module_;
    torch::Tensor image_tensor_;

    int net_input_width_;
    int net_input_height_;
    int screen_width_;
    int screen_height_;
    int save_image_idx_;

    double show_scale_;

    Mat image_;
    Mat output_;
    Mat result_;
    Mat merge_image_;
};

#endif // LIBTORCH_DETECTOR_H
