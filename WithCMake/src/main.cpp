#include "DataProcesser.h"

int main()
{
    int net_input_width = 1024;
    int net_input_height = 512;
    string model_path = "/home/abaci/chLi/Project/Mine/LibTorch_Detector/model/LapNet/LapNet_Edge_Detect.pt";
    string image_path = "/home/abaci/chLi/Project/Mine/LibTorch_Detector/data/LapNet/984.jpg";
    string video_path = "/home/abaci/chLi/Project/Mine/LibTorch_Detector/data/LapNet/NVR_ch2_main_20201111164000_20201111170000.avi";
    bool show_result = true;
    int screen_width = 1920;
    int screen_height = 1080;
    bool save_result = false;
    string save_path = "/home/abaci/chLi/Project/Mine/LibTorch_Detector/data/LapNet/output";
    MsgLevel msg_level = Info;

    DataProcesser data_processer;

    bool load_model_success = data_processer.init(net_input_width,
                                                  net_input_height,
                                                  model_path,
                                                  msg_level,
                                                  show_result,
                                                  screen_width,
                                                  screen_height,
                                                  save_result,
                                                  save_path);

    if(load_model_success)
    {
        data_processer.detect_video(video_path);
    }

    if(false)
    {
        //输入图片地址进行推理
        data_processer.detect_image(image_path);

        //输入cv::Mat进行推理
        Mat image = imread(image_path);
        data_processer.detect_image(image);

        //输入视频地址进行推理
        data_processer.detect_video(video_path);

        //获取检测到的四边形的四个顶点
        vector<vector<Point>> rect_result = data_processer.get_rect_result();

        //获取网络输出
        Mat output = data_processer.get_output();

        //获取后处理结果渲染图
        Mat result = data_processer.get_result();

        //获取自动合并后的结果图
        Mat merge_image = data_processer.get_merge_image();
    }

    return 1;
}
