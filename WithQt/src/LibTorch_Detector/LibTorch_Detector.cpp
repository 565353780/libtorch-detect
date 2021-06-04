#include "LibTorch_Detector.h"

LibTorch_Detector::LibTorch_Detector()
{
    msg_level_ = Info;

    net_type_vec_.emplace_back("Free");
    net_type_vec_.emplace_back("LapNet");

    is_v_stack_ = true;
    is_merged_ = false;

    save_image_idx_ = 0;

    show_scale_ = 0.95;
}

LibTorch_Detector::~LibTorch_Detector()
{

}

bool LibTorch_Detector::init(int net_input_width,
                             int net_input_height,
                             string model_path,
                             MsgLevel msg_level,
                             bool show_result,
                             int screen_width,
                             int screen_height,
                             bool save_result,
                             string save_path)
{
    std::cout << "CUDA:   " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;

    net_input_width_ = net_input_width;
    net_input_height_ = net_input_height;

    model_path_ = model_path;
    model_name_ = split(model_path_, "/").back();

    msg_level_ = msg_level;

    show_result_ = show_result;
    screen_width_ = screen_width;
    screen_height_ = screen_height;

    save_result_ = save_result;
    save_path_ = save_path;

    if(save_path_[save_path_.length() - 1] != string("/")[0])
    {
        save_path_ += "/";
    }

    cv_window_title_ = "Libtorch Detect Result";

    try
    {
        module_ = torch::jit::load(model_path);
    }
    catch (const c10::Error& e)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::init : Please check model_path." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::init : Please check model_path." << endl;
            cout << "LibTorch_Detector::init : Error msg :" << endl << e.msg() << endl;
            break;
        }

        return false;
    }

    try
    {
        module_.to(torch::kCUDA);
    }
    catch(const c10::Error& e)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::init : Please check your cuda version with \"nvcc -V\"." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::init : Please check your cuda version with \"nvcc -V\"." << endl;
            cout << "LibTorch_Detector::init : Error msg :" << endl << e.msg() << endl;
            break;
        }

        return false;
    }

    switch(msg_level_)
    {
    case NoMsg:
        break;
    case Info:
        cout << "LibTorch_Detector::init : Init model success." << endl;
        break;
    case All:
        cout << "LibTorch_Detector::init : Init model success." << endl;
        cout << "LibTorch_Detector::init : Model Path : " << model_path_ << endl;
        cout << "LibTorch_Detector::init : Model Name : " << model_name_ << endl;
        break;
    }

    return true;
}

void LibTorch_Detector::test()
{
    image_tensor_ = torch::ones({1, 3, 512, 1024});
    image_tensor_ = image_tensor_.to(torch::kCUDA);

    image_tensor_ = module_.forward({image_tensor_}).toTensor();

    switch(msg_level_)
    {
    case NoMsg:
        break;
    case Info:
        cout << "LibTorch_Detector::test : Test passed." << endl;
        break;
    case All:
        cout << "LibTorch_Detector::test : Model Slice :" << endl << image_tensor_.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << endl;
        cout << "LibTorch_Detector::test : passed." << endl;
        break;
    }
}


vector<string> LibTorch_Detector::split(const string& str, const string& delim)
{
    vector<string> res;
    if("" == str)
    {
        return res;
    }

    char *strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p)
    {
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }

    return res;
}

Mat LibTorch_Detector::hstack(vector<Mat> image_vec)
{
    if(image_vec.size() == 0)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::hstack : Please check image_vec." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::hstack : Please check image_vec." << endl;
            cout << "LibTorch_Detector::hstack : No image input." << endl;
            break;
        }
        return Mat();
    }

    int common_type = image_vec[0].type();

    if(image_vec.size() > 1)
    {
        for(int i = 1; i < image_vec.size(); ++i)
        {
            if(image_vec[i].type() != common_type)
            {
                switch(msg_level_)
                {
                case NoMsg:
                    break;
                case Info:
                    cout << "LibTorch_Detector::hstack : Image types mixed." << endl;
                    break;
                case All:
                    cout << "LibTorch_Detector::hstack : Image types mixed." << endl;
                    cout << "LibTorch_Detector::hstack : Image 0 and " << i << " have different types." << endl;
                    cout << "LibTorch_Detector::hstack : Image 0's type : " << image_vec[0].type() << " ; image " << i << "'s type : " << image_vec[i].type() << endl;
                    break;
                }
                return Mat();
            }
        }
    }

    int row_sum = 0;
    int col_sum = 0;

    int max_row = 0;
    int max_col = 0;
    for(int i = 0; i < image_vec.size(); ++i)
    {
        row_sum += image_vec[i].rows;
        col_sum += image_vec[i].cols;

        if(image_vec[i].rows > max_row)
        {
            max_row = image_vec[i].rows;
        }
        if(image_vec[i].cols > max_col)
        {
            max_col = image_vec[i].cols;
        }
    }

    if(row_sum == 0 || col_sum == 0)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::hstack : All images are empty." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::hstack : All images are empty." << endl;
            break;
        }
        return Mat();
    }

    Mat merge_image = Mat::zeros(max_row, col_sum, common_type);

    int merged_col_sum = 0;

    for(int i = 0; i < image_vec.size(); ++i)
    {
        Mat dstMat = merge_image(Rect(merged_col_sum, 0, image_vec[i].cols, image_vec[i].rows));
        image_vec[i].colRange(0, image_vec[i].cols).copyTo(dstMat);
        merged_col_sum += image_vec[i].cols;
    }

    return merge_image;
}

Mat LibTorch_Detector::vstack(vector<Mat> image_vec)
{
    if(image_vec.size() == 0)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::vstack : Please check image_vec." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::vstack : Please check image_vec." << endl;
            cout << "LibTorch_Detector::vstack : No image input." << endl;
            break;
        }
        return Mat();
    }

    int common_type = image_vec[0].type();

    if(image_vec.size() > 1)
    {
        for(int i = 1; i < image_vec.size(); ++i)
        {
            if(image_vec[i].type() != common_type)
            {
                switch(msg_level_)
                {
                case NoMsg:
                    break;
                case Info:
                    cout << "LibTorch_Detector::vstack : Image types mixed." << endl;
                    break;
                case All:
                    cout << "LibTorch_Detector::vstack : Image types mixed." << endl;
                    cout << "LibTorch_Detector::vstack : Image 0 and " << i << " have different types." << endl;
                    cout << "LibTorch_Detector::vstack : Image 0's type : " << image_vec[0].type() << " ; image " << i << "'s type : " << image_vec[i].type() << endl;
                    break;
                }
                return Mat();
            }
        }
    }

    int row_sum = 0;
    int col_sum = 0;

    int max_row = 0;
    int max_col = 0;
    for(int i = 0; i < image_vec.size(); ++i)
    {
        row_sum += image_vec[i].rows;
        col_sum += image_vec[i].cols;

        if(image_vec[i].rows > max_row)
        {
            max_row = image_vec[i].rows;
        }
        if(image_vec[i].cols > max_col)
        {
            max_col = image_vec[i].cols;
        }
    }

    if(row_sum == 0 || col_sum == 0)
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::vstack : All images are empty." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::vstack : All images are empty." << endl;
            break;
        }
        return Mat();
    }

    Mat merge_image = Mat::zeros(row_sum, max_col, common_type);

    int merged_row_sum = 0;

    for(int i = 0; i < image_vec.size(); ++i)
    {
        Mat dstMat = merge_image(Rect(0, merged_row_sum, image_vec[i].cols, image_vec[i].rows));
        image_vec[i].rowRange(0, image_vec[i].rows).copyTo(dstMat);
        merged_row_sum += image_vec[i].rows;
    }

    return merge_image;
}

void LibTorch_Detector::merge_result()
{
    Mat output_bgr;
    if(output_.type() != CV_8UC3)
    {
        cvtColor(output_, output_bgr, CV_GRAY2BGR);
    }
    else
    {
        output_bgr = output_;
    }

    Mat result_bgr;
    if(result_.type() != CV_8UC3)
    {
        cvtColor(result_, result_bgr, CV_GRAY2BGR);
    }
    else
    {
        result_bgr = result_;
    }

    vector<Mat> image_vec;
    image_vec.emplace_back(image_);
    image_vec.emplace_back(output_bgr);
    image_vec.emplace_back(result_bgr);

    if(is_v_stack_)
    {
        merge_image_ = vstack(image_vec);
    }
    else
    {
        merge_image_ = hstack(image_vec);
    }

    is_merged_ = true;
}

bool LibTorch_Detector::detect_image(Mat image)
{
    image_ = image;
    image_path_ = "";
    image_name_ = "";

    return detect_image();
}

bool LibTorch_Detector::detect_image(string image_path)
{
    image_path_ = image_path;
    image_name_ = split(image_path_, "/").back();

    if(image_path_ == "")
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::detect_image : Please check image_path." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::detect_image : Please check image_path." << endl;
            cout << "LibTorch_Detector::detect_image : No image found." << endl;
            break;
        }
        return false;
    }

    image_ = imread(image_path_);

    return detect_image();
}

bool LibTorch_Detector::detect_video(string video_path)
{
    video_path_ = video_path;
    video_name_ = split(video_path_, "/").back();

    return detect_video();
}

void LibTorch_Detector::pre_progress()
{
    return;

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

void LibTorch_Detector::post_progress()
{
    return;

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

bool LibTorch_Detector::after_post_progress()
{
    result_ = output_;

    return true;
}

void LibTorch_Detector::update_stack(int rows, int cols)
{
    if(rows > cols)
    {
        is_v_stack_ = false;
    }
    else
    {
        is_v_stack_ = true;
    }
}

void LibTorch_Detector::update_stack()
{
    update_stack(image_.rows, image_.cols);
}

bool LibTorch_Detector::detect()
{
    pre_progress();

    bool detect_success = false;

    if(!detect_success)
    {
        try
        {
            image_tensor_ = module_.forward({image_tensor_}).toTensor();

            detect_success = true;
        }
        catch(const c10::Error &e)
        {

        }
    }

    if(!detect_success)
    {
        try
        {
            image_tensor_ = module_.forward({image_tensor_}).toTuple()->elements()[0].toTensor();

            detect_success = true;
        }
        catch(const c10::Error &e)
        {

        }
    }

    if(!detect_success)
    {
        return false;
    }

    post_progress();

    return after_post_progress();
}

bool LibTorch_Detector::detect_image()
{
    is_merged_ = false;

    update_stack();

    switch(msg_level_)
    {
    case NoMsg:
        break;
    case Info:
        cout << "LibTorch_Detector::detect_image : Load image succeed." << endl;
        break;
    case All:
        cout << "LibTorch_Detector::detect_image : Load image succeed." << endl;
        cout << "LibTorch_Detector::detect_image : Image path : " << image_path_ << endl;
        cout << "LibTorch_Detector::detect_image : Image name : " << image_name_ << endl;
        cout << "LibTorch_Detector::detect_image : Image width : " << image_.cols << endl;
        cout << "LibTorch_Detector::detect_image : Image height : " << image_.rows << endl;
        break;
    }

    auto t_start = system_clock::now();

    bool detect_success = detect();
    if(!detect_success)
    {
        return false;
    }

    int save_width = image_.cols;
    int save_height = image_.rows;

    if(is_v_stack_)
    {
        save_height *= 3;
    }
    else
    {
        save_width *= 3;
    }

    if(save_result_)
    {
        merge_result();

        string full_save_path;

        if(image_path_ != "")
        {
            vector<string> image_name_split_vec = split(image_name_, ".");
            full_save_path = save_path_ + image_name_split_vec[0] + "_detect." + image_name_split_vec[1];
        }
        else
        {
            full_save_path = save_path_ + to_string(save_image_idx_) + "_detect.jpg";
            ++save_image_idx_;
        }

        imwrite(full_save_path, merge_image_);

        if(show_result_)
        {
            double max_scale = 1.0 * screen_width_ / save_width;
            double height_scale = 1.0 * screen_height_ / save_height;

            if(max_scale > height_scale)
            {
                max_scale = height_scale;
            }

            max_scale *= show_scale_;

            int target_width = int(save_width * max_scale);
            int target_height = int(save_height * max_scale);

            Mat show_image;
            resize(merge_image_, show_image, Size(target_width, target_height));

            imshow(cv_window_title_, show_image);

            waitKey(0);
        }
    }
    else if(show_result_)
    {
        merge_result();

        double max_scale = 1.0 * screen_width_ / save_width;
        double height_scale = 1.0 * screen_height_ / save_height;

        if(max_scale > height_scale)
        {
            max_scale = height_scale;
        }

        max_scale *= show_scale_;

        int target_width = int(save_width * max_scale);
        int target_height = int(save_height * max_scale);

        Mat show_image;
        resize(merge_image_, show_image, Size(target_width, target_height));

        imshow(cv_window_title_, show_image);

        waitKey(0);
    }

    auto t_end = system_clock::now();
    auto t_duration = duration_cast<microseconds>(t_end - t_start);

    double time_spent_ms = int((double(t_duration.count()) * microseconds::period::num / microseconds::period::den) * 1000);

    switch(msg_level_)
    {
    case NoMsg:
        break;
    case Info:
        cout << "LibTorch_Detector::detect_video : Time : " << time_spent_ms << "ms" << endl;
        break;
    case All:
        cout << "LibTorch_Detector::detect_video : show_result :" << show_result_ << endl;
        cout << "LibTorch_Detector::detect_video : save_result : " << save_result_ << endl;
        cout << "LibTorch_Detector::detect_video : save_path : " << save_path_ << endl;
        cout << "LibTorch_Detector::detect_video : Time : " << time_spent_ms << "ms" << endl;
        break;
    }
}

bool LibTorch_Detector::detect_video()
{
    VideoCapture cap(video_path_);
    if(!cap.isOpened())
    {
        switch(msg_level_)
        {
        case NoMsg:
            break;
        case Info:
            cout << "LibTorch_Detector::detect_video : Please check video_path." << endl;
            break;
        case All:
            cout << "LibTorch_Detector::detect_video : Please check video_path." << endl;
            cout << "LibTorch_Detector::detect_video : No video found." << endl;
            break;
        }
        return false;
    }

    double source_fps = cap.get(CAP_PROP_FPS);
    int source_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int source_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    update_stack(source_height, source_width);

    switch(msg_level_)
    {
    case NoMsg:
        break;
    case Info:
        cout << "LibTorch_Detector::detect_video : Load video succeed." << endl;
        break;
    case All:
        cout << "LibTorch_Detector::detect_video : Load video succeed." << endl;
        cout << "LibTorch_Detector::detect_video : Video path : " << video_path_ << endl;
        cout << "LibTorch_Detector::detect_video : Video name : " << video_name_ << endl;
        cout << "LibTorch_Detector::detect_video : Video fps : " << source_fps << endl;
        cout << "LibTorch_Detector::detect_video : Video width : " << source_width << endl;
        cout << "LibTorch_Detector::detect_video : Video height : " << source_height << endl;
        break;
    }

    int save_width = source_width;
    int save_height = source_height;

    if(is_v_stack_)
    {
        save_height *= 3;
    }
    else
    {
        save_width *= 3;
    }

    if(save_result_)
    {
        vector<string> video_name_split_vec = split(video_name_, ".");
        string full_save_path = save_path_ + video_name_split_vec[0] + "_detect." + video_name_split_vec[1];

        VideoWriter writter(full_save_path, CV_FOURCC('M', 'J', 'P', 'G'), source_fps, Size(save_width, save_height), true);

        long nbFrames = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));

        auto t_start = system_clock::now();
        char ch = 13;

        for(long f = 0; f < nbFrames; ++f)
        {
            int current_frame = f + 1;

            //stop
//            if(f > 1000)
//            {
//                break;
//            }

            cap >> image_;

            bool detect_success = detect();
            if(!detect_success)
            {
                return false;
            }

            merge_result();

            writter << merge_image_;

            if(show_result_)
            {
                double max_scale = 1.0 * screen_width_ / save_width;
                double height_scale = 1.0 * screen_height_ / save_height;

                if(max_scale > height_scale)
                {
                    max_scale = height_scale;
                }

                max_scale *= show_scale_;

                int target_width = int(save_width * max_scale);
                int target_height = int(save_height * max_scale);

                Mat show_image;
                resize(merge_image_, show_image, Size(target_width, target_height));

                imshow(cv_window_title_, show_image);

                waitKey(1);
            }

            auto t_end = system_clock::now();
            auto t_duration = duration_cast<microseconds>(t_end - t_start);

            int fps = int(1.0 * current_frame / (double(t_duration.count()) * microseconds::period::num / microseconds::period::den));

            switch(msg_level_)
            {
            case NoMsg:
                break;
            case Info:
                cout << ch << "LibTorch_Detector::detect_video : Frame : " << current_frame << " / " << nbFrames << " ; FPS : " << fps << "    ";
                break;
            case All:
                cout << "LibTorch_Detector::detect_video : show_result :" << show_result_ << endl;
                cout << "LibTorch_Detector::detect_video : save_result : " << save_result_ << endl;
                cout << "LibTorch_Detector::detect_video : save_path : " << save_path_ << endl;
                cout << "LibTorch_Detector::detect_video : Frame : " << current_frame << " / " << nbFrames << " ; FPS : " << fps<< endl;
                break;
            }
        }

        cap.release();
        writter.release();
    }
    else
    {
        long nbFrames = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));

        auto t_start = system_clock::now();
        char ch = 13;

        for(long f = 0; f < nbFrames; ++f)
        {
            int current_frame = f + 1;

            cap >> image_;

            bool detect_success = detect();
            if(!detect_success)
            {
                return false;
            }

            if(show_result_)
            {
                merge_result();

                double max_scale = 1.0 * screen_width_ / save_width;
                double height_scale = 1.0 * screen_height_ / save_height;

                if(max_scale > height_scale)
                {
                    max_scale = height_scale;
                }

                max_scale *= show_scale_;

                int target_width = int(save_width * max_scale);
                int target_height = int(save_height * max_scale);

                Mat show_image;
                resize(merge_image_, show_image, Size(target_width, target_height));

                imshow(cv_window_title_, show_image);

                waitKey(1);
            }

            auto t_end = system_clock::now();
            auto t_duration = duration_cast<microseconds>(t_end - t_start);

            int fps = int(1.0 * current_frame / (double(t_duration.count()) * microseconds::period::num / microseconds::period::den));

            switch(msg_level_)
            {
            case NoMsg:
                break;
            case Info:
                cout << ch << "LibTorch_Detector::detect_video : Frame : " << current_frame << " / " << nbFrames << " ; FPS : " << fps << "    ";
                break;
            case All:
                cout << "LibTorch_Detector::detect_video : show_result :" << show_result_ << endl;
                cout << "LibTorch_Detector::detect_video : save_result : " << save_result_ << endl;
                cout << "LibTorch_Detector::detect_video : save_path : " << save_path_ << endl;
                cout << "LibTorch_Detector::detect_video : Frame : " << current_frame << " / " << nbFrames << " ; FPS : " << fps << endl;
                break;
            }
        }

        cap.release();
    }

    cout << endl;

    return true;
}
