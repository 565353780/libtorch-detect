#include "DataProcesser.h"

DataProcesser::DataProcesser()
{
    show_result_ = false;

    area_bigger_scale_ = 1.5;

    max_hull_area_ = -1;
    min_area_scale_ = 0.1;
}

DataProcesser::~DataProcesser()
{

}

bool DataProcesser::after_post_progress()
{
    Mat output = get_output();

    merged_hull_vec_ = get_merged_rect_vec(output);

    Mat result = Mat::zeros(output.size(), CV_8UC1);

    if(merged_hull_vec_.size() > 0)
    {
        for(int i = 0; i < merged_hull_vec_.size(); ++i)
        {
            polylines(result, merged_hull_vec_[i], true, 255, 3);
        }
    }

    set_result(result);

    return true;
}

int DataProcesser::get_dist_of_point(Point point_1, Point point_2)
{
    return abs(point_1.x - point_2.x) + abs(point_1.y - point_2.y);
}

int DataProcesser::get_dist_of_point_set(vector<Point> point_set_1, vector<Point> point_set_2)
{
    int min_dist = -1;

    for(int i = 0; i < point_set_1.size(); ++i)
    {
        for(int j = 0; j < point_set_2.size(); ++j)
        {
            int current_dist = get_dist_of_point(point_set_1[i], point_set_2[j]);
            if(current_dist < min_dist || min_dist == -1)
            {
                min_dist = current_dist;
            }
        }
    }

    return min_dist;
}

int DataProcesser::get_dist_of_rect(Rect rect_1, Rect rect_2)
{
    return abs(rect_1.x - rect_2.x) + abs(rect_1.x + rect_1.width - rect_2.x - rect_2.width) + abs(rect_1.y - rect_2.y) + abs(rect_1.y + rect_1.height - rect_2.y - rect_2.height);
}

double DataProcesser::get_common_area_of_rect(Rect rect_1, Rect rect_2)
{
    int x_common_min = max(rect_1.x, rect_2.x);
    int x_common_max = min(rect_1.x + rect_1.width, rect_2.x + rect_2.width);
    int y_common_min = max(rect_1.y, rect_2.y);
    int y_common_max = min(rect_1.y + rect_1.height, rect_2.y + rect_2.height);

    if(x_common_max <= x_common_min || y_common_max <= y_common_min)
    {
        return 0;
    }

    return 1.0 * (x_common_max - x_common_min) * (y_common_max - y_common_min);
}

vector<double> DataProcesser::get_iou_of_rect(Rect rect_1, Rect rect_2)
{
    vector<double> iou(3);

    double common_area = get_common_area_of_rect(rect_1, rect_2);

    if(common_area == 0)
    {
        iou[0] = 0;
        iou[1] = 0;
        iou[2] = 0;

        return iou;
    }

    int x_min = min(rect_1.x, rect_2.x);
    int x_max = max(rect_1.x + rect_1.width, rect_2.x + rect_2.width);
    int y_min = min(rect_1.y, rect_2.y);
    int y_max = max(rect_1.y + rect_1.height, rect_2.y + rect_2.height);

    double area = 1.0 * (x_max - x_min) * (y_max - y_min);

    iou[0] = common_area / area;
    iou[1] = common_area / rect_1.area();
    iou[2] = common_area / rect_2.area();

    return iou;
}

vector<vector<int>> DataProcesser::get_intersect_rect_pair(vector<Rect> rect_set)
{
    vector<vector<int>> intersect_rect_pair(rect_set.size());

    if(rect_set.size() < 2)
    {
        return intersect_rect_pair;
    }

    for(int i = 0; i < rect_set.size() - 1; ++i)
    {
        for(int j = i + 1; j < rect_set.size(); ++j)
        {
            if(get_common_area_of_rect(rect_set[i], rect_set[j]) > 0)
            {
                intersect_rect_pair[i].emplace_back(j);
            }
        }
    }

    return intersect_rect_pair;
}

Point DataProcesser::get_average_point(vector<Point> point_set)
{
    if(point_set.size() == 0)
    {
        return Point(-1, -1);
    }

    int x_sum = 0;
    int y_sum = 0;

    for(int i = 0; i < point_set.size(); ++i)
    {
        x_sum += point_set[i].x;
        y_sum += point_set[i].y;
    }

    x_sum = int(1.0 * x_sum / point_set.size());
    y_sum = int(1.0 * y_sum / point_set.size());

    return Point(x_sum, y_sum);
}

vector<int> DataProcesser::get_nearest_rect(vector<Rect> rect_set)
{
    vector<int> nearest_rect(rect_set.size(), -1);

    if(rect_set.size() < 2)
    {
        return nearest_rect;
    }

    for(int i = 0; i < rect_set.size() - 1; ++i)
    {
        int min_dist = -1;
        int min_dist_id = -1;

        for(int j = i + 1; j < rect_set.size(); ++j)
        {
            int current_min_dist = get_dist_of_rect(rect_set[i], rect_set[j]);

            if(current_min_dist < min_dist || min_dist == -1)
            {
                min_dist = current_min_dist;
                min_dist_id = j;
            }
        }

        nearest_rect[i] = min_dist_id;
    }

    return nearest_rect;
}

int DataProcesser::get_max_dist(vector<Point> point_set)
{
    int max_dist = -1;

    if(point_set.size() < 2)
    {
        return 0;
    }

    for(int i = 0; i < point_set.size() - 1; ++i)
    {
        for(int j = i + 1; j < point_set.size(); ++j)
        {
            int current_dist = get_dist_of_point(point_set[i], point_set[j]);

            if(current_dist > max_dist)
            {
                max_dist = current_dist;
            }
        }
    }

    return max_dist;
}

double DataProcesser::get_dist_to_line(Point line_point_1, Point line_point_2, Point point)
{
    double line_len = sqrt((line_point_2.x - line_point_1.x) * (line_point_2.x - line_point_1.x) + (line_point_2.y - line_point_1.y) * (line_point_2.y - line_point_1.y));

    if(line_len == 0)
    {
        return sqrt((line_point_2.x - point.x) * (line_point_2.x - point.x) + (line_point_2.y - point.y) * (line_point_2.y - point.y));
    }

    double project_len = 1.0 * (point.x - line_point_1.x) * (line_point_2.x - line_point_1.x) + (point.y - line_point_1.y) * (line_point_2.y - line_point_1.y);

    project_len /= line_len;

    double len_x = 1.0 * (point.x - line_point_1.x - line_len * line_point_2.x);
    double len_y = 1.0 * (point.y - line_point_1.y - line_len * line_point_2.y);

    return sqrt(len_x * len_x + len_y * len_y);
}

double DataProcesser::get_angle(Point point_1, Point point_mid, Point point_2)
{
    double line_1_len = sqrt((point_1.x - point_mid.x) * (point_1.x - point_mid.x) + (point_1.y - point_mid.y) * (point_1.y - point_mid.y));

    if(line_1_len == 0)
    {
        return 0;
    }

    double line_2_len = sqrt((point_2.x - point_mid.x) * (point_2.x - point_mid.x) + (point_2.y - point_mid.y) * (point_2.y - point_mid.y));

    if(line_2_len == 0)
    {
        return 0;
    }

    return acos(1.0 * ((point_1.x - point_mid.x) * (point_2.x - point_mid.x) + (point_1.y - point_mid.y) * (point_2.y - point_mid.y)) / line_1_len / line_2_len);
}

vector<Point> DataProcesser::get_approx_rect(vector<Point> point_set)
{
    RotatedRect approx_rotated_rect = minAreaRect(point_set);
    Point2f approx_rect[4];
    approx_rotated_rect.points(approx_rect);

    vector<Point> approx_point;

    for(int j = 0; j < 4; ++j)
    {
        approx_point.emplace_back(Point(int(approx_rect[j].x), int(approx_rect[j].y)));
    }

    vector<Point> approx_hull;

    convexHull(Mat(approx_point), approx_hull, false);

    return approx_hull;

    vector<Point> rect;

    if(point_set.size() < 3)
    {
        return rect;
    }

    int max_dist_id_1 = -1;
    int max_dist_id_2 = -1;
    int max_dist = -1;

    for(int i = 0; i < point_set.size() - 1; ++i)
    {
        for(int j = i + 1; j < point_set.size(); ++j)
        {
            int current_dist = get_dist_of_point(point_set[i], point_set[j]);

            if(current_dist > max_dist)
            {
                max_dist = current_dist;
                max_dist_id_1 = i;
                max_dist_id_2 = j;
            }
        }
    }

    int max_dist_id_3 = -1;
    int max_dist_3 = -1;

    for(int i = 0; i < point_set.size(); ++i)
    {
        if(i != max_dist_id_1 && i != max_dist_id_2)
        {
            double current_dist = get_dist_to_line(point_set[max_dist_id_1], point_set[max_dist_id_2], point_set[i]);

            if(current_dist > max_dist_3)
            {
                max_dist_3 = current_dist;
                max_dist_id_3 = i;
            }
        }
    }

    Point average_point(int((point_set[max_dist_id_1].x + point_set[max_dist_id_2].x) / 2.0), int((point_set[max_dist_id_1].y + point_set[max_dist_id_2].y) / 2.0));

    int diff_x = average_point.x - point_set[max_dist_id_3].x;
    int diff_y = average_point.y - point_set[max_dist_id_3].y;

    Point parallel_point = average_point + Point(diff_x, diff_y);

    rect.emplace_back(point_set[max_dist_id_1]);
    rect.emplace_back(point_set[max_dist_id_3]);
    rect.emplace_back(parallel_point);
    rect.emplace_back(point_set[max_dist_id_2]);

    vector<Point> rect_hull;

    convexHull(Mat(rect), rect_hull, false);

    return rect_hull;
}

/*计算欧式距离*/
float DataProcesser::calcuDistance(uchar* ptr, uchar* ptrCen, int cols)
{
    float d = 0.0;
    for (size_t j = 0; j < cols; j++)
    {
        d += (double)(ptr[j] - ptrCen[j])*(ptr[j] - ptrCen[j]);
    }
    d = sqrt(d);
    return d;
}

/** @brief   最大最小距离算法
 @param data  输入样本数据，每一行为一个样本，每个样本可以存在多个特征数据
 @param Theta 阈值，一般设置为0.5，阈值越小聚类中心越多
 @param centerIndex 聚类中心的下标
 @return 返回每个样本的类别，类别从1开始，0表示未分类或者分类失败
*/
cv::Mat  DataProcesser::MaxMinDisFun(cv::Mat data, float Theta, vector<int> centerIndex)
{
    double maxDistance = 0;
    int start = 0;    //初始选一个中心点
    int index = start; //相当于指针指示新中心点的位置
    int k = 0;        //中心点计数，也即是类别
    int dataNum = data.rows; //输入的样本数
                             //vector<int>	centerIndex;//保存中心点
    cv::Mat distance = cv::Mat::zeros(cv::Size(1, dataNum), CV_32FC1); //表示所有样本到当前聚类中心的距离
    cv::Mat minDistance = cv::Mat::zeros(cv::Size(1, dataNum), CV_32FC1); //取较小距离


    cv::Mat classes = cv::Mat::zeros(cv::Size(1, dataNum), CV_32SC1);     //表示类别
    centerIndex.push_back(index); //保存第一个聚类中心

    for (size_t i = 0; i < dataNum; i++)
    {
        uchar* ptr1 = data.ptr<uchar>(i);
        uchar* ptrCen = data.ptr<uchar>(centerIndex.at(0));
        float d= calcuDistance(ptr1, ptrCen, data.cols);
        distance.at<float>(i, 0) = d;
        classes.at<int>(i, 0) = k + 1;
        if (maxDistance < d)
        {
            maxDistance = d;
            index = i; //与第一个聚类中心距离最大的样本
        }
    }

    minDistance = distance.clone();
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    maxVal = maxDistance;
    while (maxVal > (maxDistance*Theta)) {
        k = k + 1;
        centerIndex.push_back(index); //新的聚类中心
        for (size_t i = 0; i < dataNum; i++)
        {
            uchar* ptr1 = data.ptr<uchar>(i);
            uchar* ptrCen = data.ptr<uchar>(centerIndex.at(k));
            float d = calcuDistance(ptr1, ptrCen, data.cols);
            distance.at<float>(i, 0) = d;
            //按照当前最近临方式分类，哪个近就分哪个类别
            if (minDistance.at<float>(i, 0) > distance.at<float>(i, 0))
            {
                minDistance.at<float>(i, 0) = distance.at<float>(i, 0);
                classes.at<int>(i, 0) = k + 1;
            }
        }
        //查找minDistance中最大值
        cv::minMaxLoc(minDistance, &minVal, &maxVal, &minLoc, &maxLoc);
        index = maxLoc.y;
    }
    return classes;
}

vector<Vec4i> DataProcesser::get_lines(Mat image)
{
    vector<Vec4i> lines;

    HoughLinesP(image, lines, 1, PI / 180, 50, 100, 0);

    if(show_result_)
    {
        Mat show_image(image.size(), CV_8UC1, Scalar(0));

        for(auto line : lines)
        {
            cout << line << endl;
            int x0 = line[2];
            int y0 = line[3];
            int x1 = x0 + 500 * line[0];
            int y1 = y0 + 500 * line[1];
            cv::line(show_image, Point(x0, y0), Point(x1, y1), Scalar(255), 3);
        }

        resize(show_image, show_image, Size(0, 0), 0.5, 0.5);

        imshow("lines", show_image);
        waitKey(0);
    }

    return lines;
}

vector<vector<Point>> DataProcesser::get_hull_vec(Mat image)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    vector<vector<Point>> hull_vec(contours.size());

    for(int i = 0; i < contours.size(); ++i)
    {
        convexHull(Mat(contours[i]), hull_vec[i], false);
    }

    if(show_result_)
    {
        Mat drawing = Mat::zeros(image.size(), CV_8UC1);

        for(int i = 0; i < hull_vec.size(); ++i)
        {
            polylines(drawing, hull_vec[i], true, 255, 3);
        }

        resize(drawing, drawing, Size(0, 0), 0.5, 0.5);

        imshow("hull", drawing);
        waitKey(0);
    }

    return hull_vec;
}

vector<vector<Point>> DataProcesser::merge_hull_vec_by_intersect(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() < 2)
    {
        return hull_vec;
    }

    vector<vector<Point>> merged_hull_vec = hull_vec;

    bool have_merged = true;

    while(have_merged)
    {
        have_merged = false;

        vector<vector<Point>> new_merged_hull_vec;

        vector<double> hull_area;

        vector<Rect> hull_rect;

        for(int i = 0; i < merged_hull_vec.size(); ++i)
        {
            hull_area.emplace_back(contourArea(merged_hull_vec[i]));

            hull_rect.emplace_back(boundingRect(merged_hull_vec[i]));
        }

        vector<vector<int>> intersect_rect_pair = get_intersect_rect_pair(hull_rect);

        for(int i = 0; i < intersect_rect_pair.size(); ++i)
        {
            if(intersect_rect_pair[i].size() > 0)
            {
                for(int j = 0; j < intersect_rect_pair[i].size(); ++j)
                {
                    int current_hull_id = intersect_rect_pair[i][j];

                    vector<Point> merged_hull_points;
                    for(int k = 0; k < merged_hull_vec[i].size(); ++k)
                    {
                        merged_hull_points.emplace_back(merged_hull_vec[i][k]);
                    }
                    for(int k = 0; k < merged_hull_vec[current_hull_id].size(); ++k)
                    {
                        merged_hull_points.emplace_back(merged_hull_vec[current_hull_id][k]);
                    }

                    vector<Point> merged_hull;

                    convexHull(Mat(merged_hull_points), merged_hull, false);

                    double merged_hull_area = contourArea(merged_hull);

                    if(hull_area[i] > hull_area[current_hull_id])
                    {
                        if(merged_hull_area / hull_area[i] < area_bigger_scale_)
                        {
                            for(int k = 0; k < merged_hull_vec.size(); ++k)
                            {
                                if(k == i)
                                {
                                    new_merged_hull_vec.emplace_back(merged_hull);
                                }
                                else if(k != current_hull_id)
                                {
                                    new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                                }
                            }

                            have_merged = true;
                            merged_hull_vec = new_merged_hull_vec;

                            break;
                        }
                    }
                    else
                    {
                        if(merged_hull_area / hull_area[current_hull_id] < area_bigger_scale_)
                        {
                            for(int k = 0; k < merged_hull_vec.size(); ++k)
                            {
                                if(k == i)
                                {
                                    new_merged_hull_vec.emplace_back(merged_hull);
                                }
                                else if(k != current_hull_id)
                                {
                                    new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                                }
                            }

                            have_merged = true;
                            merged_hull_vec = new_merged_hull_vec;

                            break;
                        }
                    }
                }
            }

            if(have_merged)
            {
                break;
            }
        }
    }

    return merged_hull_vec;
}

vector<vector<Point>> DataProcesser::merge_hull_vec_by_nearest(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() < 2)
    {
        return hull_vec;
    }

    vector<vector<Point>> merged_hull_vec = hull_vec;

    bool have_merged = true;

    while(have_merged)
    {
        have_merged = false;

        vector<vector<Point>> new_merged_hull_vec;

        vector<double> hull_area;

        vector<Rect> hull_rect;

        for(int i = 0; i < merged_hull_vec.size(); ++i)
        {
            hull_area.emplace_back(contourArea(merged_hull_vec[i]));

            hull_rect.emplace_back(boundingRect(merged_hull_vec[i]));
        }

        vector<int> nearest_rect = get_nearest_rect(hull_rect);

        for(int i = 0; i < nearest_rect.size(); ++i)
        {
            if(nearest_rect[i] > -1)
            {
                int current_hull_id = nearest_rect[i];

                vector<Point> merged_hull_points;
                for(int k = 0; k < merged_hull_vec[i].size(); ++k)
                {
                    merged_hull_points.emplace_back(merged_hull_vec[i][k]);
                }
                for(int k = 0; k < merged_hull_vec[current_hull_id].size(); ++k)
                {
                    merged_hull_points.emplace_back(merged_hull_vec[current_hull_id][k]);
                }

                vector<Point> merged_hull;

                convexHull(Mat(merged_hull_points), merged_hull, false);

                double merged_hull_area = contourArea(merged_hull);

                if(hull_area[i] > hull_area[current_hull_id])
                {
                    if(merged_hull_area / hull_area[i] < area_bigger_scale_)
                    {
                        for(int k = 0; k < merged_hull_vec.size(); ++k)
                        {
                            if(k == i)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull);
                            }
                            else if(k != current_hull_id)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                            }
                        }

                        have_merged = true;
                        merged_hull_vec = new_merged_hull_vec;

                        break;
                    }
                }
                else
                {
                    if(merged_hull_area / hull_area[current_hull_id] < area_bigger_scale_)
                    {
                        for(int k = 0; k < merged_hull_vec.size(); ++k)
                        {
                            if(k == i)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull);
                            }
                            else if(k != current_hull_id)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                            }
                        }

                        have_merged = true;
                        merged_hull_vec = new_merged_hull_vec;

                        break;
                    }
                }
            }

            if(have_merged)
            {
                break;
            }
        }
    }

    return merged_hull_vec;
}

vector<vector<Point>> DataProcesser::merge_hull_vec_by_cluster(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() < 2)
    {
        return hull_vec;
    }

    vector<vector<Point>> merged_hull_vec = hull_vec;

    bool have_merged = true;

    while(have_merged)
    {
        have_merged = false;

        vector<vector<Point>> new_merged_hull_vec;

        vector<double> hull_area;

        vector<Point> hull_average_point;

        vector<double> hull_max_dist;

        for(int i = 0; i < merged_hull_vec.size(); ++i)
        {
            hull_area.emplace_back(contourArea(merged_hull_vec[i]));

            hull_average_point.emplace_back(get_average_point(merged_hull_vec[i]));

            hull_max_dist.emplace_back(0.75 * get_max_dist(merged_hull_vec[i]));
        }

        for(int i = 0; i < merged_hull_vec.size() - 1; ++i)
        {
            for(int j = i + 1; j < merged_hull_vec.size(); ++j)
            {
                if(hull_area[i] > hull_area[j])
                {
                    if(get_dist_of_point(hull_average_point[i], hull_average_point[j]) < hull_max_dist[i])
                    {
                        vector<Point> merged_hull_points;
                        for(int k = 0; k < merged_hull_vec[i].size(); ++k)
                        {
                            merged_hull_points.emplace_back(merged_hull_vec[i][k]);
                        }
                        for(int k = 0; k < merged_hull_vec[j].size(); ++k)
                        {
                            merged_hull_points.emplace_back(merged_hull_vec[j][k]);
                        }

                        vector<Point> merged_hull;

                        convexHull(Mat(merged_hull_points), merged_hull, false);

                        for(int k = 0; k < merged_hull_vec.size(); ++k)
                        {
                            if(k == i)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull);
                            }
                            else if(k != j)
                            {
                                new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                            }
                        }

                        have_merged = true;
                        merged_hull_vec = new_merged_hull_vec;

                        break;
                    }
                }
                else if(get_dist_of_point(hull_average_point[i], hull_average_point[j]) < hull_max_dist[j])
                {
                    vector<Point> merged_hull_points;
                    for(int k = 0; k < merged_hull_vec[i].size(); ++k)
                    {
                        merged_hull_points.emplace_back(merged_hull_vec[i][k]);
                    }
                    for(int k = 0; k < merged_hull_vec[j].size(); ++k)
                    {
                        merged_hull_points.emplace_back(merged_hull_vec[j][k]);
                    }

                    vector<Point> merged_hull;

                    convexHull(Mat(merged_hull_points), merged_hull, false);

                    for(int k = 0; k < merged_hull_vec.size(); ++k)
                    {
                        if(k == i)
                        {
                            new_merged_hull_vec.emplace_back(merged_hull);
                        }
                        else if(k != j)
                        {
                            new_merged_hull_vec.emplace_back(merged_hull_vec[k]);
                        }
                    }

                    have_merged = true;
                    merged_hull_vec = new_merged_hull_vec;

                    break;
                }
            }

            if(have_merged)
            {
                break;
            }
        }
    }

    return merged_hull_vec;
}

vector<vector<Point>> DataProcesser::remove_small_hull_vec(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() == 0)
    {
        return hull_vec;
    }

    vector<vector<Point>> valid_hull_vec;

    vector<double> hull_area;

    for(int i = 0; i < hull_vec.size(); ++i)
    {
        double current_hull_area = contourArea(hull_vec[i]);

        if(current_hull_area > max_hull_area_)
        {
            max_hull_area_ = current_hull_area;
        }

        hull_area.emplace_back(current_hull_area);
    }

    double min_area = min_area_scale_ * max_hull_area_;

    for(int i = 0; i < hull_vec.size(); ++i)
    {
        if(hull_area[i] > min_area)
        {
            valid_hull_vec.emplace_back(hull_vec[i]);
        }
    }

    return valid_hull_vec;
}

vector<vector<Point>> DataProcesser::get_approx_poly_vec(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() == 0)
    {
        return hull_vec;
    }

    vector<vector<Point>> valid_hull_vec;

    for(int i = 0; i < hull_vec.size(); ++i)
    {
        vector<Point> approx_ploy;

        approxPolyDP(hull_vec[i], approx_ploy, 1, true);

        valid_hull_vec.emplace_back(approx_ploy);
    }

    return valid_hull_vec;
}

vector<vector<Point>> DataProcesser::get_rect_vec_from_hull(vector<vector<Point>> hull_vec)
{
    if(hull_vec.size() == 0)
    {
        return hull_vec;
    }

    vector<vector<Point>> valid_hull_vec;

    for(int i = 0; i < hull_vec.size(); ++i)
    {
        vector<Point> approx_rect = get_approx_rect(hull_vec[i]);

        valid_hull_vec.emplace_back(approx_rect);
    }

    return valid_hull_vec;
}

vector<vector<Point>> DataProcesser::get_merged_rect_vec(Mat image)
{
    vector<vector<Point>> merged_hull_vec = get_hull_vec(image);

    merged_hull_vec = merge_hull_vec_by_intersect(merged_hull_vec);

    merged_hull_vec = merge_hull_vec_by_nearest(merged_hull_vec);

    merged_hull_vec = merge_hull_vec_by_cluster(merged_hull_vec);

    merged_hull_vec = remove_small_hull_vec(merged_hull_vec);

    merged_hull_vec = get_approx_poly_vec(merged_hull_vec);

    merged_hull_vec = get_rect_vec_from_hull(merged_hull_vec);

    if(show_result_)
    {
        Mat drawing = Mat::zeros(image.size(), CV_8UC1);

        for(int i = 0; i < merged_hull_vec.size(); ++i)
        {
            polylines(drawing, merged_hull_vec[i], true, 255, 3);
        }

        resize(drawing, drawing, Size(0, 0), 0.5, 0.5);

        imshow("merged hull", drawing);
        waitKey(0);
    }

    return merged_hull_vec;
}
