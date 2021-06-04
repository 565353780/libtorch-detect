#ifndef DATAPROCESSER_H
#define DATAPROCESSER_H

#include <opencv2/opencv.hpp>
#include <iostream>

#include "LapNet_Detector.h"

#define PI 3.1415926535898

using namespace std;
using namespace cv;

class DataProcesser : public LapNet_Detector
{
public:
    DataProcesser();
    ~DataProcesser();

    bool after_post_progress() override;

    vector<vector<Point>> get_rect_result()
    {
        return merged_hull_vec_;
    }

private:
    int get_dist_of_point(Point point_1, Point point_2);

    int get_dist_of_point_set(vector<Point> point_set_1, vector<Point> point_set_2);

    int get_dist_of_rect(Rect rect_1, Rect rect_2);

    double get_common_area_of_rect(Rect rect_1, Rect rect_2);

    vector<double> get_iou_of_rect(Rect rect_1, Rect rect_2);

    vector<vector<int>> get_intersect_rect_pair(vector<Rect> rect_set);

    Point get_average_point(vector<Point> point_set);

    vector<int> get_nearest_rect(vector<Rect> rect_set);

    int get_max_dist(vector<Point> point_set);

    double get_dist_to_line(Point line_point_1, Point line_point_2, Point point);

    double get_angle(Point point_1, Point point_mid, Point point_2);

    vector<Point> get_approx_rect(vector<Point> point_set);

    float calcuDistance(uchar* ptr, uchar* ptrCen, int cols);

    cv::Mat MaxMinDisFun(cv::Mat data, float Theta, vector<int> centerIndex);

    vector<Vec4i> get_lines(Mat image);

    vector<vector<Point>> get_hull_vec(Mat image);

    vector<vector<Point>> merge_hull_vec_by_intersect(vector<vector<Point>> hull_vec);

    vector<vector<Point>> merge_hull_vec_by_nearest(vector<vector<Point>> hull_vec);

    vector<vector<Point>> merge_hull_vec_by_cluster(vector<vector<Point>> hull_vec);

    vector<vector<Point>> remove_small_hull_vec(vector<vector<Point>> hull_vec);

    vector<vector<Point>> get_approx_poly_vec(vector<vector<Point>> hull_vec);

    vector<vector<Point>> get_rect_vec_from_hull(vector<vector<Point>> hull_vec);

    vector<vector<Point>> get_merged_rect_vec(Mat image);

private:
    bool show_result_;

    double area_bigger_scale_;
    double max_hull_area_;
    double min_area_scale_;

    vector<vector<Point>> merged_hull_vec_;
};

#endif // DATAPROCESSER_H
