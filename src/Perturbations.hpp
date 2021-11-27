#pragma once
#include <cassert>
#include <tuple>

#include "DualIntervals.hpp"

// given a coordinate (x, y) in the rotated img, find the (possibly fractional)
// coordinate that mapped to (x, y) in the original img
std::tuple<DI, DI> inv_rotate(const DI& x, const DI& y, const DI& theta) {
    return std::make_tuple(
        x * cos(theta) + y * sin(theta),
        -x * sin(theta) + y * cos(theta));
}

// takes an (r, c) index and returns the (x, y) value, assuming the image center is the origin,
// though one could tweak this function to make a corner of the image the origin
std::tuple<double, double> rc_as_xy(double i, double j, int rows, int cols) {
    return std::make_tuple(
        (j - (double(cols - 1) / 2.)),
        ((double(rows - 1) / 2.) - i));
}

DI interpolate_using_relu(const Mat& Im, const DI& x, const DI& y) {
    int H = Im.rows();
    int W = Im.cols();
    DI sum{};

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            auto [x_, y_] = rc_as_xy(r, c, H, W);
            sum += Im(r, c) * relu(1 - abs(x - x_)) * relu(1 - abs(y - y_));
        }
    }

    return sum;
}

DI interp(const Mat& Im, const DI& x, const DI& y, double i, double j) {
    // convert i, j to x, y so we can compute everything in the xy plane
    auto [x_, y_] = rc_as_xy(i, j, Im.rows(), Im.cols());

    // gets an xy box (assuming the center of the image is the origin) for the pixels [i, i+1] x [j, j+1]
    Interval rows_as_xy = Interval(y_ - 1, y_);
    Interval cols_as_xy = Interval(x_, x_ + 1);

    // sees if the input x, y dual interval overlaps with the rows/cols (as an x,y interval)
    Interval row_intersection = intersect(y.real, rows_as_xy);
    Interval col_intersection = intersect(x.real, cols_as_xy);

    // if x's interval overlaps with i and i+1 AND y's interval overlaps with j and j+1
    if ((!empty(row_intersection)) && (!empty(col_intersection))) {
        DI x_refined = DI(col_intersection.lower(), col_intersection.upper(), x.dual.lower(), x.dual.upper());
        DI y_refined = DI(row_intersection.lower(), row_intersection.upper(), y.dual.lower(), y.dual.upper());
        return interpolate_using_relu(Im, x_refined, y_refined);
    }

    // return an empty interval if there is no overlap
    return DI(1, -1, 1, -1);
}

DI interpolate_using_join(const Mat& Im, const DI& x, const DI& y) {
    int H = Im.rows();
    int W = Im.cols();

    DI val = DI(1, -1, 1, -1);  // initialize an empty interval

    // loop over each 4-cornered interpolation region
    for (int r = 0; r < H - 1; r++) {
        for (int c = 0; c < W - 1; c++) {
            val = interp(Im, x, y, r, c) | val;  // use the join
        }
    }

    assert(!val.isEmpty());
    return val;
}

DI contrast_variation(const DI& epsilon, const DI& pixel) {
    return max(0, (min(1, (pixel - 0.5 * epsilon) / (1 - epsilon))));
}

// haze to white
DI haze(const DI& epsilon, const DI& pixel) {
    return (1 - epsilon) * pixel + epsilon;
}

// rotation for RGB images (i.e., num_channels > 1)
template <int num_channels>
Mat* Rotate(Mat img[num_channels], DI epsilon, int padding = 5) {
    int img_length = img[0].rows();
    int img_length_wpadding = img_length + 2 * padding;

    Mat* rotated_img = new Mat[num_channels];
    for (int i = 0; i < num_channels; ++i) {
        rotated_img[i] = Mat::Zero(img_length, img_length);
    }

    Mat* padded_img = Pad<num_channels>(img, padding);

    for (int row = padding; row < img_length + padding; ++row) {
        for (int col = padding; col < img_length + padding; ++col) {
            auto [x, y] = rc_as_xy(row, col, img_length_wpadding, img_length_wpadding);
            auto [x_orig, y_orig] = inv_rotate(x, y, epsilon);

            for (int c = 0; c < num_channels; ++c) {
                rotated_img[c](row - padding, col - padding) = min(1, max(0, interpolate_using_join(padded_img[c], x_orig, y_orig)));
            }
        }
    }

    delete[] padded_img;
    return rotated_img;
}

// rotation for greyscale images (i.e., num_channels == 1)
Mat Rotate1(Mat* img, DI epsilon, int padding = 5) {
    int img_length = img[0].rows();
    int img_length_wpadding = img_length + 2 * padding;

    Mat rotated_img = Mat::Zero(img_length, img_length);

    Mat padded_img = Mat::Zero(img_length + 2 * padding, img_length + 2 * padding);
    padded_img.block(padding, padding, img_length, img_length) = img[0];

    for (int i = padding; i < img_length + padding; ++i) {
        for (int j = padding; j < img_length + padding; ++j) {
            auto [x, y] = rc_as_xy(i, j, img_length_wpadding, img_length_wpadding);
            auto [x_orig, y_orig] = inv_rotate(x, y, epsilon);
            rotated_img(i - padding, j - padding) = min(1, max(0, interpolate_using_join(padded_img, x_orig, y_orig)));
        }
    }

    return rotated_img;
}
