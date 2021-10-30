#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

template <class Type>
Matrix<Type, Dynamic, Dynamic, RowMajor>* load_img(const std::string& path, int num_channels, int img_length) {
    using Mat = Matrix<Type, Dynamic, Dynamic, RowMajor>;

    Mat* img = new Mat[num_channels];

    int cellno = 0, c = 0;
    int img_size = img_length * img_length;

    std::ifstream in(path);
    std::string line;
    std::vector<Type> values(img_size);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            Type val = Type(std::stod(cell));
            values[cellno++] = val;

            if (cellno == img_size) {
                cellno = 0;
                img[c] = Map<Mat>(values.data(), img_length, img_length);
                values.clear();
                ++c;
            }
        }
    }

    return img;
}

template <class Type>
Matrix<Type, Dynamic, Dynamic, RowMajor>** load_conv_weights(const std::string& path, int num_channels, int num_kernels, int kernel_length) {
    using Mat = Matrix<Type, Dynamic, Dynamic, RowMajor>;

    Mat** kernels = new Mat*[num_kernels];
    for (int k = 0; k < num_kernels; ++k) {
        kernels[k] = new Mat[num_channels];
    }

    int cellno = 0, c = 0, k = 0;
    int kernel_size = kernel_length * kernel_length;

    std::ifstream in(path);
    std::string line;
    std::vector<Type> values;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            Type val = Type(std::stod(cell));
            values.push_back(val);
            ++cellno;

            if (cellno == kernel_size) {
                cellno = 0;
                kernels[k][c] = Map<Mat>(values.data(), kernel_length, kernel_length);
                values.clear();
                ++c;
                if (c == num_channels) {
                    c = 0;
                    ++k;
                }
            }
        }
    }
    return kernels;
}

template <class Type>
Matrix<Type, 1, Dynamic, RowMajor> load_conv_biases(const std::string& path, int num_kernels) {
    using RowVec = Matrix<Type, 1, Dynamic, RowMajor>;

    std::ifstream in(path);
    std::string cell;
    std::vector<Type> values;
    while (std::getline(in, cell, ',')) {
        Type val = Type(std::stod(cell));
        values.push_back(val);
    }
    return Map<RowVec>(values.data(), 1, num_kernels);
}

template <class Type>
Matrix<Type, Dynamic, Dynamic, RowMajor> load_fc(const std::string& path, int M, int N) {
    std::ifstream indata(path);
    std::string line;
    std::vector<Type> values;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            Type val = Type(std::stod(cell));
            values.push_back(val);
        }
    }
    return Map<Matrix<Type, Dynamic, Dynamic, RowMajor>>(values.data(), M, N);
}
