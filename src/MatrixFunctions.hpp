#pragma once
#include <Eigen/Core>

#include "DualIntervals.hpp"

using namespace Eigen;

#ifdef USEDOUBLE
    using NType = double;
#else
    using NType = DI;
#endif

typedef Matrix<NType, Dynamic, Dynamic, RowMajor> Mat;
typedef Matrix<NType, 1, Dynamic, RowMajor> RowVec;
typedef Matrix<Interval, Dynamic, Dynamic, RowMajor> IntervalMat;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> DoubleMat;

////////////////////////////////////////////////////////////////////////////////
//
// TENSOR FUNCTIONS
//
////////////////////////////////////////////////////////////////////////////////
// note: tensors are implemented as arrays (3D) or multi-dimensional arrays (4D+) of matrices

/*
 * 2D-Convolution with variable stride, padding, number of channels/kernels, and optional bias
 * Note: Assumes square image and kernel
 * Note: Bias is a tied bias (i.e., each kernel shares the same bias) -- pass in NULL if no bias
 * Note: delete[] must be called on the returned value explicitly
 * 
 * img[n] represents an image with n channels (e.g., 3 for RGB), like a 3D-Tensor
 * kernels[i][j] represents i kernels, each with j channels (note: j must equal n), like a 4D-Tensor
 * stride represents how much the kernel is "moved" when doing the convolution
 * padding represents how much img is padded (with zeros) by on all sides
 */
template <int num_channels, int num_kernels>
Mat* Conv2d(Mat* img, Mat** kernels, RowVec* biases, int stride = 1, int padding = 0) {
    int img_length = img[0].rows();            // side length of the square image
    int kernel_length = kernels[0][0].rows();  // side length of the kernel's square sliding window
    int kernel_size = kernel_length * kernel_length;
    int new_img_length = std::floor(((img_length + 2 * padding - kernel_length) / stride) + 1);

    // pad img
    Mat padded_img[num_channels];
    for (int c = 0; c < num_channels; ++c) {
        padded_img[c] = Mat::Zero(img_length + 2 * padding, img_length + 2 * padding);
        padded_img[c].block(padding, padding, img_length, img_length) = img[c];
    }

    // img im2col, see here: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
    // note: implemention is the transpose of what is described in the link above
    Mat resized_img(new_img_length * new_img_length, num_channels * kernel_size);
    for (int c = 0; c < num_channels; ++c) {
        int curr_resized_row = 0;
        for (int row = 0; row + kernel_length <= img_length + 2 * padding; row += stride) {
            for (int col = 0; col + kernel_length <= img_length + 2 * padding; col += stride) {
                Mat b = padded_img[c].block(row, col, kernel_length, kernel_length);
                Map<RowVec> b_flattened(b.data(), 1, kernel_size);
                resized_img.block(curr_resized_row, c * kernel_size, 1, kernel_size) = b_flattened;
                ++curr_resized_row;
            }
        }
    }
    resized_img.transposeInPlace();

    // kernel im2col
    Mat resized_kernel(num_kernels, num_channels * kernel_size);
    for (int k = 0; k < num_kernels; ++k) {
        for (int c = 0; c < num_channels; ++c) {
            Map<RowVec> k_flattened(kernels[k][c].data(), 1, kernel_size);
            resized_kernel.block(k, c * kernel_size, 1, kernel_size) = k_flattened;
        }
    }

    // convolution as big matrix multiplication now!
    Mat res = resized_kernel * resized_img;

    // col2im, reshape back to correct dimensions, and apply bias (if applicable)
    if (biases) {
        res.colwise() += biases->transpose();  // adds bias value to each row of res (corresponding to each kernel)
    }

    Mat* res_resized = new Mat[num_kernels];
    for (int k = 0; k < num_kernels; ++k) {
        res_resized[k] = Map<Mat>(res.row(k).data(), new_img_length, new_img_length);
    }

    return res_resized;
}

/*
 * Max-Pooling with variable stride, padding, and number of channels/kernel sizes
 * Note: Assumes square image and kernel
 * Note: delete[] must be called on the returned value explicitly
 * 
 * img[n] represents an image with n channels (e.g., 3 for RGB), like a 3D-Tensor
 * kernels_length represents the side length of the kernel (in the first two dimensions)
 * padding represents how much img is padded (with zeros) by on all sides
 * stride represents how much the kernel is "moved" when doing the pooling
 */
template <int num_channels>
Mat* MaxPool2d(Mat img[num_channels], int kernel_length, int stride = 1, int padding = 0) {
    int img_length = img[0].rows();
    int new_img_length = std::floor(((img_length + 2 * padding - kernel_length) / stride) + 1);

    // pad img
    Mat padded_img[num_channels];
    for (int c = 0; c < num_channels; ++c) {
        padded_img[c] = Mat::Zero(img_length + 2 * padding, img_length + 2 * padding);
        padded_img[c].block(padding, padding, img_length, img_length) = img[c];
    }

    Mat* pooled_img = new Mat[num_channels];
    for (int c = 0; c < num_channels; ++c) {
        int curr_row = 0, curr_col = 0;
        pooled_img[c] = Mat(new_img_length, new_img_length);
        for (int row = 0; row + kernel_length <= img_length + 2 * padding; row += stride) {
            for (int col = 0; col + kernel_length <= img_length + 2 * padding; col += stride) {
                pooled_img[c](curr_row, curr_col) = padded_img[c].block(row, col, kernel_length, kernel_length).maxCoeff();
                ++curr_col;
                if (curr_col >= new_img_length) {
                    curr_col = 0;
                    ++curr_row;
                }
            }
        }
    }

    return pooled_img;
}

/* 
 * Average-Pooling with variable stride, padding, and number of channels/kernel sizes
 * Everything the same as max-pooling, except we take the mean of the values in each window
 */
template <int num_channels>
Mat* AvgPool2d(Mat img[num_channels], int kernel_length, int stride = 1, int padding = 0) {
    int img_length = img[0].rows();
    int new_img_length = std::floor(((img_length + 2 * padding - kernel_length) / stride) + 1);

    // pad img
    Mat padded_img[num_channels];
    for (int c = 0; c < num_channels; ++c) {
        padded_img[c] = Mat::Zero(img_length + 2 * padding, img_length + 2 * padding);
        padded_img[c].block(padding, padding, img_length, img_length) = img[c];
    }

    Mat* pooled_img = new Mat[num_channels];
    for (int c = 0; c < num_channels; ++c) {
        int curr_row = 0, curr_col = 0;
        pooled_img[c] = Mat(new_img_length, new_img_length);
        for (int row = 0; row + kernel_length <= img_length + 2 * padding; row += stride) {
            for (int col = 0; col + kernel_length <= img_length + 2 * padding; col += stride) {
                pooled_img[c](curr_row, curr_col) = padded_img[c].block(row, col, kernel_length, kernel_length).mean();
                ++curr_col;
                if (curr_col >= new_img_length) {
                    curr_col = 0;
                    ++curr_row;
                }
            }
        }
    }

    return pooled_img;
}

/*
 * Pad an image by "padding" amount on all sides
 */
template <int num_channels>
Mat* Pad(Mat img[num_channels], int padding) {
    int img_length = img[0].rows();

	Mat* padded_img = new Mat[num_channels];;
    for (int c = 0; c < num_channels; ++c) {
        padded_img[c] = Mat::Zero(img_length + 2 * padding, img_length + 2 * padding); 
        padded_img[c].block(padding, padding, img_length, img_length) = img[c];
    }

	return padded_img;
}

/*
 * Flattens a 3D-Tensor into a single row vector, so that it can be fed into a fully-connected layer
 */
template <int num_channels>
RowVec Flatten(Mat img[num_channels]) {
    int img_size = img[0].rows() * img[0].cols();
    RowVec flattened_img(1, num_channels * img_size);
    for (int c = 0; c < num_channels; ++c) {
        flattened_img.block(0, c * img_size, 1, img_size) = Map<RowVec>(img[c].data(), 1, img_size);
    }
    return flattened_img;
}

// IMPORTANT: use as an in-place operation (output will replace input)
template <int num_channels>
Mat* Relu(Mat* x) {
    for (int c = 0; c < num_channels; ++c) {
        #ifdef USEDOUBLE
            x[c] = x[c].unaryExpr([](double a) { return std::max(0., a); });
        #else
            x[c] = x[c].unaryExpr(std::ref(relu));
        #endif
    }
    return x;
}

////////////////////////////////////////////////////////////////////////////////
//
// MATRIX FUNCTIONS
//
////////////////////////////////////////////////////////////////////////////////

Mat Exp(const Mat& x) {
    return ((x.array()).exp()).matrix();
}

Mat Log(const Mat& x) {
    return ((x.array()).log()).matrix();
}

Mat Sqrt(const Mat& x) {
    return ((x.array()).sqrt()).matrix();
}

Mat Sin(const Mat& x) {
    return ((x.array()).sin()).matrix();
}

Mat Cos(const Mat& x) {
    return ((x.array()).cos()).matrix();
}

Mat Tanh(const Mat& x) {
    return ((x.array()).tanh()).matrix();
}

Mat Atan(const Mat& x) {
    return ((x.array()).atan()).matrix();
}

Mat Logistic(const Mat& x) {
    return (((x.array() * 0.5).tanh()) * 0.5 + 0.5).matrix();
}

Mat Softmax(const Mat& x) {
    Mat temp = Exp(x);
    Mat::Scalar total = temp.sum();
    return temp / total;
}

Mat Relu(const Mat& x) {
    #ifdef USEDOUBLE
        return x.unaryExpr([](double a) { return std::max(0., a); });
    #else
        return x.unaryExpr(std::ref(relu));
    #endif
}

Mat Dropout(const Mat& mask, const Mat& x) {
    return (mask.array() * x.array()).matrix();
}

template <class MatType>
double Norm1(const MatType& array) {
    size_t num_rows = array.rows();
    size_t num_cols = array.cols();

    double norm = 0.;
    for (size_t j = 0; j < num_cols; ++j) {
        double sum = 0.;
        for (size_t i = 0; i < num_rows; ++i) {
            #ifdef USEDOUBLE
                sum += std::abs(array(i, j));
            #else
                Interval ij = array(i, j);
                sum += std::max(std::abs(ij.lower()), std::abs(ij.upper()));
            #endif
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

template <class MatType>
double NormInf(const MatType& array) {
    size_t num_rows = array.rows();
    size_t num_cols = array.cols();

    double norm = 0.;
    for (size_t i = 0; i < num_rows; ++i) {
        double sum = 0.;
        for (size_t j = 0; j < num_cols; ++j) {
            #ifdef USEDOUBLE
                sum += std::abs(array(i, j));
            #else
                Interval ij = array(i, j);
                sum += std::max(std::abs(ij.lower()), std::abs(ij.upper()));
            #endif
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

void Print(const Mat& x) {
    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.cols(); j++) {
            std::cout << x(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}
