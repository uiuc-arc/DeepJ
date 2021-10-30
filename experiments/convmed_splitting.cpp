#include <cassert>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>
#include "../src/DualIntervals.hpp"
#include "../src/MatrixFunctions.hpp"
#include "../src/Perturbations.hpp"
#include "../src/Reader.hpp"

// compile with: g++ -std=c++17 -O2 -fopenmp convmed_splitting.cpp -o convmed_splitting
/* args:
    [DATASET NAME] -- CIFAR, MNIST
    [PERTURBATION TYPE] -- Rotation, ContrastVariation, Haze
    [PERTURBATION PARAM] -- +-x radians for Rotation, [0, x] for ContrastVariation/Haze
    [NUMBER OF IMAGES] -- how many images PER CATEGORY to average results over
    [NUMBER OF SPLITS] -- how many intervals to split into
    [ROTATION PADDING] -- amount of padding to use for rotation
*/

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(int argc, char** argv) {
    std::string dataset = argv[1];
    std::string perturbation = argv[2];
    double perturb_param = std::stod(argv[3]);
    int num_images = std::stoi(argv[4]);
    int num_intervals = std::stoi(argv[5]);
    int rotate_padding = std::stoi(argv[6]);

    std::vector<std::string> categories;
    if (dataset == "CIFAR") {
        categories = {
            "airplane", "automobile", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck"
        };
    } else {
        categories = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        };
    }

    std::vector<std::vector<int>> correct_images(10);
    for (int i = 0; i < 10; ++i) {
        const std::string& img_category = categories[i];
        correct_images[i] = std::vector<int>(num_images);

        int n = 0;

        // get indices of correctly classified images by ALL THREE networks
        std::ifstream in("correctly_classified/Common_" + dataset + "_" + img_category + ".csv");
        std::string line;
        while (std::getline(in, line)) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',') && n < num_images) {
                correct_images[i][n++] = std::stoi(cell);
            }
        }

        assert(num_images == correct_images[i].size());
    }

    int after_conv, num_channels, img_dim;
    if (dataset == "MNIST") {
        after_conv = 1568;
        img_dim = 28;
        num_channels = 1;
    } else if (dataset == "CIFAR") {
        after_conv = 2048;
        img_dim = 32;
        num_channels = 3;
    } else {
        std::cerr << "No such dataset!" << std::endl;
        return 1;
    }

    std::vector<std::pair<double, double>> intervals;
    double interval_size;
    if (perturbation == "Rotation") {
        interval_size = 2 * perturb_param / num_intervals;
    } else {
        interval_size = perturb_param / num_intervals;
    }
    for (int i = 0; i < num_intervals; ++i) {
        if (perturbation == "Rotation")
            intervals.emplace_back(-perturb_param + i*interval_size, -perturb_param + (i+1)*interval_size);
        else
            intervals.emplace_back(i*interval_size, (i+1)*interval_size);
    }

    std::string network_path = "networks/" + dataset + "/ConvMed";
    Mat** conv1w = load_conv_weights<NType>(network_path + "/conv1_weights.csv", num_channels, 16, 4);
    RowVec conv1b = load_conv_biases<NType>(network_path + "/conv1_bias.csv", 16);
    Mat** conv2w = load_conv_weights<NType>(network_path + "/conv2_weights.csv", 16, 32, 4);
    RowVec conv2b = load_conv_biases<NType>(network_path + "/conv2_bias.csv", 32);
    Mat fc1w = load_fc<NType>(network_path + "/fc1_weights.csv", 100, after_conv);
    Mat fc1b = load_fc<NType>(network_path + "/fc1_bias.csv", 100, 1);
    Mat fc2w = load_fc<NType>(network_path + "/fc2_weights.csv", 10, 100);
    Mat fc2b = load_fc<NType>(network_path + "/fc2_bias.csv", 10, 1);

    std::vector<Interval> output_jacobian(num_images * 10 * 10 * num_intervals);
    auto t0 = high_resolution_clock::now();
    
    #pragma omp parallel for collapse(2)
    for (int category_i = 0; category_i < 10; ++category_i) {
        for (int input_num = 0; input_num < num_images; ++input_num) {
            const std::string& img_category = categories[category_i];

            // load image
            int img_index = correct_images[category_i][input_num];  // get actual index of input img file
            Mat* img;
            if (dataset == "MNIST") {
                img = load_img<NType>("datasets/MNIST/digit" + img_category + "/" + std::to_string(img_index) + ".csv", 1, 28);
            } else {
                std::ostringstream ss;
                ss << std::setw(4) << std::setfill('0') << img_index;
                std::string img_num_str = ss.str();
                img = load_img<NType>("datasets/CIFAR/" + img_category + "/" + img_num_str + ".csv", 3, 32);
            }

            for (int interval_i = 0; interval_i < num_intervals; ++interval_i) {
                std::pair<double, double> curr_interval = intervals[interval_i];
                DI epsilon = DI(Interval(curr_interval.first, curr_interval.second), Interval(1, 1));

                Mat* imgcpy = new Mat[num_channels];
                for (int c = 0; c < num_channels; ++c) {
                    imgcpy[c] = img[c];
                }

                // perturb
                if (perturbation == "ContrastVariation") {
                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = contrast_variation(epsilon, imgcpy[c](row, col));
                            }
                        }
                    }
                } else if (perturbation == "Haze") {
                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = haze(epsilon, imgcpy[c](row, col));
                            }
                        }
                    }
                } else {
                    if (dataset == "MNIST") {
                        imgcpy[0] = Rotate1(imgcpy, epsilon, rotate_padding);
                    } else {
                        Mat* rotated_img = Rotate<3>(imgcpy, epsilon, rotate_padding);
                        delete[] imgcpy;
                        imgcpy = rotated_img;
                    }
                }

                // normalize
                if (dataset == "MNIST") {
                    imgcpy[0] = ((imgcpy[0].array() - 0.1307) / 0.3081).matrix();
                } else {
                    double mean[] = {0.4914, 0.4822, 0.4465};
                    double std[] = {0.2023, 0.1994, 0.2010};
                    for (int c = 0; c < 3; ++c) {
                        imgcpy[c] = ((imgcpy[c].array() - mean[c]) / std[c]).matrix();
                    }
                }

                // pass thru network
                Mat* x1;
                if (dataset == "MNIST")
                    x1 = Relu<16>(Conv2d<1, 16>(imgcpy, conv1w, &conv1b, 2, 1));
                else
                    x1 = Relu<16>(Conv2d<3, 16>(imgcpy, conv1w, &conv1b, 2, 1));
                delete[] imgcpy;
                Mat* x2 = Relu<32>(Conv2d<16, 32>(x1, conv2w, &conv2b, 2, 1));
                delete[] x1;
                RowVec x_fc = Flatten<32>(x2);
                delete[] x2;
                Mat out = Relu(fc1w * x_fc.transpose() + fc1b);
                out = fc2w * out + fc2b;
                
                int j_index = category_i * (10 * num_intervals * num_images) + input_num * (10 * num_intervals) + 10 * interval_i;
                for (int i = 0; i < 10; ++i) {
                    output_jacobian[j_index + i] = out(i, 0).getDual();
                }
            }

            delete[] img;
        }
    }

    auto t1 = high_resolution_clock::now();
    double total_time = duration<double, std::milli>(t1 - t0).count();

    std::cout 
        << "splitting,ConvMed,"
        << dataset << ","
        << perturbation << ","
        << perturb_param << ","
        << num_images << ","
        << num_intervals << ","
        << total_time
        << std::endl;
    for (auto it = output_jacobian.begin(); it != output_jacobian.end(); ++it) {
        std::cout << *it << ";";
    }
    std::cout << std::endl;

    for (int i = 0; i < 16; ++i) {
        delete[] conv1w[i];
    }
    delete[] conv1w;
    for (int i = 0; i < 32; ++i) {
        delete[] conv2w[i];
    }
    delete[] conv2w;
}
