#include <cassert>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
#include "../src/DualIntervals.hpp"
#include "../src/MatrixFunctions.hpp"
#include "../src/Perturbations.hpp"
#include "../src/Reader.hpp"

// compile with: g++ -std=c++17 -O2 -fopenmp ffnn_compose.cpp -o ffnn_compose
/* args:
    [DATASET NAME] -- CIFAR, MNIST
    [PERTURBATION TYPE] -- HazeThenRotation, ContrastVariationThenRotation, ContrastVariationThenHaze
    [PERTURBATION PARAM 1] -- +-x radians for Rotation, [0, x] for ContrastVariation/Haze
    [PERTURBATION PARAM 2] -- +-x radians for Rotation, [0, x] for ContrastVariation/Haze
    [NUMBER OF IMAGES] -- how many images PER CATEGORY to average results over
    [ROTATION PADDING] -- amount of padding to use for rotation
*/

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(int argc, char** argv) {
    std::string dataset = argv[1];
    std::string perturbation = argv[2];
    double perturb_param1 = std::stod(argv[3]);
    double perturb_param2 = std::stod(argv[4]);
    int num_images = std::stoi(argv[5]);
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

    int num_channels, img_dim;
    if (dataset == "MNIST") {
        img_dim = 28;
        num_channels = 1;
    } else if (dataset == "CIFAR") {
        img_dim = 32;
        num_channels = 3;
    } else {
        std::cerr << "No such dataset!" << std::endl;
        return 1;
    }

    std::string network_path = "networks/" + dataset + "/FFNN";

    Mat w[7];
    Mat b[7];
    int rr[] = {1024, 512, 256, 128, 64, 32, 10};
    int cc[] = {784, 1024, 512, 256, 128, 64, 32};
    if (dataset == "CIFAR")
        cc[0] = 32*32*3;
    for (int i = 0; i < 7; ++i) {
        w[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_weights.csv", rr[i], cc[i]);
        b[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_bias.csv", rr[i], 1);
    }

    std::vector<Interval> output_jacobian(num_images * 200);
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

            for (int perturb_i = 0; perturb_i < 2; ++perturb_i) {
                Mat* imgcpy = new Mat[num_channels];
                for (int c = 0; c < num_channels; ++c) {
                    imgcpy[c] = img[c];
                }

                DI epsilon1, epsilon2;

                // perturb
                if (perturbation == "HazeThenRotation") {
                    if (perturb_i == 0) {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(1, 1));
                        epsilon2 = DI(Interval(-perturb_param2, perturb_param2), Interval(0, 0));
                    } else {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(0, 0));
                        epsilon2 = DI(Interval(-perturb_param2, perturb_param2), Interval(1, 1));
                    }

                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = haze(epsilon1, imgcpy[c](row, col));
                            }
                        }
                    }

                    if (dataset == "MNIST") {
                        imgcpy[0] = Rotate1(imgcpy, epsilon2, rotate_padding);
                    } else {
                        Mat* rotated_img = Rotate<3>(imgcpy, epsilon2, rotate_padding);
                        delete[] imgcpy;
                        imgcpy = rotated_img;
                    }
                } else if (perturbation == "ContrastVariationThenRotation") {
                    if (perturb_i == 0) {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(1, 1));
                        epsilon2 = DI(Interval(-perturb_param2, perturb_param2), Interval(0, 0));
                    } else {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(0, 0));
                        epsilon2 = DI(Interval(-perturb_param2, perturb_param2), Interval(1, 1));
                    }

                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = contrast_variation(epsilon1, imgcpy[c](row, col));
                            }
                        }
                    }

                    if (dataset == "MNIST") {
                        imgcpy[0] = Rotate1(imgcpy, epsilon2, rotate_padding);
                    } else {
                        Mat* rotated_img = Rotate<3>(imgcpy, epsilon2, rotate_padding);
                        delete[] imgcpy;
                        imgcpy = rotated_img;
                    }
                } else if (perturbation == "ContrastVariationThenHaze") {
                    if (perturb_i == 0) {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(1, 1));
                        epsilon2 = DI(Interval(0, perturb_param2), Interval(0, 0));
                    } else {
                        epsilon1 = DI(Interval(0, perturb_param1), Interval(0, 0));
                        epsilon2 = DI(Interval(0, perturb_param2), Interval(1, 1));
                    }

                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = contrast_variation(epsilon1, imgcpy[c](row, col));
                            }
                        }
                    }

                    for (int c = 0; c < num_channels; ++c) {
                        for (int row = 0; row < img_dim; ++row) {
                            for (int col = 0; col < img_dim; ++col) {
                                imgcpy[c](row, col) = haze(epsilon2, imgcpy[c](row, col));
                            }
                        }
                    }
                } else {
                    std::cerr << "No such perturbation!" << std::endl;
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
                Mat res;
                if (dataset == "MNIST")
                    res = Flatten<1>(imgcpy).transpose();
                else
                    res = Flatten<3>(imgcpy).transpose();
                delete[] imgcpy;

                for (int l = 0; l < 7; ++l) {
                    res = Relu(w[l] * res + b[l]);
                }

                int j_index = category_i * (20 * num_images) + input_num * 20 + 10 * perturb_i;
                for (int i = 0; i < 10; ++i) {
                    output_jacobian[j_index + i] = res(i, 0).getDual();
                }
            }

            delete[] img;
        }
    }

    auto t1 = high_resolution_clock::now();
    double total_time = duration<double, std::milli>(t1 - t0).count();

    std::cout 
        << "composition,FFNNRelu,"
        << dataset << ","
        << perturbation << ","
        << perturb_param1 << ","
        << perturb_param2 << ","
        << num_images << ","
        << total_time
        << std::endl;
    for (auto it = output_jacobian.begin(); it != output_jacobian.end(); ++it) {
        std::cout << *it << ";";
    }
    std::cout << std::endl;
}
