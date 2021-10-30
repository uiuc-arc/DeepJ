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

// compile with: g++ -std=c++17 -O2 -fopenmp ffnn_varylayers.cpp -o ffnn_varylayers
/* args:
    [PERTURBATION TYPE] -- Rotation, ContrastVariation, Haze
    [PERTURBATION PARAM] -- +-x radians for Rotation, [0, x] for ContrastVariation/Haze
    [NUMBER OF IMAGES] -- how many images PER CATEGORY to look at
    [NUMBER OF LAYERS] -- how many layers the network has
    [NUMBER OF SPLITS] -- how many intervals to split into
    [ROTATION PADDING] -- amount of padding to use for rotation
*/

int NUM_NEURONS = 30;  // number of neurons per hidden layer

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(int argc, char** argv) {
    std::string perturbation = argv[1];
    double perturb_param = std::stod(argv[2]);
    int num_images = std::stoi(argv[3]);
    int num_layers = std::stoi(argv[4]);
    int num_intervals = std::stoi(argv[5]);
    int rotate_padding = std::stoi(argv[6]);

    std::vector<std::string> categories = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };

    std::vector<std::vector<int>> correct_images(10);
    for (int i = 0; i < 10; ++i) {
        const std::string& img_category = categories[i];
        correct_images[i] = std::vector<int>(num_images);

        int n = 0;

        // get indices of correctly classified images by ALL networks
        std::ifstream in("correctly_classified/mnist_varylayers/Common_" + img_category + ".csv");
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

    std::string network_path = "mnist_varylayer_networks/FFNNRelu_" + std::to_string(num_layers);

    Mat w[num_layers];
    Mat b[num_layers];
    for (int i = 0; i < num_layers; ++i) {
        if (i == 0) {
            w[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_weights.csv", NUM_NEURONS, 784);
            b[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_bias.csv", NUM_NEURONS, 1);
        } else if (i == num_layers - 1) {
            w[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_weights.csv", 10, NUM_NEURONS);
            b[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_bias.csv", 10, 1);
        } else {
            w[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_weights.csv", NUM_NEURONS, NUM_NEURONS);
            b[i] = load_fc<NType>(network_path + "/fc" + std::to_string(i+1) + "_bias.csv", NUM_NEURONS, 1);
        }
    }

    std::vector<Interval> output_jacobian(num_images * 10 * 10 * num_intervals);
    auto t0 = high_resolution_clock::now();


    #pragma omp parallel for collapse(2)
    for (int category_i = 0; category_i < 10; ++category_i) {
        for (int input_num = 0; input_num < num_images; ++input_num) {
            const std::string& img_category = categories[category_i];
            
            // load image
            int img_index = correct_images[category_i][input_num];  // get actual index of input img file
            Mat img = load_fc<NType>("datasets/MNIST/digit" + img_category + "/" + std::to_string(img_index) + ".csv", 28, 28);

            for (int interval_i = 0; interval_i < num_intervals; ++interval_i) {
                std::pair<double, double> curr_interval = intervals[interval_i];
                DI epsilon = DI(Interval(curr_interval.first, curr_interval.second), Interval(1, 1));

                Mat imgcpy = img;
                // perturb
                if (perturbation == "ContrastVariation") {
                    for (int row = 0; row < 28; ++row) {
                        for (int col = 0; col < 28; ++col) {
                            imgcpy(row, col) = contrast_variation(epsilon, imgcpy(row, col));
                        }
                    }
                } else if (perturbation == "Haze") {
                    for (int row = 0; row < 28; ++row) {
                        for (int col = 0; col < 28; ++col) {
                            imgcpy(row, col) = haze(epsilon, imgcpy(row, col));
                        }
                    }
                } else {
                    imgcpy = Rotate1(&imgcpy, epsilon, rotate_padding);
                }

                // normalize
                imgcpy = ((imgcpy.array() - 0.1307) / 0.3081).matrix(); 
                
                Mat res = Map<Mat>(imgcpy.data(), 784, 1);
                for (int l = 0; l < num_layers; ++l) {
                    res = Relu(w[l] * res + b[l]);
                }

                int j_index = category_i * (10 * num_intervals * num_images) + input_num * (10 * num_intervals) + 10 * interval_i;
                for (int i = 0; i < 10; ++i) {
                    output_jacobian[j_index + i] = res(i, 0).getDual();
                }
            }
        }
    }

    auto t1 = high_resolution_clock::now();
    double total_time = duration<double, std::milli>(t1 - t0).count();

    std::cout 
        << "splitting,MNISTVaryLayers,"
        << perturbation << ","
        << perturb_param << ","
        << num_images << ","
        << num_layers << ","
        << num_intervals << ","
        << total_time
        << std::endl;
    for (auto it = output_jacobian.begin(); it != output_jacobian.end(); ++it) {
        std::cout << *it << ";";
    }
    std::cout << std::endl;



    // std::cout << no_zero << std::endl;
    // if (!no_zero) {
    //     std::cout << total_time << ",";
    // } else {
    //     std::cout << std::endl;
    // }

    // std::cout 
    //     << "ours" << ","
    //     << "FFNN" << ","
    //     << activation << ","
    //     << net_type << ","
    //     << dataset << ","
    //     << perturbation << ","
    //     << perturb_param << ","
    //     << img_category << ","
    //     << num_images << ","
    //     << no_zero
    //     << std::endl;
}
