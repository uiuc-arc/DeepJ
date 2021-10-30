#include <cmath>
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

// compile with: g++ -std=c++17 -O2 -fopenmp convmed_classify.cpp -o convmed_classify
/* args:
    [DATASET NAME] -- CIFAR, MNIST
    [INPUT CATEGORY] -- airplane, automobile, etc. for CIFAR, 0-9 for MNIST
*/

int main(int argc, char** argv) {
    std::string dataset = argv[1];
    std::string img_category = argv[2];

    int img_category_index;
    if (dataset == "MNIST") {
        img_category_index = std::stoi(img_category);
    } else {
        std::vector<std::string> categories = {
            "airplane", "automobile", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck"
        };
        img_category_index = find(categories.begin(), categories.end(), img_category) - categories.begin();
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

    std::string network_path = "networks/" + dataset + "/ConvMed";
    Mat** conv1w = load_conv_weights<NType>(network_path + "/conv1_weights.csv", num_channels, 16, 4);
    RowVec conv1b = load_conv_biases<NType>(network_path + "/conv1_bias.csv", 16);
    Mat** conv2w = load_conv_weights<NType>(network_path + "/conv2_weights.csv", 16, 32, 4);
    RowVec conv2b = load_conv_biases<NType>(network_path + "/conv2_bias.csv", 32);
    Mat fc1w = load_fc<NType>(network_path + "/fc1_weights.csv", 100, after_conv);
    Mat fc1b = load_fc<NType>(network_path + "/fc1_bias.csv", 100, 1);
    Mat fc2w = load_fc<NType>(network_path + "/fc2_weights.csv", 10, 100);
    Mat fc2b = load_fc<NType>(network_path + "/fc2_bias.csv", 10, 1);

    int correct_count = 0;
    
    #pragma omp parallel for
    for (int input_num = 0; input_num < 500; ++input_num) {
        // load image
        Mat* img;
        if (dataset == "MNIST") {
            img = load_img<NType>("datasets/MNIST/digit" + img_category + "/" + std::to_string(input_num) + ".csv", 1, 28);
            img[0] = ((img[0].array() - 0.1307) / 0.3081).matrix();
        } else {
            std::ostringstream ss;
            ss << std::setw(4) << std::setfill('0') << (input_num + 1);
            std::string img_num_str = ss.str();
            img = load_img<NType>("datasets/CIFAR/" + img_category + "/" + img_num_str + ".csv", 3, 32);

            double mean[] = {0.4914, 0.4822, 0.4465};
            double std[] = {0.2023, 0.1994, 0.2010};
            for (int c = 0; c < 3; ++c) {
                img[c] = ((img[c].array() - mean[c]) / std[c]).matrix();
            }
        }
        
        // verify that img is classified correctly
        Mat* x1;
        if (dataset == "MNIST")
            x1 = Relu<16>(Conv2d<1, 16>(img, conv1w, &conv1b, 2, 1));
        else
            x1 = Relu<16>(Conv2d<3, 16>(img, conv1w, &conv1b, 2, 1));
        delete[] img;
        Mat* x2 = Relu<32>(Conv2d<16, 32>(x1, conv2w, &conv2b, 2, 1));
        delete[] x1;
        RowVec x_fc = Flatten<32>(x2);
        delete[] x2;
        Mat out = Relu(fc1w * x_fc.transpose() + fc1b);
        out = fc2w * out + fc2b;

        int maxi = 0;
        double maxval = 0;
        for (int i = 0; i < 10; ++i) {      
            if (out(i, 0).getReal().upper() > maxval) {
                maxval = out(i, 0).getReal().upper();
                maxi = i;
            }
        }
        if (maxi == img_category_index) {
            #pragma omp critical
            {
                ++correct_count;
                if (dataset == "MNIST")
                    std::cout << input_num << ",";
                else
                    std::cout << input_num + 1 << ",";
            }
        }
    }

    std::cerr << correct_count << std::endl;

    for (int i = 0; i < 16; ++i) {
        delete[] conv1w[i];
    }
    delete[] conv1w;
    for (int i = 0; i < 32; ++i) {
        delete[] conv2w[i];
    }
    delete[] conv2w;
}
