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

// compile with: g++ -std=c++17 -O2 -fopenmp convbig_classify.cpp -o convbig_classify
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
        after_conv = 3136;
        img_dim = 28;
        num_channels = 1;
    } else {
        after_conv = 4096;
        img_dim = 32;
        num_channels = 3;
    }

    std::string network_path = "networks/" + dataset + "/ConvBig";
    Mat** conv1w = load_conv_weights<NType>(network_path + "/conv1_weights.csv", num_channels, 32, 3);
    RowVec conv1b = load_conv_biases<NType>(network_path + "/conv1_bias.csv", 32);
    Mat** conv2w = load_conv_weights<NType>(network_path + "/conv2_weights.csv", 32, 32, 4);
    RowVec conv2b = load_conv_biases<NType>(network_path + "/conv2_bias.csv", 32);
    Mat** conv3w = load_conv_weights<NType>(network_path + "/conv3_weights.csv", 32, 64, 3);
    RowVec conv3b = load_conv_biases<NType>(network_path + "/conv3_bias.csv", 64);
    Mat** conv4w = load_conv_weights<NType>(network_path + "/conv4_weights.csv", 64, 64, 4);
    RowVec conv4b = load_conv_biases<NType>(network_path + "/conv4_bias.csv", 64);
    Mat fc1w = load_fc<NType>(network_path + "/fc1_weights.csv", 512, after_conv);
    Mat fc1b = load_fc<NType>(network_path + "/fc1_bias.csv", 512, 1);
    Mat fc2w = load_fc<NType>(network_path + "/fc2_weights.csv", 512, 512);
    Mat fc2b = load_fc<NType>(network_path + "/fc2_bias.csv", 512, 1);
    Mat fc3w = load_fc<NType>(network_path + "/fc3_weights.csv", 10, 512);
    Mat fc3b = load_fc<NType>(network_path + "/fc3_bias.csv", 10, 1);

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
            x1 = Relu<32>(Conv2d<1, 32>(img, conv1w, &conv1b, 1, 1));
        else
            x1 = Relu<32>(Conv2d<3, 32>(img, conv1w, &conv1b, 1, 1));
        delete[] img;
        Mat* x2 = Relu<32>(Conv2d<32, 32>(x1, conv2w, &conv2b, 2, 1));
        delete[] x1;
        Mat* x3 = Relu<64>(Conv2d<32, 64>(x2, conv3w, &conv3b, 1, 1));
        delete[] x2;
        Mat* x4 = Relu<64>(Conv2d<64, 64>(x3, conv4w, &conv4b, 2, 1));
        delete[] x3;
        RowVec x_fc = Flatten<64>(x4);
        delete[] x4;
        Mat out = Relu(fc1w * x_fc.transpose() + fc1b);
        out = Relu(fc2w * out + fc2b);
        out = fc3w * out + fc3b;

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

    for (int i = 0; i < 32; ++i) {
        delete[] conv1w[i];
        delete[] conv2w[i];
    }
    delete[] conv1w;
    delete[] conv2w;
    for (int i = 0; i < 64; ++i) {
        delete[] conv3w[i];
        delete[] conv4w[i];
    }
    delete[] conv3w;
    delete[] conv4w;
}
