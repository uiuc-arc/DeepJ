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

// compile with: g++ -std=c++17 -O2 -fopenmp ffnn_classify.cpp -o ffnn_classify
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

    int num_channels, img_dim;
    if (dataset == "MNIST") {
        num_channels = 1;
        img_dim = 28;
    } else if (dataset == "CIFAR") {
        num_channels = 3;
        img_dim = 32;
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
        Mat res;
        if (dataset == "MNIST")
            res = Flatten<1>(img).transpose();
        else
            res = Flatten<3>(img).transpose();
        for (int l = 0; l < 7; ++l) {
            res = Relu(w[l] * res + b[l]);
        }

        int maxi = 0;
        double maxval = 0;
        for (int i = 0; i < 10; ++i) {      
            if (res(i, 0).getReal().upper() > maxval) {
                maxval = res(i, 0).getReal().upper();
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
}
