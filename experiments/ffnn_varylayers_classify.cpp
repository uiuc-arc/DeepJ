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

// compile with: g++ -std=c++17 -O2 -fopenmp ffnn_varylayers_classify.cpp -o ffnn_varylayers_classify
/* args:
    [INPUT CATEGORY] -- 0-9
    [NUM LAYERS] -- how many layers in the network
*/

int NUM_NEURONS = 30;  // number of neurons per hidden layer

int main(int argc, char** argv) {
    std::string img_category = argv[1];
    int num_layers = std::stoi(argv[2]);

    int img_category_index = std::stoi(img_category);
    int num_channels = 1, img_dim = 28;

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

    int correct_count = 0;
    
    #pragma omp parallel for
    for (int input_num = 0; input_num < 892; ++input_num) {
        // load image
        Mat img = load_fc<NType>("datasets/MNIST/digit" + img_category + "/" + std::to_string(input_num) + ".csv", 784, 1);
        img = ((img.array() - 0.1307) / 0.3081).matrix();
        
        // verify that img is classified correctly
        Mat res = img;
        for (int l = 0; l < num_layers; ++l) {
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
                std::cout << input_num << ",";
            }
        }
    }

    std::cerr << correct_count << std::endl;
}
