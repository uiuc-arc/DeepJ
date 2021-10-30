DeepJ
==========
DeepJ is an interval-domain based analyzer that soundly over-approximates the Clarke Jacobian of a Lipschitz, but not necessarily differentiable function. This repository is the implementation for the paper "A Dual Number Abstraction for Static Analysis of Clarke Jacobians" (POPL 2022) by Jacob Laurel, Rem Yang, Gagandeep Singh, and Sasa Misailovic.

**Contacts**:  
[Jacob Laurel](https://jsl1994.github.io/) (contact for paper) — jlaurel2@illinois.edu  
[Rem Yang](https://remyang55.github.io/) (contact for code) — remyang2@illinois.edu  
[Gagandeep Singh](https://ggndpsngh.github.io/) — ggnds@illinois.edu  
[Sasa Misailovic](http://misailo.cs.illinois.edu/) — misailo@illinois.edu  


Requirements
-------------------------
To run the tool itself, you would need g++ 7 or higher, boost 1.65 or later, Eigen 3.3.9, and OpenMP 4.5 or later.
To automatically compile/run the experiments and analyze the results, you would need python3 and Jupyter Notebook.

We also recommend a machine with at least 12 cores to exploit sufficient parallelism needed to make all the experiments run in a reasonable amount of time.

Installation
-------------------------
Clone the DeepJ repository using git:
```
git clone https://github.com/uiuc-arc/DeepJ.git
cd DeepJ
```
If you do not have boost, you may download it [here](https://www.boost.org/users/download/).
Eigen version 3.3.9 is available [here](https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip).

Files
-------------------------
- **src/**: Contains the core source code. Defines the DualInterval class and its associated mathematical operations, tensor operations (e.g., convolution, pooling), perturbation functions (e.g., rotation, haze), and loaders for image inputs and networks.
- **experiments/**: Contains all the experimental scripts, drivers, datasets, and networks that were used in our paper.
- **eval/**: Contains all the experimental results included in our paper, as well as the notebooks used to analyze them.

Usage
-------------------------
### Reproducing Lipschitz results 
First, compile all the scripts. Note that to run the floating-point sound version of our code, first set the ```is_fpsound``` variable to True in compile_scripts.py. From the main directory:
```
cd experiments
python3 compile_scripts.py
```
Then, to run the experiments for single perturbations:
```
python3 make_folders_single.py
python3 run_experiments_single.py
```
Likewise, to run the experiments for composite perturbations:
```
python3 make_folders_compose.py
python3 run_experiments_compose.py
```
If you would like to only run a certain subset of experiments, you may easily edit the top of the driver scripts. The adjustable parameters are:
- ```sizes```: a list of floats specifying the range of perturbation inputs to use
- ```datasets```: a list of strings specifying the dataset(s) to use (currently, we support MNIST and CIFAR; you can also use other datasets as long as the images are converted to .csv in CHW channel-height-width format)
- ```net_names```: a list of strings specifying the network(s) to use (by default, we supply ConvBig, ConvMed, and FFNN; you can also train your own)
- ```perturbations```: a list of strings specifying the perturbation(s) to use (by default, we implement Haze, ContrastVariation, Rotation, and their compositions, e.g., HazeThenRotation; you may also implement your own as long as they use the primitive functions defined for DualIntervals)
- ```splits```: a list of integers specifying the number of splits to use when running the splitting experiments
- ```NUM_IMAGES```: an integer specifying the number of images _per output category_ to analyze (e.g., specifying 10 for MNIST will analyze 100 images, 10 imges for each digit 0-9)

### Reproducing optimization landscape results
Once inside the experiments/ directory, compile the experimental file:
```
g++ -std=c++17 -O2 -fopenmp ffnn_varylayers.cpp -o ffnn_varylayers
```
Then, run the experiments (which will run on all MNIST networks from 3-9 layers):
```
mkdir ffnn_varylayers_results
python3 run_mnistlayers_experiment.py
```
Again, if you would like to only run a certain subset of experiments, you may edit the top of the driver scripts. The adjustable parameters are:
- ```sizes```: a list of floats specifying the range of perturbation inputs to use
- ```perturbations```: a list of strings specifying the perturbation(s) to use
- ```splits```: an integer specifying the number of splits to use
- ```NUM_IMAGES```: an integer specifying the number of images _per output category_ to analyze (e.g., specifying 10 for MNIST will analyze 100 images, 10 imges for each digit 0-9)

### Individual files
To run just one experiment by itself, open the corresponding .cpp file and check the comment at the top of the file. It will contain the command with which to compile the file, as well as the specification for the command-line arguments to pass in.

Results Analysis
-------------------------
To analyze the results, simply enter the eval/ directory and run the corresponding jupyter notebook:
```
cd eval
jupyter notebook
```
Open notebook _Analyze_LC.ipynb_ for single perturbation Lipschitz experiments,  _Analyze_LC_Compose.ipynb_ for composite perturbation Lipschitz experiments, and _Analyze_Landscape.ipynb_ for the optimization landscape experiments. 

To analyze results that you have run yourself, you may copy the results' folder(s) from the experiments/ directory into the eval/ directory.

License
--------------------
This codebase is licensed under the MIT license.
