import os

is_fpsound = False

if is_fpsound:
    sound = '-D SOUND'
else:
    sound = ''

print('Compiling ConvBig_Classify')
os.system(f'g++ -std=c++17 -O2 -fopenmp convbig_classify.cpp -o convbig_classify')
print('Compiling ConvBig')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convbig.cpp -o convbig')
print('Compiling ConvBig_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convbig_splitting.cpp -o convbig_splitting')
print('Compiling ConvBig_Compose')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convbig_compose.cpp -o convbig_compose')
print('Compiling ConvBig_Compose_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convbig_compose_splitting.cpp -o convbig_compose_splitting')

print('Compiling ConvMed_Classify')
os.system(f'g++ -std=c++17 -O2 -fopenmp convmed_classify.cpp -o convmed_classify')
print('Compiling ConvMed')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convmed.cpp -o convmed')
print('Compiling ConvMed_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convmed_splitting.cpp -o convmed_splitting')
print('Compiling ConvMed_Compose')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convmed_compose.cpp -o convmed_compose')
print('Compiling ConvMed_Compose_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp convmed_compose_splitting.cpp -o convmed_compose_splitting')

print('Compiling FFNN_Classify')
os.system(f'g++ -std=c++17 -O2 -fopenmp ffnn_classify.cpp -o ffnn_classify')
print('Compiling FFNN')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp ffnn.cpp -o ffnn')
print('Compiling FFNN_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp ffnn_splitting.cpp -o ffnn_splitting')
print('Compiling FFNN_Compose')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp ffnn_compose.cpp -o ffnn_compose')
print('Compiling FFNN_Compose_Splitting')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp ffnn_compose_splitting.cpp -o ffnn_compose_splitting')

print('Compiling Perturb Baseline')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp perturb_baseline.cpp -o perturb_baseline')
print('Compiling Perturb_Compose Baseline')
os.system(f'g++ {sound} -std=c++17 -O2 -fopenmp perturb_compose_baseline.cpp -o perturb_compose_baseline')
