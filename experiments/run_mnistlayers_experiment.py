import os
import csv
import numpy as np

sizes = [10**(-0.25*k) for k in range(2, 22)]
splits = 25
perturbations = ['Rotation', 'ContrastVariation', 'Haze']

NUM_IMAGES = 10  # number of images per category
rotation_padding = 4

for perturbation in perturbations:
    for n in range(3, 10):
        filename = f"layer{n}_{perturbation}_split{splits}.txt"

        for s in sizes:
            if perturbation != 'Rotation':
                s *= 2
            os.system(f'./ffnn_varylayers {perturbation} {s} {NUM_IMAGES} {n} {splits} {rotation_padding} >> ffnn_varylayers_results/{filename}')
        
        print(f'{n}-layer finished')
