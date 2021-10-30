import os

sizes = [10**(-0.25*k) for k in range(2, 19)]
splits = [2, 5, 9, 25]
datasets = ['CIFAR', 'MNIST']
net_names = ['ConvBig', 'ConvMed', 'FFNN']
perturbations = ['ContrastVariation', 'Haze', 'Rotation']

NUM_IMAGES = 10  # number of images per category
rotate_padding = 4

for net in net_names:
    for dataset in datasets:
        for perturbation in perturbations:
            for i, s in enumerate(sizes):
                if perturbation != 'Rotation':
                    s *= 2
                filename = f'{net}_{dataset}_{perturbation}_nosplit.txt'
                os.system(f'./{net.lower()} {dataset} {perturbation} {s} {NUM_IMAGES} {rotate_padding} >> results/results_nosplit/{filename}')

for split in splits:
    for net in net_names:
        for dataset in datasets:               
                for perturbation in perturbations:
                    for i, s in enumerate(sizes):
                        if perturbation != 'Rotation':
                            s *= 2
                        
                        filename = f'{net}_{dataset}_{perturbation}_split{split}.txt'
                        os.system(f'./{net.lower()}_splitting {dataset} {perturbation} {s} {NUM_IMAGES} {split} {rotate_padding} >> results/results_split{split}/{filename}')

for dataset in datasets:
    for perturbation in perturbations:
        for s in sizes:
            if perturbation != 'Rotation':
                s *= 2
            os.system(f'./perturb_baseline {dataset} {perturbation} {s} {NUM_IMAGES} {rotate_padding} >> results/results_baseline/perturbation_baseline.csv')
