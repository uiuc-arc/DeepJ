import os

sizes = [10**(-0.25*k) for k in range(4, 20, 3)]
splits = [2, 3, 5, 9]
datasets = ['CIFAR', 'MNIST']
net_names = ['ConvBig', 'ConvMed', 'FFNN']
perturbations = ['HazeThenRotation', 'ContrastVariationThenRotation', 'ContrastVariationThenHaze']

NUM_IMAGES = 1  # number of images per category
rotate_padding = 4

for net in net_names:
    for dataset in datasets:
        for perturbation in perturbations:
            for s in sizes:
                s1, s2 = s*2, s
                if perturbation == 'ContrastVariationThenHaze':
                    s2 *= 2
                filename = f'{net}_{dataset}_{perturbation}_nosplit.txt'
                os.system(f'./{net.lower()}_compose {dataset} {perturbation} {s1} {s2} {NUM_IMAGES} {rotate_padding} >> results_compose/results_compose_nosplit/{filename}')

    for split in splits:
        for dataset in datasets:
            for perturbation in perturbations:
                for s in sizes:
                    s1, s2 = s*2, s
                    if perturbation == 'ContrastVariationThenHaze':
                        s2 *= 2
                    filename = f'{net}_{dataset}_{perturbation}_split{split}.txt'
                    os.system(f'./{net.lower()}_compose_splitting {dataset} {perturbation} {s1} {s2} {NUM_IMAGES} {split} {rotate_padding} >> results_compose/results_compose_split{split}/{filename}')

for dataset in datasets:
    for perturbation in perturbations:
        for s in sizes:
            s1, s2 = s*2, s
            if perturbation == 'ContrastVariationThenHaze':
                s2 *= 2
            os.system(f'./perturb_compose_baseline {dataset} {perturbation} {s1} {s2} {NUM_IMAGES} {rotate_padding} >> results_compose/results_compose_baseline/perturbation_baseline.csv')
