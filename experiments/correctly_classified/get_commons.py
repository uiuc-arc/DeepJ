import csv
from functools import reduce

mnist_categories = [k for k in range(10)]
cifar_categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
net_names = ['FFNN', 'ConvMed', 'ConvBig']

for dataset in ['CIFAR', 'MNIST']:
    if dataset == 'CIFAR':
        categories = cifar_categories
    else:
        categories = mnist_categories

    for category in categories:
        sets = []

        for net in net_names:
            with open(f'{net}_{dataset}_{category}.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            sets.append(set(map(lambda x: int(x), data[0][:-1])))

        with open(f'Common_{dataset}_{category}.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(list(reduce(lambda a, b: a & b, sets)))
