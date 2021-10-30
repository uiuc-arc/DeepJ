import os

mnist_categories = [k for k in range(10)]
cifar_categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
net_names = ['FFNN', 'ConvMed', 'ConvBig']

for dataset in ["CIFAR", "MNIST"]:
    if dataset == 'CIFAR':
        categories = cifar_categories
    else:
        categories = mnist_categories

    for net in net_names:
        for category in categories:
            os.system(f'./{net.lower()}_classify {dataset} {category} >> correctly_classified/{net}_{dataset}_{category}.csv')
