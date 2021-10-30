import os

mnist_categories = [k for k in range(10)]
for num_layers in range(3, 10):
    for category in mnist_categories:
        os.system(f'./ffnn_varylayers_classify {category} {num_layers} >> correctly_classified/mnist_varylayers/FFNNRelu{num_layers}_{category}.csv')
