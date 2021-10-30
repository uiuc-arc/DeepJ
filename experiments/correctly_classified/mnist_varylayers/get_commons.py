import csv
from functools import reduce

mnist_categories = [k for k in range(10)]

for category in mnist_categories:
    sets = []

    for num_layers in range(3, 10):
        with open(f'FFNNRelu{num_layers}_{category}.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        sets.append(set(map(lambda x: int(x), data[0][:-1])))

    with open(f'Common_{category}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(list(reduce(lambda a, b: a & b, sets)))
