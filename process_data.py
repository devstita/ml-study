import numpy as np
import csv

train_data = []
train_label = []

test_data = []
test_label = []

with open('dataset/mnist_train.csv') as train_csv:
    reader = csv.reader(train_csv)
    next(reader)
    for idx, line in enumerate(reader):
        label = [0 for i in range(10)]
        label[int(line[0])] = 1
        train_label.append(label)
        train_data.append(np.array(list(map(int, line[1:]))).reshape((28, 28)))
        print(f'Training Data Loading: {idx + 1}/{60000}')

with open('dataset/mnist_test.csv') as test_csv:
    reader = csv.reader(test_csv)
    next(reader)
    for line in reader:
        label = [0 for i in range(10)]
        label[int(line[0])] = 1
        test_label.append(label)
        test_data.append(np.array(list(map(int, line[1:]))).reshape((28, 28)))
        print(f'Test Data Loading: {idx + 1}/{10000}')

train_data = np.array(train_data)
train_label = np.array(train_label)

test_data = np.array(test_data)
test_label = np.array(test_label)

np.save('dataset/train_data', train_data)
np.save('dataset/train_label', train_label)

np.save('dataset/test_data', test_data)
np.save('dataset/test_label', test_label)
