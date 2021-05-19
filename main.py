# Todo: Change Loading Data Mechanism

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Todo: Develop draw Function with multiple parameters
def draw(*args):
    tensor = args[0]
    if len(args) > 1:
        column = args[1], row = args[2]
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, column * row + 1):
            fig.add_subplot(row, column, i)
            plt.imshow(img)
        plt.show()
    else:
        image = np.reshape(tensor.numpy(), (28, 28))
        plt.imshow(image, cmap='gray')
        plt.show()

command = input('Input your Command: ')

if command == 'n':  # New model
    batch_size = 16
    learning_rate = 1e-5

    # Read Data from CSV File
    train_data = []
    train_label = []

    with open('dataset/mnist_train.csv') as train_csv:
        reader = csv.reader(train_csv)
        next(reader)
        for idx, line in enumerate(reader):
            label = [0 for i in range(10)]
            label[int(line[0])] = 1
            train_label.append(label)
            train_data.append(np.array([list(map(int, line[1:]))]).reshape((28, 28, 1)))
            print(f'Loading {idx + 1}/{60000} : {line}')

    # Make Dataset
    x_train = torch.FloatTensor(train_data)
    y_train = torch.FloatTensor(train_label)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Training Data Load Done !')

    # Generate Model and Training
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ), nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ), nn.Sequential(
            nn.Linear(7 * 7 * 64, 10, bias=True)
        )
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Model Generation Done !')

    epochs = 40
    for epoch in range(epochs + 1):
        for samples in train_dataloader:
            X, Y = samples
            draw(X, 4, 4)

            prediction = model(X)
            cost = functional.mse_loss(prediction, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{epochs}\tCost: {cost.item()}')

    torch.save(model, 'model_save.dat')
elif command == 'la':  # Load model
    if not os.path.isfile('model_saved.dat'):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load('model_save.dat')
    model.eval()

    test_data = []
    test_label = []

    with open('dataset/mnist_test.csv') as test_csv:
        reader = csv.reader(test_csv)
        next(reader)
        for line in reader:
            label = [0 for i in range(10)]
            label[int(line[0])] = 1
            test_label.append(label)
            test_data.append(np.array([list(map(int, line[1:]))]).reshape((28, 28, 1)))

    x_test = torch.FloatTensor(test_data)
    y_test = torch.FloatTensor(test_label)

    data_size = len(test_label)
    count_of_true = 0
    for idx, in_data in enumerate(x_test):
        prediction = torch.argmax(model(in_data))
        real_data = torch.argmax(y_test[idx])
        print(f'Prediction: {prediction}, y_test[idx]: {real_data}')
        if prediction == real_data:
            count_of_true += 1
            print('Correct:', count_of_true)
        else:
            print('Wrong   :', count_of_true)

    print('Accuracy: ' + str((count_of_true / data_size) * 100))
elif command == 'lq':
    file_path = input('File Path: ')
    if not os.path.isfile(file_path):
        print('File does not exist..')
        exit(2)

    if not os.path.isfile('model_save.dat'):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load('model_save.dat')
    model.eval()

    read_data = np.array(Image.open(file_path).convert('L'), dtype=np.float32).reshape(784)
    img = torch.from_numpy(read_data)
    draw(img)

    print('My Prediction is', torch.argmax(model(img)).item())
else:
    print("Exit..")
