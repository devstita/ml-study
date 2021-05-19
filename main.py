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


def draw(*args):
    tensor = args[0]
    if len(args) > 1:
        column = args[1], row = args[2]
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, column * row + 1):
            fig.add_subplot(row, column, i)
            plt.imshow(tensor[i - 1])
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

    # Make Dataset
    x_train = torch.FloatTensor(np.array(np.load('dataset/train_data.npy')))
    y_train = torch.FloatTensor(np.array(np.load('dataset/train_label.npy')))

    print(x_train.shape)
    print(x_train)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Training Data Load Done !')

    # Generate Model and Training
    # Todo: Check model structure
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
        for X, Y in train_dataloader:
            print(X)
            print(X.shape)
            print(Y)
            print(Y.shape)

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

    x_test = torch.FloatTensor(np.array(np.load('dataset/test_data.npy')))
    y_test = torch.FloatTensor(np.array(np.load('dataset/test_label.npy')))

    data_size = len(y_test)
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
