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

saved_model_file_name = 'model_save_cnn.dat'

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
    batch_size = 64
    learning_rate = 1e-4

    # Read Data from CSV File

    # Make Dataset
    x_train = torch.FloatTensor(np.array(np.load('dataset/train_data.npy')).reshape((60000, 1, 28, 28)))
    y_train = torch.FloatTensor(np.array(np.load('dataset/train_label.npy')))

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Training Data Load Done!')

    # Generate Model and Training
    conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
    )
    conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            Flatten()
    )
    fc = nn.Sequential(nn.Linear(7 * 7 * 64, 50, bias=True),
                       nn.ReLU(),
                       nn.Linear(50, 10, bias=True)
    )

    model = nn.Sequential(conv1, conv2, fc)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Model Generation Done!')

    epochs = 2
    for epoch in range(epochs + 1):
        for X, Y in train_dataloader:
            prediction = model(X)
            cost = functional.mse_loss(prediction, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{epochs}\tCost: {cost.item()}')

    torch.save(model, saved_model_file_name)
elif command == 'la':  # Load model and Check Accuracy
    if not os.path.isfile(saved_model_file_name):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load(saved_model_file_name)
    model.eval()

    x_test = torch.FloatTensor(np.array(np.load('dataset/test_data.npy').reshape((10000, 1, 28, 28))))
    y_test = torch.FloatTensor(np.array(np.load('dataset/test_label.npy')))

    data_size = len(y_test)
    count_of_true = 0
    for idx, in_data in enumerate(x_test):
        draw(in_data)
        prediction = torch.argmax(model(in_data.reshape(1, 1, 28, 28)))
        real_data = torch.argmax(y_test[idx])
        print(f'Prediction: {prediction}, y_test[idx]: {real_data}')

    # Todo: Find why the accuracy is too low
    print('Accuracy: ' + str((count_of_true / data_size) * 100))
elif command == 'lq':
    file_path = input('File Path: ')
    if not os.path.isfile(file_path):
        print('File does not exist..')
        exit(2)

    if not os.path.isfile(saved_model_file_name):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load(saved_model_file_name)
    model.eval()

    read_data = np.array(Image.open(file_path).convert('L'), dtype=np.float32).reshape(784)
    img = torch.from_numpy(read_data)
    draw(img)

    print('My Prediction is', torch.argmax(model(img)).item())
else:
    print("Exit..")
