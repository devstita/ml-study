import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

import drawing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

saved_model_file_name = 'model_save_cnn.dat'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


command = input('Input your Command: ').replace(' ', '')

if command == 'n':  # New model
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    batch_size = 100
    learning_rate = 1e-3

    # Read Data from CSV File

    # Make Dataset
    x_train = torch.FloatTensor(np.array(np.load('dataset/train_data.npy')).reshape((60000, 1, 28, 28)))
    y_train = torch.FloatTensor(np.array(np.load('dataset/train_label.npy')))

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Training Data Load Done!')

    # Generate Model and Training
    conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    )
    conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
    )
    fc = nn.Sequential(Flatten(),
                       nn.Linear(64 * 7 * 7, 10, bias=True)
   )

    model = nn.Sequential(OrderedDict({
        'Convolutional Layer1': conv1,
        'Convolutional Layer2': conv2,
        'Fully Connected Layer': fc
    }))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer.add_graph(model, iter(train_dataloader).next()[0])

    print('Model Generation Done!')

    epochs = 15
    iteration = len(train_dataloader)
    for epoch in range(epochs + 1):
        for idx, (X, Y) in enumerate(train_dataloader):
            prediction = model(X)
            cost = functional.cross_entropy(prediction, Y.argmax(axis=1))
            writer.add_scalar('Training Cost', cost, epoch * iteration + idx)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{epochs} Batch: {idx}/{len(train_dataloader)}\tCost: {cost.item()}')

    torch.save(model, saved_model_file_name)
    writer.close()

elif command == 'la':  # Load model and Check Accuracy
    if not os.path.isfile(saved_model_file_name):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load(saved_model_file_name)
    model.eval()

    with torch.no_grad():
        x_test = torch.FloatTensor(np.array(np.load('dataset/test_data.npy').reshape((10000, 1, 28, 28))))
        y_test = torch.FloatTensor(np.array(np.load('dataset/test_label.npy'))).argmax(axis=1)
        prediction = model(x_test).argmax(axis=1)

        accuracy = (prediction == y_test).to(torch.float32).mean()
        print(f'Accuracy: {accuracy:.4f}')

elif command == 'lq':
    if not os.path.isfile(saved_model_file_name):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load(saved_model_file_name)
    model.eval()

    image = drawing.draw(28).astype(np.float32)

    print('My Prediction is', torch.argmax(model(torch.from_numpy(image).reshape(1, 1, 28, 28))).item())
    drawing.show(image)

elif command == 'tc':
    model = torch.load('model_save_cnn.dat')
    model.eval()

    conv1_weights, conv2_weights = None, None
    for name, param in model[0][0].named_parameters():
        if name == 'weight':
            conv1_weights = param.data

    for name, param in model[1][0].named_parameters():
        if name == 'weight':
            conv2_weights = param.data

    plt.figure(figsize=(20, 20))
    for i, kernel in enumerate(conv1_weights):
        plt.subplot(8, 8, i + 1)
        plt.imshow(kernel[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(20, 20))
    for i, kernel in enumerate(conv2_weights):
        plt.subplot(8, 8, i + 1)
        plt.imshow(kernel[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()

else:
    print("Exit..")
