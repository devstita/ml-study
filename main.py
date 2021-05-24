import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import rotate
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
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    fc = nn.Sequential(Flatten(),
                       nn.Dropout(0.5),
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

    epochs = 20
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

    while True:
        image = drawing.draw(28).astype(np.float32)
        prediction = int(torch.argmax(model(torch.from_numpy(image).reshape(1, 1, 28, 28))).item())

        print(f'I Think it is {prediction}')
        drawing.show(image)

        if prediction == 0:
            break

elif command == 'tc':
    main_model = torch.load('model_save_cnn.dat')
    rotation_model = torch.load('model_save_rotation.dat')

    main_model.eval()
    rotation_model.eval()

    angles = list(range(-180, 180, 30))

    while True:
        image = drawing.draw(28).astype(np.float32)

        rotated_angle_idx = rotation_model(torch.from_numpy(image.reshape(1, 1, 28, 28))).argmax(axis=1)
        print(rotated_angle_idx)
        image = rotate(image, -angles[rotated_angle_idx], reshape=False)

        prediction = int(torch.argmax(main_model(torch.from_numpy(image).reshape(1, 1, 28, 28))).item())

        print(f'Rotated: {angles[rotated_angle_idx]}\tI Think it is {prediction}')
        drawing.show(image)

        if prediction == 0:
            break

else:
    print("Exit..")
