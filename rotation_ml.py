import random
from collections import OrderedDict

import numpy as np
import torch.functional
import torch.nn as nn
from torch.nn import functional
from scipy.ndimage import rotate
from torch.utils.data import TensorDataset, DataLoader
import drawing
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

saved_model_file_name = 'model_save_rotation.dat'


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


angles = list(range(-180, 180, 30))

command = input('Input your command: ')
if command == 't':  # Training
    # Todo: Develop auto rotation
    learning_rate = 5e-5
    batch_size = 100
    epochs = 10

    x_train_tmp = torch.FloatTensor(np.array(np.load('dataset/train_data.npy')).reshape((60000, 1, 28, 28)))
    x_train = torch.from_numpy(np.concatenate([np.array(x_train_tmp), np.array(x_train_tmp)]))
    y_train = np.zeros((x_train.shape[0], len(angles)))

    for idx, cur in enumerate(x_train):
        print(f'Processing: {idx}/{len(x_train)}')
        array = cur.numpy()[0]
        angle_idx = random.randrange(0, len(angles))
        angle = angles[angle_idx]

        array = rotate(array, angle, reshape=False)
        x_train[idx][0] = torch.from_numpy(array)
        y_train[idx][angle_idx] = 1

    y_train = torch.from_numpy(y_train)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

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
                       nn.Linear(64 * 7 * 7, len(angles))
                       )

    model = nn.Sequential(OrderedDict({
        'Convolutional Layer1': conv1,
        'Convolutional Layer2': conv2,
        'Fully Connected Layer': fc
    }))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        for idx, (X, Y) in enumerate(train_dataloader):
            prediction = model(X)
            cost = functional.cross_entropy(prediction, Y.argmax(axis=1))

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{epochs} Batch: {idx}/{len(train_dataloader)}\tCost: {cost.item()}')

    torch.save(model, saved_model_file_name)

elif command == 'a':  # Accuracy
    if not os.path.isfile(saved_model_file_name):
        print('Saved model does not exist..')
        exit(3)

    print('Model Exist!!')
    print('... Loading Model ...')

    model = torch.load(saved_model_file_name)
    model.eval()

    with torch.no_grad():
        x_train_tmp = torch.FloatTensor(np.array(np.load('dataset/train_data.npy')).reshape((60000, 1, 28, 28)))
        x_train = torch.from_numpy(np.concatenate([np.array(x_train_tmp), np.array(x_train_tmp)]))
        y_train = np.zeros((x_train.shape[0], len(angles)))

        for idx, cur in enumerate(x_train):
            print(f'Processing: {idx}/{len(x_train)}')
            array = cur.numpy()[0]
            angle_idx = random.randrange(0, len(angles))
            angle = angles[angle_idx]

            array = rotate(array, angle, reshape=False)
            x_train[idx][0] = torch.from_numpy(array)
            y_train[idx][angle_idx] = 1

            # drawing.show(array)

        y_train = torch.from_numpy(y_train).argmax(axis=1)
        prediction = model(x_train).argmax(axis=1)
        accuracy = (prediction == y_train).to(torch.float32).mean()
        print(f'Accuracy: {accuracy:.4f}')

else:
    print('Command error..')
    exit(-1)
