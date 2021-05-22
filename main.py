import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

saved_model_file_name = 'model_save_cnn.dat'
writer = SummaryWriter()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def draw(image):
    plt.imshow(image, cmap='gray')
    plt.show()


command = input('Input your Command: ')

if command == 'n':  # New model
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

    model = nn.Sequential(conv1, conv2, fc)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer.add_graph(model, iter(train_dataloader).next()[0])

    print('Model Generation Done!')

    epochs = 15
    for epoch in range(epochs + 1):
        idx = 0
        for X, Y in train_dataloader:
            prediction = model(X)
            cost = functional.mse_loss(prediction, Y)
            writer.add_scalar('Training Cost', cost, epoch * 60000 + idx)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{epochs}\tCost: {cost.item()}')
            idx += 1

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
    y_test = torch.FloatTensor(np.array(np.load('dataset/test_label.npy'))).argmax(axis=1)
    prediction = model(x_test).argmax(axis=1)

    correct_count = (prediction == y_test).sum()
    print(f'{(correct_count * 100 / x_test.shape[0]):.2f}%')

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

    read_data = np.array(Image.open(file_path).convert('L'), dtype=np.float32)
    img = torch.from_numpy(read_data)
    draw(img)

    # # Todo: Develop function to see inside of MACHINE
    # img = model[0](img.reshape(1, 1, 28, 28))
    # merged_img = np.zeros((5, 5))
    # for cur_stack in img[0]:
    #     merged_img += cur_stack.detach().numpy()
    # draw(merged_img)
    #
    # img = model[1](img)
    # print(img.shape)
    #
    # img = model[2](img)
    # print(img.shape)
    # print(img)

    # Todo: Know why predictions are WRONG
    print('My Prediction is', torch.argmax(img[0]).item())
else:
    print("Exit..")

writer.close()
