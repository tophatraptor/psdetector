
# coding: utf-8


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from pprint import pprint
import scipy.ndimage
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import shutil
import torchvision
import torch.optim as optim

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25])
])

master_dataset = torchvision.datasets.ImageFolder(
    root = '/media/jay/NVME/PSBattles/blur_patches',
    transform=data_transform,
)

# roughly len(master_dataset)/3
train_size = test_size = 994366
val_size = len(master_dataset) - train_size - test_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(master_dataset, [train_size, test_size, val_size])



trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 128,
    num_workers = 16,
    shuffle = True
)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = 32,
    num_workers = 8,
    shuffle = True
)

valloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = 32,
    num_workers = 8,
    shuffle = True
)

class Net(torch.nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(18 * 100 * 100, 64)
        self.fc2 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, 18 * 100 * 100)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return(x)


# Our classes here are: Original, Gaussian, Median, and Noise filters.


net = Net()
net.cuda()

# loss function for classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(3):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i % 1000 == 0) and (i > 0):
            with torch.no_grad():
                total = 0
                correct = 0
                for j, data in enumerate(testloader, 0):
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if j % 300:
                        break
                print("Epoch {} iteration {} test accuracy: {:.2f}%".format(epoch, i, correct/total * 100))

with torch.no_grad():
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Validation accuracy: {:.2f}%".format(correct/total * 100))



torch.save(net, 'filter_detection_basic.pth')



import matplotlib.pyplot as plt
orig_mapping = master_dataset.class_to_idx

mapping = {}
for key in orig_mapping:
    mapping[orig_mapping[key]] = key

for image_index in range(50):
    cimages = []
    clabels = []

    icimages = []
    icpreds = []
    iclabels = []
    
    for i, data in enumerate(valloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_images = images[predicted == labels].cpu()
        correct_labels = labels[predicted == labels].cpu()
        incorrect_images = images[predicted != labels].cpu()
        incorrect_labels = labels[predicted != labels].cpu()
        incorrect_preds = predicted[predicted != labels].cpu()
        for i, x in enumerate(correct_images):
            x = np.swapaxes(x, 0, 2)
            x = np.swapaxes(x, 0, 1)
            cimages.append(x)
            clabels.append(correct_labels[i])
        for i, x in enumerate(incorrect_images):
            x = np.swapaxes(x, 0, 2)
            x = np.swapaxes(x, 0, 1)
            icimages.append(x)
            iclabels.append(incorrect_labels[i])
            icpreds.append(incorrect_preds[i])
        if len(cimages) >= 9 and len(icimages) >= 9:
            break

    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Ground Truth: {}'.format(mapping[int(clabels[i])]))
        plt.imshow(cimages[i])
        plt.suptitle('Correct model classification examples')
    plt.savefig('simple_blur_output/correct_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Pred: {}, Ground Truth:{}'.format(mapping[int(icpreds[i])], mapping[int(iclabels[i])]))
        plt.imshow(icimages[i])
        plt.suptitle('Incorrect model classification examples')
    plt.savefig('simple_blur_output/incorrect_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()

# AlexNet with some experimentation

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 5, stride = 4, padding = 1)
        # ReLU activation, does not change size
        self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(96, 128, kernel_size = 5, padding = 1)
        # another ReLU
        self.mp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size = 3, padding = 1)
        # ReLU
        self.mp3 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
        # ReLU
        self.mp4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(192 * 3 * 3, 2084)
        # ReLU
        self.fc2 = nn.Linear(2084, 2084)
        # ReLU
        self.fc3 = nn.Linear(2084, 4)
    
    def forward(self, x):
        x = self.mp1(self.relu(self.conv1(x)))
        x = self.mp2(self.relu(self.conv2(x)))
        x = self.mp3(self.relu(self.conv3(x)))
        x = self.mp4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), 1728)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = AlexNet()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i % 2000 == 0) and (i > 0):
            with torch.no_grad():
                total = 0
                correct = 0
                for j, data in enumerate(testloader, 0):
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if j % 300:
                        break
                print("Epoch {} Iteration {} test accuracy: {:.2f}%".format(epoch, i, correct/total * 100))

with torch.no_grad():
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Validation accuracy: {:.2f}%".format(correct/total * 100))

torch.save(net, 'filter_detection_alexnet.pth')



orig_mapping = master_dataset.class_to_idx

mapping = {}
for key in orig_mapping:
    mapping[orig_mapping[key]] = key

for image_index in range(50):
    cimages = []
    clabels = []

    icimages = []
    icguesses = []
    iclabels = []
    
    for i, data in enumerate(valloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_images = images[predicted == labels].cpu()
        correct_labels = labels[predicted == labels].cpu()
        incorrect_images = images[predicted != labels].cpu()
        incorrect_labels = labels[predicted != labels].cpu()
        for i, x in enumerate(correct_images):
            x = np.swapaxes(x, 0, 2)
            x = np.swapaxes(x, 0, 1)
            cimages.append(x)
            clabels.append(correct_labels[i])
        for i, x in enumerate(incorrect_images):
            x = np.swapaxes(x, 0, 2)
            x = np.swapaxes(x, 0, 1)
            icimages.append(x)
            iclabels.append(incorrect_labels[i])
        if len(cimages) >= 9 and len(icimages) >= 9:
            break

    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Ground Truth: {}'.format(mapping[int(clabels[i])]))
        plt.imshow(cimages[i])
        plt.suptitle('Correct model classification examples')
    plt.savefig('alexnet_blur_output/correct_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Ground Truth:{}'.format(mapping[int(iclabels[i])]))
        plt.imshow(icimages[i])
        plt.suptitle('Incorrect model classification examples')
    plt.savefig('alexnet_blur_output/incorrect_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()

