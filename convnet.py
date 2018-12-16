
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
import matplotlib.pyplot as plt
import shutil
import torchvision
import torch.optim as optim


train_dataset = torchvision.datasets.ImageFolder(
    root = 'train',
    transform=torchvision.transforms.ToTensor(),
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 32,
    num_workers = 16,
    shuffle = True
)

test_dataset = torchvision.datasets.ImageFolder(
    root = 'test',
    transform=torchvision.transforms.ToTensor(),
)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = 16,
    num_workers = 8,
    shuffle = True
)

val_dataset = torchvision.datasets.ImageFolder(
    root = 'val',
    transform=torchvision.transforms.ToTensor(),
)

valloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = 16,
    num_workers = 8,
    shuffle = True
)
 

# consulted https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/ for initial simple model
class Net(torch.nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(18 * 200 * 200, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 200 * 200)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return(x)


# Using a modified version of the network from the example docs, we were able to hit ~67% accuracy with a simple setup of
# * 3x3 convolutional layer
# * ReLU
# * max pool
# * Linear
# * ReLU
# * Linear


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Epoch {} test accuracy: {:.2f}%".format(epoch, correct/total * 100))

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



orig_mapping = val_dataset.class_to_idx

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
    plt.savefig('simple_model_output/correct_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Ground Truth:{}'.format(mapping[int(iclabels[i])]))
        plt.imshow(icimages[i])
        plt.suptitle('Incorrect model classification examples')
    plt.savefig('simple_model_output/incorrect_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()



torch.save(net, 'simple_conv_bb_classifier.pth')


# AlexNet architecture:
# * Per specification (section 3.4 of paper), max pools of kernel size 3, stride 2


# AlexNet with some experimentation

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # output channel of size 96
        self.conv1 = nn.Conv2d(3, 192, kernel_size = 11, stride = 4, padding = 1)
        # ReLU activation, does not change size
        self.mp1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(192, 256, kernel_size = 5, padding = 1)
        # another ReLU
        self.mp2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
        # ReLU
        self.mp3 = nn.MaxPool2d(kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
        # ReLU
        self.mp4 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear(384 * 3 * 3, 4096)
        # ReLU
        self.fc2 = nn.Linear(4096, 4096)
        # ReLU
        self.fc3 = nn.Linear(4096, 2)
    
    def forward(self, x):
        x = self.mp1(self.relu(self.conv1(x)))
        x = self.mp2(self.relu(self.conv2(x)))
        x = self.mp3(self.relu(self.conv3(x)))
        x = self.mp4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), 384 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = AlexNet()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(40):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Epoch {} test accuracy: {:.2f}%".format(epoch, correct/total * 100))

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


orig_mapping = val_dataset.class_to_idx

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
    plt.savefig('alexnet_model_output/correct_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(16, 16))
    for i in range(9):
        plt.subplot(330 + i + 1)
        plt.title('Ground Truth: {}'.format(mapping[int(iclabels[i])]))
        plt.imshow(icimages[i])
        plt.suptitle('Incorrect model classification examples')
    plt.savefig('alexnet_model_output/incorrect_{}.jpg'.format(image_index), bbox_inches = 'tight')
    plt.close()



torch.save(net, 'alexnet_bb_classifier.pth')

