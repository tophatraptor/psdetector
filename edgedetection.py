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
    root = '/media/jay/NVME/PSBattles/edge_patches',
    transform=data_transform,
)

train_size = int(len(master_dataset) * 0.7)
test_size = int(len(master_dataset) * 0.15)
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
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, 18 * 100 * 100)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return(x)


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(10):

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i % 500 == 0) and (i > 0):
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

# AlexNet with some experimentation

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # output channel of size 96
        self.conv1 = nn.Conv2d(3, 192, kernel_size = 11, stride = 4, padding = 1)
        # ReLU activation, does not change size
        self.mp1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(192, 256, kernel_size = 3, padding = 1)
        # another ReLU
        self.mp2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
        # ReLU
        self.mp3 = nn.MaxPool2d(kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
        # ReLU
        self.mp4 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear(384, 4096)
        # ReLU
        self.fc2 = nn.Linear(4096, 4096)
        # ReLU
        self.fc3 = nn.Linear(4096, 4)
    
    def forward(self, x):
        x = self.mp1(self.relu(self.conv1(x)))
        x = self.mp2(self.relu(self.conv2(x)))
        x = self.mp3(self.relu(self.conv3(x)))
        x = self.mp4(self.relu(self.conv4(x)))
        x = x.view(x.size(0), 384)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = AlexNet()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i % 500 == 0) and (i > 0):
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


"""
Below lies what we had so far for the Siamese network
"""
# class SiameseNet(nn.Module):
#     def __init__(self):
#         super(SiameseNet, self).__init__()
#         self.relu = nn.ReLU(inplace = True)
#         self.conv1 = nn.Conv2d(3, 96, kernel_size = 5, stride = 4, padding = 1)
#         # ReLU activation, does not change size
#         self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv2 = nn.Conv2d(96, 128, kernel_size = 5, padding = 1)
#         # another ReLU
#         self.mp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv3 = nn.Conv2d(128, 192, kernel_size = 3, padding = 1)
#         # ReLU
#         self.mp3 = nn.MaxPool2d(kernel_size = 2, padding = 1)
#         self.conv4 = nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
#         # ReLU
#         self.mp4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.fc1 = nn.Linear(192 * 3 * 3, 2084)
#         # ReLU
    
#     def forward(self, x):
#         x = self.mp1(self.relu(self.conv1(x)))
#         x = self.mp2(self.relu(self.conv2(x)))
#         x = self.mp3(self.relu(self.conv3(x)))
#         x = self.mp4(self.relu(self.conv4(x)))
#         x = x.view(x.size(0), 1728)
#         x = self.relu(self.fc1(x))
#         return x

# # contrastive loss definition from https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """

#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

#         return loss_contrastive

# net = SiameseNet()
# criterion = ContrastiveLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# for epoch in range(10):
#     for iteration in range(5000):
#         a = next(iter(trainloader))
#         b = next(iter(testloader))
#         ainputs, alabels = a
#         binputs, blabels = b
#         ainputs = ainputs.cuda()
#         binputs = binputs.cuda()
        
#         labels = alabels != blabels
#         for i, adata in enumerate(ainputs):
#             bdata = binputs
#             optimizer.zero_grad()
#             aout = net.forward(adata)
#             bout = net.forward(bdata)
#             outputs = net.forward(inputs)
#             loss = criterion(aout, bout, labels[i])
#             loss.backward()
#             optimizer.step()
