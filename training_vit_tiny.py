# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch_directml
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import timm

model: nn.Module = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
imsize = 224

device = torch_directml.device()
model = model.to(device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)


def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    inputs: torch.Tensor
    targets: torch.Tensor
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs: torch.Tensor = model(inputs)

        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, correct / total


@torch.no_grad()
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    inputs: torch.Tensor
    targets: torch.Tensor
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return test_loss, correct / total


for epoch in range(100):
    t1 = time.time()
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    t2 = time.time()

    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Time: {t2 - t1:.4f}s')
