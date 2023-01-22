# -*- coding: utf-8 -*-
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch_directml
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import timm

init_model = "models/resnet18_100.pth"
model: nn.Module = timm.create_model('resnet18', pretrained=False, num_classes=10)
imsize = 32

torch.manual_seed(0)
device = torch_directml.device()
model = model.to(device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=512, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)

if init_model is not None and os.path.exists(init_model):
    checkpoint = torch.load(init_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch']
else:
    init_epoch = 0


def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    inputs: torch.Tensor
    targets: torch.Tensor

    total_batchs = len(trainloader)
    for i, (inputs, targets) in enumerate(trainloader):
        t1 = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs: torch.Tensor = model(inputs)
        t2 = time.time()

        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f"Batch: {i+1}/{total_batchs}, Loss: {loss.item():.4f}, Time: {t2 - t1:.4f}s", end="\r")
    print()

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

if not os.path.exists('models'):
    os.mkdir('models')

for epoch in range(init_epoch + 1, init_epoch + 101):
    t1 = time.time()
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    t2 = time.time()

    print("--------------------")
    print(f"Epoch: {epoch}, Time: {t2 - t1:.4f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("--------------------")

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"models/resnet18_{epoch}.pth")
