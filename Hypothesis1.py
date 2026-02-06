# %%
#Importing all the required modules

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_url
from typing import List, Tuple, Dict
import pandas as pd
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

torch.manual_seed(42)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(42)
    cudnn.benchmark = True

# %%
#function to demonstrate normalization mechanics
def demonstrate_normalization_mechanics():
    N, C, H, W = 2, 4, 2, 2
    x = torch.randn(N, C, H, W)

    bn_mean = x.mean(dim=(0, 2, 3), keepdim=True)
    print(f"Batch Norm Mean Shape: {bn_mean.shape}")

    ln_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    print(f"Layer Norm Mean Shape: {ln_mean.shape}")

    G = 2
    x_reshaped = x.view(N, G, C // G, H, W)
    gn_mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
    print(f"Group Norm Mean Shape: {gn_mean.shape}")

demonstrate_normalization_mechanics()

# %%
#class to define the ResNet architecture with different normalization techniques
class XBNBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type='GN'):
        super(XBNBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if norm_type == 'GN':
            self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes)
        elif norm_type == 'LN':
            self.bn2 = nn.GroupNorm(num_groups=1, num_channels=planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

#class to define the ResNet model for ImageNet dataset
class ResNetImageNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, norm_type='GN'):
        super(ResNetImageNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_type=norm_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_type=norm_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, norm_type='GN'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# %%
#Getting the data loaders for the Imagenette dataset
def get_imagenette_loaders(batch_size):
    data_dir = './data/imagenette'
    if not os.path.exists(data_dir):
        print("Downloading ImageNette (160px version)...")
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        download_url(url, root='./data', filename="imagenette.tgz", md5=None)
        with tarfile.open('./data/imagenette.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        os.rename('./data/imagenette2-160', data_dir)

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train', transform=train_tfms)
    validset = torchvision.datasets.ImageFolder(root=f'{data_dir}/val', transform=valid_tfms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size*2, shuffle=False, num_workers=2)

    return trainloader, validloader

#Using the data loaders and visualizing some sample images
trainloader, validloader = get_imagenette_loaders(batch_size=32)
print("Dataset Summary")
print(f"Training Dataset Size: {len(trainloader.dataset):,} images")
print(f"Validation Dataset Size: {len(validloader.dataset):,} images")

def unnormalize(img_tensor):
    stats_mean = np.array([0.485, 0.456, 0.406])
    stats_std = np.array([0.229, 0.224, 0.225])
    img_array = img_tensor.cpu().numpy()
    img_array = img_array * stats_std[:, None, None] + stats_mean[:, None, None]
    return np.transpose(np.clip(img_array, 0, 1), (1, 2, 0))

dataiter = iter(validloader)
images, labels = next(dataiter)
class_names = validloader.dataset.classes

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(len(axes)):
    if i >= len(images): break
    axes[i].imshow(unnormalize(images[i]))

    full_name = class_names[labels[i]]
    name_parts = full_name.split('_')
    display_name = name_parts[1] if len(name_parts) > 1 else full_name

    axes[i].set_title(display_name)
    axes[i].axis('off')

plt.suptitle('Sample Training Data', fontsize=14)
plt.tight_layout()
plt.show()

# %%
#Function to train and evaluate the model
def train_experiment(norm_type_to_test):
    print(f"Starting to train XBNBlock with {norm_type_to_test}")

    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.001

    trainloader, validloader = get_imagenette_loaders(BATCH_SIZE)
    model = ResNetImageNet(XBNBlock, [3, 4, 6, 3], num_classes=10, norm_type=norm_type_to_test)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return best_val_acc

# %%
#Conducting the experiments
results = {}

print("Reproducing Baseline (Standard BN)")
results['BN'] = train_experiment('BN')

print("\n Reproducing Paper Results (XBNBlock-GN)")
results['GN'] = train_experiment('GN')

print("\n 3. Testing hypothesis (XBNBlock-LN)")
results['LN'] = train_experiment('LN')

# %%
#Measuring model parameters and speed
def measure_speed(model, input_tensor):
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(50):
            _ = model(input_tensor)
        end = time.time()
    return (end - start) / 50 * 1000

print(f"\n{'NORM':<5} | {'PARAMS (M)':<10} | {'SPEED (ms)':<10}")
dummy = torch.randn(1, 3, 224, 224).to(DEVICE)

for norm in ['BN', 'GN', 'LN']:
    m = ResNetImageNet(XBNBlock, [3, 4, 6, 3], num_classes=10, norm_type=norm).to(DEVICE)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    speed = measure_speed(m, dummy)
    print(f"{norm:<5} | {params:.2f} M     | {speed:.2f} ms")

# %%
#Displaying the results
print("\nAccuracy Report:")
for k, v in results.items():
    print(f"{k}: {v:.2f}%")

names = list(results.keys())
values = list(results.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(names, values, color=['red', 'yellow', 'green'])

plt.ylim(min(values)-2, max(values)+2)
plt.ylabel('Validation Accuracy (%)')
plt.title('Hypothesis 1: Impact of Blocking Mechanism Type on Performance')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.show()

def create_h1_summary_table(res):
    acc_bn = res['BN']
    acc_gn = res['GN']
    acc_ln = res['LN']

    data = {
        "Method": ["Standard BN (Control)", "Group Norm (Paper)", "Layer Norm (Hypothesis)"],
        "Top-1 Accuracy": [
            f"{acc_bn:.2f}%",
            f"{acc_gn:.2f}%",
            f"{acc_ln:.2f}%"
        ],
        "BN": [
            "—",
            f"{acc_gn - acc_bn:+.2f}%",
            f"{acc_ln - acc_bn:+.2f}%"
        ],
        "GN": [
            "—",
            "—",
            f"{acc_ln - acc_gn:+.2f}%"
        ]
    }

    df = pd.DataFrame(data)
    return df

df_summary = create_h1_summary_table(results)

print("\n HYPOTHESIS 1: SUMMARY TABLE")
print(df_summary.to_markdown(index=False))

print("\n Conclusion")
if results['GN'] > results['BN']:
    print("Successfully reproduced paper claim (GN > BN).")
    print("The Group Norm concept works better than standard Batch Norm.")
else:
    print("Paper claim not reproduced on this dataset size.")

if results['LN'] >= results['GN']:
    print("Finding: Hypothesis supported. Layer Norm is competitive/superior.")
else:
    print("Result: Hypothesis Disproved. Group Norm is superior.")
    print("Layer Norm lost distinct feature information.")


