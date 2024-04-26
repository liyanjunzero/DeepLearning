#It's a Kaggle Image Classification competition
#URL:https://www.kaggle.com/competitions/image-classification-real-or-ai-generated-photo

import os
import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('../../')
from backbone.ResNet34 import ResNet34
from dataset.BinaryDataset import BinaryDataset
from PIL import Image

train_image = '../../../data/train/train'
train_label = '../../../data'

label_data = pd.read_csv(os.path.join(train_label,'train.csv'))
image_paths = [train_image + "/" + f for f in os.listdir(train_image)]
labels = label_data['Label'].tolist()

dataset = BinaryDataset(image_paths, labels)
dataloader = DataLoader(dataset,1,True)

model = ResNet34()
model.train()

epochs = 20
lr = 1e-3
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    for i, data in enumerate(dataloader):
        image, label = data
        pred = model(image)
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"batch: {i}   train loss: {loss.item()}")
    print(f"Epoch: {epoch}   train loss: {loss.item()}")
    torch.save(model.state_dict(), 'resnet34.pth')





