import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        #input shape is 3@32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        #shape: 6@28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #shape: 6@14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        #shape: 16@10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #shape: 16@5x5
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

if __name__=='__main__':
    tensor = torch.Tensor(3,32,32)
    model = LeNet()
    output = model(tensor)
    print(output.shape)