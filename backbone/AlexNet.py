import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4) # real input size: 1*227*227
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096,1000)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        output = self.relu(self.conv1(x))
        output = self.pool1(output)
        output = self.relu(self.conv2(output))
        output = self.pool2(output)
        output = self.relu(self.conv3(output))
        output = self.relu(self.conv4(output))
        output = self.relu(self.conv5(output))
        output = self.pool3(output)
        output = output.view(-1,256*6*6)
        output = self.dropout(self.relu(self.fc1(output)))
        output = self.dropout(self.relu(self.fc2(output)))
        output = self.fc3(output)
        return output
    
if __name__=="__main__":
    tensor = torch.Tensor(1,1, 227, 227)
    model = AlexNet()
    pred = model(tensor)
    print(pred.shape)
