import torch
import torch.nn as nn

class ResidualA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualA,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += x
        output = self.relu(output)
        return output
    
class ResidualB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualB,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.conv3(x)
        output = self.relu(output)
        return output
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.pool1 = nn.MaxPool2d(3,2,1)
        self.conv2_x = nn.Sequential(ResidualA(64,64),
                                     ResidualA(64,64),
                                     ResidualA(64,64))
        self.conv3_x = nn.Sequential(ResidualB(64,128,2),
                                     ResidualA(128,128),
                                     ResidualA(128,128),
                                     ResidualA(128,128))
        self.conv4_x = nn.Sequential(ResidualB(128,256,2),
                                     ResidualA(256,256),
                                     ResidualA(256,256),
                                     ResidualA(256,256),
                                     ResidualA(256,256),
                                     ResidualA(256,256))
        self.conv5_x = nn.Sequential(ResidualB(256,512,2),
                                     ResidualA(512,512),
                                     ResidualA(512,512))
        self.pool2 = nn.AvgPool2d(7,1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512,2)
    def forward(self,x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    
if __name__ == "__main__":
    tensor = torch.Tensor(1,3,224,224)
    net = ResNet34()
    out = net(tensor)
    print(out.shape)