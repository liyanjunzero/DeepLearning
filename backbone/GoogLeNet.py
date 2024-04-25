import torch
import torch.nn as nn

class Inception3a(nn.Module):
    def __init__(self):
        super(Inception3a,self).__init__()
        self.conv1x1 = nn.Conv2d(192,64,1,1)
        self.conv3x3 = nn.Conv2d(96,128,3,1,1)
        self.conv5x5 = nn.Conv2d(16,32,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(192,96,1,1)
        self.conv5x5_reduce = nn.Conv2d(192,16,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(192,32,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception3b(nn.Module):
    def __init__(self):
        super(Inception3b,self).__init__()
        self.conv1x1 = nn.Conv2d(256,128,1,1)
        self.conv3x3 = nn.Conv2d(128,192,3,1,1)
        self.conv5x5 = nn.Conv2d(32,96,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(256,128,1,1)
        self.conv5x5_reduce = nn.Conv2d(256,32,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(256,64,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception4a(nn.Module):
    def __init__(self):
        super(Inception4a,self).__init__()
        self.conv1x1 = nn.Conv2d(480,192,1,1)
        self.conv3x3 = nn.Conv2d(96,208,3,1,1)
        self.conv5x5 = nn.Conv2d(16,48,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(480,96,1,1)
        self.conv5x5_reduce = nn.Conv2d(480,16,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(480,64,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception4b(nn.Module):
    def __init__(self):
        super(Inception4b,self).__init__()
        self.conv1x1 = nn.Conv2d(512,160,1,1)
        self.conv3x3 = nn.Conv2d(112,224,3,1,1)
        self.conv5x5 = nn.Conv2d(24,64,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(512,112,1,1)
        self.conv5x5_reduce = nn.Conv2d(512,24,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(512,64,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception4c(nn.Module):
    def __init__(self):
        super(Inception4c,self).__init__()
        self.conv1x1 = nn.Conv2d(512,128,1,1)
        self.conv3x3 = nn.Conv2d(128,256,3,1,1)
        self.conv5x5 = nn.Conv2d(24,64,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(512,128,1,1)
        self.conv5x5_reduce = nn.Conv2d(512,24,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(512,64,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception4d(nn.Module):
    def __init__(self):
        super(Inception4d,self).__init__()
        self.conv1x1 = nn.Conv2d(512,112,1,1)
        self.conv3x3 = nn.Conv2d(144,288,3,1,1)
        self.conv5x5 = nn.Conv2d(32,64,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(512,144,1,1)
        self.conv5x5_reduce = nn.Conv2d(512,32,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(512,64,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception4e(nn.Module):
    def __init__(self):
        super(Inception4e,self).__init__()
        self.conv1x1 = nn.Conv2d(528,256,1,1)
        self.conv3x3 = nn.Conv2d(160,320,3,1,1)
        self.conv5x5 = nn.Conv2d(32,128,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(528,160,1,1)
        self.conv5x5_reduce = nn.Conv2d(528,32,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(528,128,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)

class Inception5a(nn.Module):
    def __init__(self):
        super(Inception5a,self).__init__()
        self.conv1x1 = nn.Conv2d(832,256,1,1)
        self.conv3x3 = nn.Conv2d(160,320,3,1,1)
        self.conv5x5 = nn.Conv2d(32,128,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(832,160,1,1)
        self.conv5x5_reduce = nn.Conv2d(832,32,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(832,128,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class Inception5b(nn.Module):
    def __init__(self):
        super(Inception5b,self).__init__()
        self.conv1x1 = nn.Conv2d(832,384,1,1)
        self.conv3x3 = nn.Conv2d(192,384,3,1,1)
        self.conv5x5 = nn.Conv2d(48,128,5,1,2)
        self.conv3x3_reduce = nn.Conv2d(832,192,1,1)
        self.conv5x5_reduce = nn.Conv2d(832,48,1,1)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.pool_proj = nn.Conv2d(832,128,1,1)
    def forward(self,x):
        return torch.concat((self.conv1x1(x),
                          self.conv3x3(self.conv3x3_reduce(x)),
                          self.conv5x5(self.conv5x5_reduce(x)),
                          self.pool_proj(self.maxpool(x))),
                          dim = 1)
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3,64,7,2,3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3,2,1),
                                   nn.Conv2d(64,64,1,1),
                                   nn.ReLU(),
                                   nn.Conv2d(64,192,3,1,1),
                                   nn.MaxPool2d(3,2,1),
                                   Inception3a(),
                                   nn.ReLU(),
                                   Inception3b(),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3,2,1),
                                   Inception4a(),
                                   nn.ReLU(),
                                   Inception4b(),
                                   nn.ReLU(),
                                   Inception4c(),
                                   nn.ReLU(),
                                   Inception4d(),
                                   nn.ReLU(),
                                   Inception4e(),
                                   nn.ReLU(),
                                   nn.MaxPool2d(3,2,1),
                                   Inception5a(),
                                   nn.ReLU(),
                                   Inception5b(),
                                   nn.ReLU(),
                                   nn.AvgPool2d(7,1),
                                   nn.Dropout(0.4),
                                   nn.Flatten(),
                                   nn.Linear(1024,1000),
                                   nn.Softmax(1))
    def forward(self,x):
        x = self.model(x)
        return x


if '__main__'==__name__:
    tensor = torch.Tensor(1,3,224,224)
    block = GoogLeNet()
    pred = block(tensor)
    print(pred.shape)