import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        #input: 3*224*224
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2) #output: 64*112*112

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2) #output: 128*56*56

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2) #output: 256*28*28

        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2) #output: 512*14*14

        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2) #output: 512*7*7

        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool4(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(-1,512*7*7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
if __name__=='__main__':
    tensor = torch.Tensor(1,3,224,224)
    model = VGG16()
    pred = model(tensor)
    print(pred.shape)