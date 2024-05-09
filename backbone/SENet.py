import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channel, rate=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_channel, in_channel//rate),
                                         nn.ReLU(),
                                         nn.Linear(in_channel//rate, in_channel),
                                         nn.Sigmoid()
                                         )
    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        attention = self.squeeze(x).view(batch_size, channels)
        attention = self.excitation(attention).view(batch_size, channels).view(batch_size, channels, 1, 1)
        print(attention)
        print(attention.expand_as(x))
        x *= attention.expand_as(x)
        return x
    
if __name__=='__main__':
    x = torch.ones(64,64,224,224)
    net = SEBlock(64)
    x = net(x)
    print(net)
    print(x.shape)