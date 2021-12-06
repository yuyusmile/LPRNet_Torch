import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

activate_faction = {'relu': nn.ReLU(inplace=True),
                    'relu6': nn.ReLU6(inplace=True),
                    'leak': nn.LeakyReLU(inplace=True),
                    }

class Conv(nn.Module): # k_size = 3*3, padding=1 | k_size=1*1, padding=0
    
    def __init__(self, in_plane, out_plane, k_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, bn=True, activate=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_plane,
                              out_channels = out_plane, 
                              kernel_size = k_size, 
                              stride = stride, 
                              padding=padding, 
                              dilation=dilation, 
                              groups=groups, 
                              bias=bias, 
                              )
        if bn:
            self.bn = nn.BatchNorm2d(num_features=out_plane)
        
        if activate:
            assert activate in activate_faction.keys()
            if activate == 'relu':
                self.activate = activate_faction[activate]
            elif activate == 'relu6':
                self.activate = activate_faction[activate]
            elif activate == 'leakrelu':
                self.activate = activate_faction[activate]
                
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        elif self.activate is not None:
            x = self.activate(x)
            
        return x
    

class Maxpool(nn.Module):
    
    def __init__(self, k_size, stride, padding):
        super(Maxpool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=k_size, stride=stride, padding=padding)
       
    def forward(self, x):
        x = self.pool(x)
        
        return x
    

class Small_Bais_Conv(nn.Module):
    
    def __init__(self, in_put, out_put):
        super(Small_Bais_Conv, self).__init__()
        
        self.conv = nn.Sequential(Conv(in_plane=in_put, out_plane=out_put//4, k_size=(1, 1), padding=0, activate='relu'),
                                  Conv(in_plane=out_put//4, out_plane=out_put//4, k_size=(1, 3), padding=(0, 1), activate='relu'),
                                  Conv(in_plane=out_put//4, out_plane=out_put//4, k_size=(3, 1), padding=(1, 0), activate='relu'),
                                  Conv(in_plane=out_put//4, out_plane=out_put, k_size=(1, 1), padding=0))
    
    def forward(self, x):
        
        x = self.conv(x)

        return x

