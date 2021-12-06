import torch as nn
import torch
import torch.nn.functional as F
import torchvision
from conv import * 

class LPR(nn.Module):
    
    def __init__(self, num_chars):
        super(LPR, self).__init__()
        self.num_chars = num_chars
        
        self.conv1 = nn.Sequential(Conv(in_plane=3, out_plane=32, k_size=3),
                                 Maxpool(k_size=3, stride=1, padding=1),
                                 Small_Bais_Conv(in_put=32, out_put=64),
                                 )
        
        
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True),
                                   Maxpool(k_size=3, stride=(1, 2), padding=1),
                                   Small_Bais_Conv(in_put=64, out_put=128),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ReLU(inplace=True),
                                   Small_Bais_Conv(in_put=128, out_put=256))
        
        
        self.conv3 = nn.Sequential(nn.ReLU(inplace=True),
                                   Maxpool(k_size=3, stride=(1, 3), padding=1),
                                   nn.Dropout2d(inplace=True, p=0.5),
                                   Conv(in_plane=256, out_plane=256, k_size=(1, 4), padding=(0, 1)),
                                   nn.Dropout2d(inplace=True, p=0.5),
                                   nn.BatchNorm2d(num_features=256),
                                   nn.ReLU(inplace=True),
                                   Conv(in_plane=256, out_plane=num_chars, k_size=(13, 1), padding=(6, 0), activate='relu'))
        
     
        self.conv4 = nn.AvgPool2d(kernel_size=(1, 6), stride=(1, 6))
        self.conv5 = nn.AvgPool2d(kernel_size=(1, 6), stride=(1, 6))
        self.conv6 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv7 = Conv(in_plane=256+self.num_chars+3+64, out_plane=self.num_chars, k_size=1, padding=0)
        
        
    def forward(self, x):
        
        result2_0 = self.conv4(x)
        result2 = torch.mean(torch.pow(result2_0, 2))
        result2 = torch.div(result2_0, result2)
        
        x = self.conv1(x)
        x1 = x
        x = self.conv2(x)
        x2 = x
        x = self.conv3(x)
        
        result1 = torch.mean(torch.pow(x, 2))
        result1 = torch.div(x, result1)
        
        result3_0 = self.conv5(x1)
        result3 = torch.mean(torch.pow(result3_0, 2))
        result3 = torch.div(result3_0, result3)
        
        result4_0 = self.conv6(x2)
        result4 = torch.mean(torch.pow(result4_0, 2))
        result4 = torch.div(result4_0, result4)
        
        x = torch.cat([result1, result2, result3, result4], dim=1)
        
        x = self.conv7(x)
        logits = torch.mean(x, dim=2)
        
        return logits


if __name__ == "__main__":
    net = LPR(num_chars=68)
    print(net)