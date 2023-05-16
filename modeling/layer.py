import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 relu=True, 
                 same_padding=False, 
                 padding=(0,0),
                 bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        if same_padding:
          padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResidualBlock2D(nn.Module):

  def __init__(self, planes, kernel_size=(11,5), padding=(5,2), downsample=True):
    super(ResidualBlock2D, self).__init__()
    self.c1 = nn.Conv2d(planes,   planes,   kernel_size=1, stride=1, bias=False)
    self.b1 = nn.BatchNorm2d(planes)
    self.c2 = nn.Conv2d(planes,   planes*2, kernel_size=kernel_size, stride=1,
                  padding=padding, bias=False)
    self.b2 = nn.BatchNorm2d(planes*2)
    self.c3 = nn.Conv2d(planes*2, planes*4, kernel_size=1, stride=1, bias=False)
    self.b3 = nn.BatchNorm2d(planes * 4)
    self.downsample = nn.Sequential(
        nn.Conv2d(planes,   planes*4,   kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(planes*4),
    )
    self.relu  = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.c1(x)
    out = self.b1(out)
    out = self.relu(out)

    out = self.c2(out)
    out = self.b2(out)
    out = self.relu(out)

    out = self.c3(out)
    out = self.b3(out)

    if self.downsample:
        identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out
    
class KMersNet(nn.Module):
  def __init__(self, base_channel, out_channels):
    super().__init__()

    if base_channel is None:
      base_channel = 8
    if out_channels is None:
      out_channels = 32
    # TODO: should we use padding here
    self.conv = Conv2d(1, base_channel, 
                       kernel_size=(3, 5),
                       bn=True, same_padding=True)
    
    self.res2d = ResidualBlock2D(base_channel, kernel_size=(3, 5), padding=(1, 2)) 
    self.fc = nn.Linear(base_channel*4*4, out_channels)
    # self.bn = nn.BatchNorm1d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, input):
    x = self.conv(input) # (N,C=1,Seq,4) => (N,C,Seq,4)
    x = F.dropout(x, 0.1, training=self.training)

    x = self.res2d(x) # (N,C,Seq,4) => (N,C*4,Seq,4)
    x = F.dropout(x, 0.5, training=self.training)

    x = x.permute(0,2,1,3).contiguous() # (N,C*4,Seq,4) => (N,Seq,C*4,4)
    x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # (N,Seq,C*4*4)
    
    x = self.fc(x) # (N,Seq,C')
    # x = self.bn(x)
    x = self.relu(x)
    return x












