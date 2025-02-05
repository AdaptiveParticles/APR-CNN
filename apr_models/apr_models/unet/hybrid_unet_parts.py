import torch
import torch.nn as nn
import torch.nn.functional as F
from pyapr.aprnet import cuda as aprnet
from .unet_parts import ConvBlock


class PixelBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), activation=nn.ReLU(), norm='group', num_groups=None, bn_momentum=0.1):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding='same',
                              bias=False)
        if norm == 'group':
            self.norm = nn.GroupNorm(num_groups=num_groups or out_channels, num_channels=out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        else:
            raise ValueError('norm must be either \'group\' or \'batch\'')
        self.activation = activation if isinstance(activation, nn.Module) else activation()

    def forward(self, x):        
        return self.activation(self.norm(self.conv(x)))


class PixelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3, 3, 3),
                 activation=nn.ReLU(), num_blocks=2, norm='group', num_groups=None, bn_momentum=0.1):
        super().__init__()

        mid_channels = mid_channels or out_channels
        channels = [in_channels] + [mid_channels] * (num_blocks - 1) + [out_channels]

        self.blocks = nn.ModuleList([PixelBasicBlock(channels[i],
                                                     channels[i+1],
                                                     kernel_size=kernel_size,
                                                     activation=activation,
                                                     norm=norm,
                                                     num_groups=num_groups,
                                                     bn_momentum=bn_momentum)
                                     for i in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PixelDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3, 3, 3), activation=nn.ReLU(),
                 num_conv_blocks=2, norm='group', num_groups=None, bn_momentum=0.1):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv = PixelConvBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, 
                                   activation=activation, num_blocks=num_conv_blocks, num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x):
        return self.conv(self.pool(x))


class TransitionDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3, 3, 3), activation=nn.ReLU(),
                 num_conv_blocks=2, norm='group', num_groups=None, bn_momentum=0.1):
        super().__init__()

        self.pool = aprnet.MaxPool()
        self.recon = aprnet.Reconstruct()
        self.conv = PixelConvBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, 
                                   activation=activation, num_blocks=num_conv_blocks, num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x, aprs, level_deltas):
        y = self.pool(x, aprs, level_deltas)
        y = self.recon(y, aprs, level_deltas)
        return self.conv(y)


class PixelUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3, 3, 3), activation=nn.ReLU(),
                 num_conv_blocks=2, num_groups=None, norm='group', bn_momentum=0.1):
        super().__init__()

        self.reduce_ch = PixelBasicBlock(in_channels, in_channels//2, kernel_size=(1,1,1), activation=activation, num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = PixelConvBlock(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, 
                                   activation=activation, num_blocks=num_conv_blocks, num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x1, x2):
        x1 = self.reduce_ch(x1)
        x1 = self.upsample(x1)
        
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2,
                        diffZ//2, diffZ-diffZ//2),
                   mode='constant', value=0)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TransitionUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, nstencils=1, kernel_size=(3, 3, 3),
                 activation=nn.ReLU, num_conv_blocks=2, num_groups=None, norm='group', bn_momentum=0.1):
        super().__init__()

        self.reduce_ch = PixelBasicBlock(in_channels, in_channels//2, kernel_size=(1, 1, 1), activation=activation, num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)
        self.sample = aprnet.SampleParticles()
        self.upsample = aprnet.UpSampleConst()
        self.conv = ConvBlock(in_channels, out_channels, mid_channels=mid_channels, nstencils=nstencils,
                              kernel_size=kernel_size, activation=activation, num_blocks=num_conv_blocks,
                              num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x1, x2, aprs, level_deltas):
        x1 = self.reduce_ch(x1)
        x1 = self.sample(x1, aprs, level_deltas)
        x1 = self.upsample(x1, aprs, level_deltas)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, aprs, level_deltas)



