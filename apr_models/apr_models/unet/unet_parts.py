import torch
import torch.nn as nn
from pyapr.aprnet import cuda as aprnet


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nstencils=1, kernel_size=(3, 3, 3), activation=nn.ReLU, num_groups=None, norm='group', bn_momentum=0.1):
        super().__init__()

        self.conv = aprnet.APRConv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   nstencils=nstencils,
                                   kernel_size=kernel_size,
                                   bias=False)

        # if num_groups is not specified, use one group per channel
        if norm == 'group':
            self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels) if num_groups is not None else out_channels, num_channels=out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        else:
            raise ValueError('norm must be either \'group\' or \'batch\'')
        self.activation = activation if isinstance(activation, nn.Module) else activation()

    def forward(self, x, aprs, level_deltas):
        return self.activation(self.norm(self.conv(x, aprs, level_deltas)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, nstencils=1, kernel_size=(3, 3, 3),
                 activation=nn.ReLU, num_blocks=2, num_groups=None, norm='group', bn_momentum=0.1):
        super().__init__()

        mid_channels = mid_channels or out_channels
        channels = [in_channels] + [mid_channels] * (num_blocks - 1) + [out_channels]

        self.blocks = nn.ModuleList([BasicBlock(channels[i],
                                                channels[i+1],
                                                nstencils=nstencils,
                                                kernel_size=kernel_size,
                                                activation=activation,
                                                num_groups=num_groups,
                                                norm=norm,
                                                bn_momentum=bn_momentum)
                                     for i in range(num_blocks)])

    def forward(self, x, aprs, level_deltas):
        for block in self.blocks:
            x = block(x, aprs, level_deltas)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, nstencils=1, kernel_size=(3, 3, 3),
                 activation=nn.ReLU, num_conv_blocks=2, num_groups=None, norm='group', bn_momentum=0.1):
        super(DownConv, self).__init__()

        self.pool = aprnet.MaxPool()
        self.conv = ConvBlock(in_channels, out_channels, mid_channels=mid_channels, nstencils=nstencils,
                              kernel_size=kernel_size, activation=activation, num_blocks=num_conv_blocks,
                              num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x, aprs, level_deltas):
        y = self.pool(x, aprs, level_deltas)
        return self.conv(y, aprs, level_deltas)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, nstencils=1, kernel_size=(3, 3, 3),
                 activation=nn.ReLU, num_conv_blocks=2, num_groups=None, norm='group', bn_momentum=0.1):
        super(UpConv, self).__init__()

        self.reduce_ch = BasicBlock(in_channels, in_channels//2, nstencils=nstencils, kernel_size=(1, 1, 1), activation=activation, norm=norm, bn_momentum=bn_momentum) #TODO: set num_groups (breaks prev models)
        self.upsample = aprnet.UpSampleConst()
        self.conv = ConvBlock(in_channels, out_channels, mid_channels=mid_channels, nstencils=nstencils,
                              kernel_size=kernel_size, activation=activation, num_blocks=num_conv_blocks,
                              num_groups=num_groups, norm=norm, bn_momentum=bn_momentum)

    def forward(self, x1, x2, aprs, level_deltas):
        x1 = self.reduce_ch(x1, aprs, level_deltas)
        x1 = self.upsample(x1, aprs, level_deltas)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, aprs, level_deltas)
