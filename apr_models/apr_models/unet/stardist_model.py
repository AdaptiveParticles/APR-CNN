import torch
import torch.nn as nn
from pyapr import aprnet
from .unet_parts import ConvBlock, DownConv, UpConv



class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nstencils=1, kernel_size=(3, 3, 3), activation=nn.ReLU()):
        super().__init__()

        self.conv = aprnet.cuda.APRConv(in_channels=in_channels,
                                        out_channels=out_channels,
                                        nstencils=nstencils,
                                        kernel_size=kernel_size,
                                        bias=False)
        self.activation = activation

    def forward(self, x, aprs, level_deltas):
        return self.activation(self.conv(x, aprs, level_deltas))


class APRStardistUNet(nn.Module):
    def __init__(self, in_channels, n_rays, depth=3, kernel_size=3, dims=3, activation=nn.ReLU(inplace=True),
                 n_filters_base=16, n_conv_per_block=2, n_stencils=2, decrement_stencils=False, num_groups=None,
                 channels_per_group=None, output_level_delta=0, net_conv_after_unet=None):
        super(APRStardistUNet, self).__init__()

        self.in_channels = in_channels
        self.n_rays = n_rays
        self.depth = depth
        self.output_level_delta = output_level_delta
        self.net_conv_after_unet = net_conv_after_unet

        self.num_groups = None
        self.channels_per_group = None

        if num_groups is None and channels_per_group is None:
            self.num_groups = n_filters_base
        elif channels_per_group:
            if num_groups:
                print("APRUNet was given arguments `num_groups={}, channels_per_group={}`, using channels_per_group "
                      "- to use GroupNorm with a fixed number of groups in all stages of the network,"
                      "specify channels_per_group=None")
            self.channels_per_group = channels_per_group
        elif num_groups:
            self.num_groups = num_groups

        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * dims

        while len(self.kernel_size) < 3:
            self.kernel_size = (1, *self.kernel_size)

        if decrement_stencils:
            n_stencils = [max(n_stencils - i, 1) for i in range(depth + 1)]
        else:
            n_stencils = [max(n_stencils, 1)] * (depth + 1)

        n_filters = [n_filters_base * 2 ** i for i in range(depth + 1)]

        self.encoder = nn.ModuleList()
        self.encoder.append(ConvBlock(in_channels=in_channels,
                                      out_channels=n_filters_base,
                                      nstencils=n_stencils[0],
                                      kernel_size=self.kernel_size,
                                      activation=activation,
                                      num_blocks=n_conv_per_block,
                                      num_groups=self.num_groups or n_filters_base//self.channels_per_group)
                            )
        for i in range(self.depth):
            self.encoder.append(DownConv(in_channels=n_filters[i],
                                         out_channels=n_filters[i+1],
                                         nstencils=n_stencils[i+1],
                                         kernel_size=self.kernel_size,
                                         activation=activation,
                                         num_conv_blocks=n_conv_per_block,
                                         num_groups=self.num_groups or n_filters[i+1]//self.channels_per_group)
                                )

        self.decoder = nn.ModuleList()
        for i in reversed(range(self.output_level_delta, self.depth)):
            self.decoder.append(UpConv(in_channels=n_filters[i+1],
                                       out_channels=n_filters[i],
                                       nstencils=n_stencils[i],
                                       kernel_size=self.kernel_size,
                                       activation=activation,
                                       num_conv_blocks=n_conv_per_block,
                                       num_groups=self.num_groups or n_filters[i]//self.channels_per_group)
                                )

        if self.net_conv_after_unet is not None:
            self.conv_after_unet = ConvBlock(in_channels=n_filters[self.output_level_delta],
                                             out_channels=self.net_conv_after_unet,
                                             nstencils = n_stencils[0],
                                             kernel_size=self.kernel_size,
                                             num_blocks=1)
        else:
            self.conv_after_unet = None
        
        n_features = self.net_conv_after_unet if self.net_conv_after_unet else n_filters[self.output_level_delta]
        
        self.output_prob = OutBlock(in_channels=n_features,
                                    out_channels=1,
                                    nstencils=n_stencils[0],
                                    kernel_size=(1, 1, 1),
                                    activation=nn.Sigmoid())
        
        self.output_dist = aprnet.cuda.APRConv(in_channels=n_features,
                                               out_channels=self.n_rays,
                                               nstencils=n_stencils[0],
                                               kernel_size=(1, 1, 1))
    
    def forward(self, x, aprs, level_deltas=None):
        level_deltas = torch.zeros(len(aprs), dtype=torch.int32) if level_deltas is None else level_deltas

        x_d = [self.encoder[0](x, aprs, level_deltas)]
        for i in range(1, len(self.encoder)):
            x_d.append(self.encoder[i](x_d[i-1], aprs, level_deltas))

        if self.decoder:
            x_u = self.decoder[0](x_d[-1], x_d[-2], aprs, level_deltas)
            for i in range(1, len(self.decoder)):
                x_u = self.decoder[i](x_u, x_d[-(i+2)], aprs, level_deltas)
        else:
            x_u = x_d[-1]

        if self.conv_after_unet is not None:
            x_u = self.conv_after_unet(x_u, aprs, level_deltas)
        
        return self.output_prob(x_u, aprs, level_deltas), self.output_dist(x_u, aprs, level_deltas)


