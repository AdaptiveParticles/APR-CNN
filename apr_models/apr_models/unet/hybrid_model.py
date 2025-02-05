import torch
import torch.nn as nn
from pyapr.aprnet import cuda as aprnet
from .unet_parts import ConvBlock, DownConv, UpConv
from .hybrid_unet_parts import TransitionDownConv, TransitionUpConv, PixelDownConv, PixelUpConv

class HybridAPRUNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, apr_depth=2, kernel_size=3, dims=3, activation=nn.ReLU,
                 n_filters_base=16, n_conv_per_block=2, n_stencils=2, decrement_stencils=False, num_groups=None,
                 channels_per_group=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.apr_depth = apr_depth
        assert apr_depth >= 0        

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
            if i < self.apr_depth:
                self.encoder.append(DownConv(in_channels=n_filters[i],
                                             out_channels=n_filters[i+1],
                                             nstencils=n_stencils[i+1],
                                             kernel_size=self.kernel_size,
                                             activation=activation,
                                             num_conv_blocks=n_conv_per_block,
                                             num_groups=self.num_groups or n_filters[i+1]//self.channels_per_group)
                                    )
            elif i == self.apr_depth:
                self.encoder.append(TransitionDownConv(in_channels=n_filters[i],
                                                       out_channels=n_filters[i+1],
                                                       kernel_size=self.kernel_size,
                                                       activation=activation,
                                                       num_conv_blocks=n_conv_per_block,
                                                       num_groups=self.num_groups or n_filters[i+1]//self.channels_per_group)
                                    )
            else: 
                self.encoder.append(PixelDownConv(in_channels=n_filters[i],
                                                  out_channels=n_filters[i+1],
                                                  kernel_size=self.kernel_size,
                                                  activation=activation,
                                                  num_conv_blocks=n_conv_per_block,
                                                  num_groups=self.num_groups or n_filters[i+1]//self.channels_per_group)
                                    )

        self.decoder = nn.ModuleList()
        for i in reversed(range(self.depth)):
            
            if i < self.apr_depth:
                self.decoder.append(UpConv(in_channels=n_filters[i+1],
                                           out_channels=n_filters[i],
                                           nstencils=n_stencils[i],
                                           kernel_size=self.kernel_size,
                                           activation=activation,
                                           num_conv_blocks=n_conv_per_block,
                                           num_groups=self.num_groups or n_filters[i]//self.channels_per_group)
                                    )

            elif i == self.apr_depth: 
                self.decoder.append(TransitionUpConv(in_channels=n_filters[i+1],
                                                     out_channels=n_filters[i],
                                                     nstencils=n_stencils[i],
                                                     kernel_size=self.kernel_size,
                                                     activation=activation,
                                                     num_conv_blocks=n_conv_per_block,
                                                     num_groups=self.num_groups or n_filters[i]//self.channels_per_group)
                                    )

            else:
                self.decoder.append(PixelUpConv(in_channels=n_filters[i+1],
                                                out_channels=n_filters[i],
                                                kernel_size=self.kernel_size,
                                                activation=activation,
                                                num_conv_blocks=n_conv_per_block,
                                                num_groups=self.num_groups or n_filters[i]//self.channels_per_group)
                                    )

        self.head = aprnet.APRConv(in_channels=n_filters_base,
                                   out_channels=out_channels,
                                   nstencils=n_stencils[0],
                                   kernel_size=(1, 1, 1))

    def forward(self, x, aprs, level_deltas=None):
        level_deltas = torch.zeros(len(aprs), dtype=torch.int32, device=torch.device('cpu')) if level_deltas is None else level_deltas

        x_d = [self.encoder[0](x, aprs, level_deltas)]
        
        for i, block in enumerate(self.encoder[1:]):
            if isinstance(block, (DownConv, TransitionDownConv)):
                x_d.append(block(x_d[i], aprs, level_deltas))
            else:
                x_d.append(block(x_d[i]))

        x_u = x_d[-1]
        if self.decoder:
            for i, block in enumerate(self.decoder):
                if isinstance(block, (UpConv, TransitionUpConv)):
                    x_u = block(x_u, x_d[-(i+2)], aprs, level_deltas)
                else:
                    x_u = block(x_u, x_d[-(i+2)])

        return self.head(x_u, aprs, level_deltas)


