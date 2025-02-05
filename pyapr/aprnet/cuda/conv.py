import torch
import math
from torch import nn
from torch.nn import init
from torch.autograd import Function
from _pyaprwrapper.data_containers import APRPtrVector
from _pyaprwrapper.aprnet.cuda import conv333_forward, conv333_backward, \
                                      conv111_forward, conv111_backward, \
                                      restrict_kernel333_forward, restrict_kernel333_backward


class Conv333Function(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, aprs, level_deltas):
        ctx.aprs = aprs
        ctx.num_stencils = weights.shape[0]

        num_levels = max([a.level_max()-a.level_min()+1 for a in aprs])
        weights_expanded = restrict_kernel333_forward(weights, num_levels)
        
        ctx.save_for_backward(inputs, weights_expanded, bias, level_deltas.clone())

        output = conv333_forward(APRPtrVector(aprs), inputs, weights_expanded, level_deltas)
        
        return output if bias is None else output + bias.view((1, -1, 1))

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        inputs, weights_expanded, bias, level_deltas = ctx.saved_tensors

        grad_input, grad_weights_expanded = conv333_backward(APRPtrVector(aprs), inputs, weights_expanded, grad_output, level_deltas)
        grad_bias = None if bias is None else torch.sum(grad_output, dim=(0, 2))
        grad_weights = restrict_kernel333_backward(grad_weights_expanded, ctx.num_stencils)
        return grad_input, grad_weights, grad_bias, None, None


class Conv111Function(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, aprs, level_deltas):
        ctx.aprs = aprs
        ctx.save_for_backward(inputs, weights, bias, level_deltas.clone())
        output = conv111_forward(APRPtrVector(aprs), inputs, weights, level_deltas)
        return output if bias is None else output + bias.view((1, -1, 1))

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        inputs, weights, bias, level_deltas = ctx.saved_tensors
        grad_input, grad_weights = conv111_backward(APRPtrVector(aprs), inputs, weights, grad_output, level_deltas)
        grad_bias = None if bias is None else torch.sum(grad_output, dim=(0, 2))
        return grad_input, grad_weights, grad_bias, None, None


__size2fun__ = {
    '111': Conv111Function,
    '333': Conv333Function,
}


def _kernel_size_to_function(kernel_size):
    return __size2fun__['{}{}{}'.format(*kernel_size)]


class APRConv(nn.Module):
    def __init__(self, in_channels, out_channels, nstencils=1, kernel_size=(3, 3, 3), bias=True):
        super(APRConv, self).__init__()

        assert kernel_size in [(1, 1, 1), (3, 3, 3)], 'aprnet.cuda.APRConv currently only supports kernels of size (1, 1, 1) and (3, 3, 3)'

        self.__function = _kernel_size_to_function(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nstencils = nstencils
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.Tensor(nstencils, out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, nstencils={nstencils}'
        s += ', bias={}'.format(False if self.bias is None else True)
        return s.format(**self.__dict__)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_features, aprs, level_deltas):
        return self.__function.apply(input_features, self.weight, self.bias, aprs, level_deltas)

