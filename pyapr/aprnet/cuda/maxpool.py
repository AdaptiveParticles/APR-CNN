import torch
from torch import nn
from torch.autograd import Function
from _pyaprwrapper.data_containers import APRPtrVector
from _pyaprwrapper.aprnet.cuda import maxpool_forward_cuda, maxpool_backward_cuda


class MaxPoolFunction(Function):
    @staticmethod
    def forward(ctx, inputs, aprs, level_deltas, increment_level_deltas):
        ctx.aprs = aprs
        
        output, max_indices = maxpool_forward_cuda(APRPtrVector(aprs), inputs, level_deltas)

        ctx.save_for_backward(max_indices, level_deltas.clone())

        if increment_level_deltas:
            level_deltas.add_(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        max_indices, level_deltas = ctx.saved_tensors

        grad_input = maxpool_backward_cuda(APRPtrVector(aprs), grad_output, max_indices, level_deltas)

        return grad_input, None, None, None


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

        # If True, increments the level_deltas tensor after the operation. Set to False in case the module is called
        # multiple times with no expected change to the level_deltas tensor, e.g. for numerical gradient tests.
        self.increment_level_deltas = True

    def forward(self, input_features, aprs, level_deltas):
        return MaxPoolFunction.apply(input_features, aprs, level_deltas, self.increment_level_deltas)

