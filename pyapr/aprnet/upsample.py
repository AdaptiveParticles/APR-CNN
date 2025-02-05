from torch import nn
from torch.autograd import Function
from _pyaprwrapper.data_containers import APRPtrVector
from _pyaprwrapper.aprnet.upsample import upsample_const_forward, upsample_const_backward


class UpSampleConstFunction(Function):
    @staticmethod
    def forward(ctx, inputs, aprs, level_deltas, decrement_level_deltas):
        ctx.aprs = aprs
        ctx.save_for_backward(level_deltas.clone())
        output = upsample_const_forward(APRPtrVector(aprs), inputs, level_deltas)
        if decrement_level_deltas:
            level_deltas.subtract_(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        level_deltas, = ctx.saved_tensors
        grad_input = upsample_const_backward(APRPtrVector(aprs), grad_output, level_deltas)
        return grad_input, None, None, None


class UpSampleConst(nn.Module):
    def __init__(self):
        super(UpSampleConst, self).__init__()

        # If True, decrements the level_deltas tensor after the operation. Set to False in case the module is called
        # multiple times with no expected change to the level_deltas tensor, e.g. for numerical gradient tests.
        self.decrement_level_deltas = True

    def forward(self, input_features, aprs, level_deltas):
        return UpSampleConstFunction.apply(input_features, aprs, level_deltas, self.decrement_level_deltas)
