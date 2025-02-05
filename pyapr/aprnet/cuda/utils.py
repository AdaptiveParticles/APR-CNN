import torch
import math
from torch import nn
from torch.nn import init
from torch.autograd import Function
from _pyaprwrapper.data_containers import APRPtrVector
from _pyaprwrapper.aprnet.cuda import reconstruct_forward, reconstruct_backward, sample_particles_forward, sample_particles_backward


class ReconstructFunction(Function):
    @staticmethod
    def forward(ctx, inputs, aprs, level_deltas):
        ctx.aprs = aprs
        ctx.save_for_backward(level_deltas.clone())
        output = reconstruct_forward(APRPtrVector(aprs), inputs, level_deltas)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        level_deltas, = ctx.saved_tensors
        grad_input = reconstruct_backward(APRPtrVector(aprs), grad_output, level_deltas)
        return grad_input, None, None

class Reconstruct(nn.Module):
    def __init__(self):
        super(Reconstruct, self).__init__()

    def forward(self, input_features, aprs, level_deltas=None):
        level_deltas = torch.zeros(len(aprs), dtype=torch.int32) if level_deltas is None else level_deltas
        return ReconstructFunction.apply(input_features, aprs, level_deltas)


class SampleParticlesFunction(Function):
    @staticmethod
    def forward(ctx, inputs, aprs, level_deltas):
        ctx.aprs = aprs
        ctx.save_for_backward(level_deltas.clone())
        output = sample_particles_forward(APRPtrVector(aprs), inputs, level_deltas)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aprs = ctx.aprs
        level_deltas, = ctx.saved_tensors
        grad_input = sample_particles_backward(APRPtrVector(aprs), grad_output, level_deltas)
        return grad_input, None, None

class SampleParticles(nn.Module):
    def __init__(self):
        super(SampleParticles, self).__init__()

    def forward(self, input_features, aprs, level_deltas=None):
        level_deltas = torch.zeros(len(aprs), dtype=torch.int32) if level_deltas is None else level_deltas
        return SampleParticlesFunction.apply(input_features, aprs, level_deltas)

