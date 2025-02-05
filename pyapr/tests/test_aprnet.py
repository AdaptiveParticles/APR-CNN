import os
import pytest
from requests import request
import torch
import pyapr
from pyapr import aprnet
import numpy as np
from torch.autograd import gradcheck

"""
Tests the gradients of individual APRNet modules by computing the 'analytical' (computed by backward
methods) and numerical Jacobians. The tests are performed using `torch.autograd.gradcheck` 
(see https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html for details)
"""

TEST_DATA_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', 'sphere_2D.apr'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files', 'two_objects.apr')
]


def _load_data(file_idx: int, dlvl: int = 0, channels: int = 1):
    apr = pyapr.io.read_apr(TEST_DATA_PATHS[file_idx])
    num_parts = aprnet.utils.number_particles(apr, dlvl)
    x = torch.rand((channels, num_parts), dtype=torch.float64)
    x.requires_grad = True
    level_delta = dlvl * torch.ones(1, dtype=torch.int)
    return x, [apr], level_delta


TEST_DATA = {
    '2D': _load_data(0, dlvl=0, channels=1),
    '2Dds1': _load_data(0, dlvl=1, channels=1),
    '2Dds2': _load_data(0, dlvl=2, channels=1),
    '3D': _load_data(1, dlvl=0, channels=1),
    '3Dds1': _load_data(1, dlvl=1, channels=1),
    '3Dds2': _load_data(1, dlvl=2, channels=1)
}

def _generate_batch(data, channels: int = 1):
    dsets = [TEST_DATA[k] for k in data]
    apr_list = [d[1][0] for d in dsets]
    level_delta = torch.cat([d[2] for d in dsets])
    max_parts = max([d[0].shape[-1] for d in dsets])
    x = torch.rand((len(apr_list), channels, max_parts), dtype=torch.float64)
    for i, d in enumerate(dsets):
        x[i, :, d[0].shape[-1]:] = 0
    x.requires_grad = True
    return x, apr_list, level_delta


def _generate_pixel_batch(data, channels: int = 1):
    dsets = [TEST_DATA[k] for k in data]
    apr_list = [d[1][0] for d in dsets]
    level_delta = torch.cat([d[2] for d in dsets])
    max_parts = max([d[0].shape[-1] for d in dsets])
    
    z_max, x_max, y_max = 0, 0, 0
    for i, apr in enumerate(apr_list):
        lmax = apr.level_max() - level_delta[i].item()
        z_max = max(z_max, apr.z_num(lmax))
        x_max = max(x_max, apr.x_num(lmax))
        y_max = max(y_max, apr.y_num(lmax))

    x = torch.rand((len(apr_list), channels, z_max, x_max, y_max), dtype=torch.float64)
    #for i, d in enumerate(dsets):
    #    x[i, :, d[0].shape[-1]:] = 0
    x.requires_grad = True
    return x, apr_list, level_delta



TEST_CASES_2D = [
    [['2D'], 1],
    [['2Dds1', '2Dds2'], 2]
]

TEST_CASES_3D = [
    [['3D'], 1],
    [['3Dds1'], 2],
    [['3Dds2', '3Dds2'], 2]
]


"""
@pytest.mark.parametrize("_input", TEST_CASES_2D + TEST_CASES_3D)
def test_aprnet_maxpool(_input):
    model = aprnet.MaxPool().double()
    model.increment_level_deltas = False
    inputs = _generate_batch(_input[0], _input[1])
    assert gradcheck(model, inputs, eps=1e-6)


@pytest.mark.parametrize("_input", TEST_CASES_2D)
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_aprnet_conv2D(_input, kernel_size):
    model = aprnet.APRConv(in_channels=_input[1],
                           out_channels=2,
                           nstencils=3,
                           kernel_size=(1, kernel_size, kernel_size),
                           bias=True).double()
    inputs = _generate_batch(_input[0], _input[1])
    assert gradcheck(model, inputs, eps=1e-6)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_aprnet_conv3D(_input, kernel_size):
    model = aprnet.APRConv(in_channels=_input[1],
                           out_channels=1,
                           nstencils=3,
                           kernel_size=(kernel_size, kernel_size, kernel_size),
                           bias=True).double()
    inputs = _generate_batch(_input[0], _input[1])
    assert gradcheck(model, inputs, eps=1e-6)


TEST_CASES_DS = [
    [['2Dds1'], 2],
    [['2Dds1', '2Dds2'], 1],
    [['3Dds1'], 2],
    [['3Dds1', '3Dds2'], 1]
]

@pytest.mark.parametrize("_input", TEST_CASES_DS)
def test_aprnet_upsample(_input):
    model = aprnet.UpSampleConst().double()
    model.decrement_level_deltas = False
    inputs = _generate_batch(_input[0], _input[1])
    assert gradcheck(model, inputs, eps=1e-6)
"""

@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_conv333_cuda_vs_cpu(_input):
    """Compare the forward pass of `aprnet.cuda.APRConvCuda` to that of `aprnet.APRConv` for a stencil size of (3, 3, 3)"""

    in_channels = 33
    out_channels = 33
    x, apr_list, level_delta = _generate_batch(_input[0], in_channels)

    model_cuda = aprnet.cuda.APRConv(in_channels, out_channels, nstencils=3).double().cuda()

    # send data to gpu
    x = x.cuda()
    for a in apr_list:
        a.init_cuda()
    
    # compute CUDA output and gradients
    output_cuda = model_cuda(x, apr_list, level_delta)
    grad_out = torch.rand_like(output_cuda)
    output_cuda.backward(grad_out, inputs=(x, model_cuda.weight, model_cuda.bias))
    dx_cuda = x.grad.clone().detach().cpu()
    dw_cuda = model_cuda.weight.grad.clone().detach().cpu()
    db_cuda = model_cuda.bias.grad.clone().detach().cpu()
    output_cuda = output_cuda.detach().cpu()

    # CPU model for comparison
    model_cpu = aprnet.APRConv(x.shape[1], out_channels)

    # copy parameters from CUDA model
    w = model_cuda.weight.detach().cpu()
    b = model_cuda.bias.detach().cpu()
    model_cpu.weight = torch.nn.Parameter(torch.permute(w, (2, 0, 1, 3, 4, 5)).contiguous())
    model_cpu.bias = torch.nn.Parameter(b)

    # compute CPU output and gradients
    x = x.cpu()
    x.grad = None
    output_cpu = model_cpu(x, apr_list, level_delta)
    grad_out = grad_out.cpu()
    output_cpu.backward(grad_out, inputs=(x, model_cpu.weight, model_cpu.bias))
    dx_cpu = x.grad.clone().detach()
    dw_cpu = model_cpu.weight.grad.clone().detach()
    db_cpu = model_cpu.bias.grad.clone().detach()

    assert torch.allclose(output_cuda, output_cpu)
    assert torch.allclose(dw_cuda, dw_cpu.permute((1, 2, 0, 3, 4, 5)))
    assert torch.allclose(db_cuda, db_cpu)
    assert torch.allclose(dx_cuda, dx_cpu)



@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_maxpool_cuda_gradcheck(_input):
    model = aprnet.cuda.MaxPool().double()
    model.increment_level_deltas = False
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])

    x.requires_grad = False
    for i in range(x.shape[2]):
        x[0, :, i] += 10*i
    x.requires_grad = True
    x = x.cuda()

    for a in apr_list:
        a.init_cuda()
    
    assert gradcheck(model, (x, apr_list, level_delta), eps=1e-3)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_conv333_cuda_gradcheck(_input):
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])
    model = aprnet.cuda.APRConv(x.shape[1], 1, kernel_size=(3, 3, 3), nstencils=2).double().cuda()
    x = x.cuda()

    for a in apr_list:
        a.init_cuda()
    
    assert gradcheck(model, (x, apr_list, level_delta), eps=1e-5)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_conv111_cuda_gradcheck(_input):
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])
    model = aprnet.cuda.APRConv(x.shape[1], 1, kernel_size=(1, 1, 1), nstencils=6).double().cuda()
    x = x.cuda()

    for a in apr_list:
        a.init_cuda()
    
    assert gradcheck(model, (x, apr_list, level_delta), eps=1e-5)


TEST_CASES_3D_DS = [
    [['3Dds1'], 2],
    [['3Dds1', '3Dds2'], 1]
]


@pytest.mark.parametrize("_input", TEST_CASES_3D_DS)
def test_upsample_cuda_gradcheck(_input):
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])
    model = aprnet.cuda.UpSampleConst().double()
    model.decrement_level_deltas = False
    
    x.requires_grad = False
    x += torch.arange(0, x.numel(), 1).reshape(x.shape)
    x.requires_grad = True
    x = x.cuda()

    for a in apr_list:
        a.init_cuda()
    
    assert gradcheck(model, (x, apr_list, level_delta), eps=1e-3)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_reconstruct_cuda_gradcheck(_input):
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])
    model_cuda = aprnet.cuda.Reconstruct()

    for a in apr_list:
        a.init_cuda()
    x = x.cuda()

    assert gradcheck(model_cuda, (x, apr_list, level_delta), eps=1e-5)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_sample_particles_cuda_gradcheck(_input):
    x, apr_list, level_delta = _generate_pixel_batch(_input[0], _input[1])
    model_cuda = aprnet.cuda.SampleParticles()

    for a in apr_list:
        a.init_cuda()
    x = x.cuda()

    assert gradcheck(model_cuda, (x, apr_list, level_delta), eps=1e-5)


@pytest.mark.parametrize("_input", TEST_CASES_3D)
def test_reconstruct_sample_cuda(_input):
    x, apr_list, level_delta = _generate_batch(_input[0], _input[1])
    recon = aprnet.cuda.Reconstruct()
    sample = aprnet.cuda.SampleParticles()
    
    for a in apr_list:
        a.init_cuda()
    x = x.cuda()

    y = recon(x, apr_list, level_delta)
    y = sample(y, apr_list, level_delta)

    assert torch.allclose(x, y)

