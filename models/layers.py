import torch
import torch.nn as nn


class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class SpikeModule(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.ann_module = module

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        out = self.ann_module(x.reshape(B * T, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        return out


def fire_function(gamma):
    class ZIF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).float()
            ctx.save_for_backward(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).float() / gamma
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        mem = torch.zeros_like(x[:, 0])
        spikes = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = fire_function(self.gamma)(mem - self.thresh)
            mem = (1 - spike) * mem
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        out = super().forward(x.reshape(B * T, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        return out


# x = torch.randn(1, 2, 3, 4, 5)
# B, T, *spatial_dims = x.shape
# print(spatial_dims)