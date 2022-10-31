from Simulation.models.unet import *
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F




class Recon(torch.nn.Module):
    def __init__(self, channel):
        super(Recon, self).__init__()
        self.deta = nn.Parameter(torch.Tensor([0.5]))
        self.eta = nn.Parameter(torch.Tensor([0.8]))
        self.channel = channel
        self.net = unetlittle(channel,channel,1)


    def x2y(self, x, Cu):

        sz = x.size()

        if len(sz) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        temp = Cu * x
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[2] + 2 * 28]).cuda()
        for t in range(28):
            y[:, :, :, 0 + 2 * t: sz[2] + 2 * t] = temp[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
        y = y/28
        return y

    def y2x(self, y,Cu):
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        x = x*Cu
        return x

    def forward(self, xt, x, Cu):
        _, _, w, h = xt.shape
        z = self.net(xt)
        yt1 = self.x2y(xt, Cu)
        xt2 = self.y2x(yt1, Cu)
        xt = (1 - self.deta * self.eta) * xt - self.deta * xt2 + self.deta * x + self.deta * self.eta * z
        return xt, z


class Network(torch.nn.Module):
    def __init__(self, LayerNo, channel):
        super(Network, self).__init__()
        self.LayerNo = LayerNo
        layer = []
        for i in range(LayerNo):
            layer.append(Recon(channel))

        self.net = nn.ModuleList(layer)

    def y2x(self, y):
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def forward(self, y,mask):

        net_input = self.y2x(y)
        xt = net_input
        for i in range(self.LayerNo):
            xt,z = self.net[i](xt, net_input, mask)
        return z


class unetlittle(nn.Module):
    def __init__(self, in_channels, ouput_channels, n_file):
        super(unetlittle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = ouput_channels
        self.n_file = n_file

        self.conv1 = conv_layer(in_channels, 48, n_file)
        self.conv2 = conv_layer(48, 48, n_file)
        self.conv3 = conv_layer(48, 48, n_file)
        self.conv4 = conv_layer(48, 48, n_file)

        self.conv5 = conv_layer(96, 96, n_file)
        self.conv6 = conv_layer(96, 96, n_file)
        self.conv7 = conv_layer(96 + in_channels, 64, n_file)
        self.conv8 = conv_layer(64, 32, n_file)
        self.conv9 = conv_layer(32, ouput_channels, n_file)

    def y2x(self, y):
        ##  Spilt operator
        sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 24, sz[2], sz[2]]).cuda()
        for t in range(24):
            temp = y[:, :, :, 0 + 2 * t: sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x

    def forward(self, x):

        skips = [x]

        x = self.conv1(x)
        x = self.conv2(x)

        _random_samples = torch.rand(x.size(0), x.size(1), 2, dtype=x.dtype, device=x.device)
        x, a = nn.FractionalMaxPool2d(kernel_size=4, output_ratio=(0.8, 0.8), return_indices=True,
                                      )(x)
        x, a = nn.FractionalMaxPool2d(kernel_size=4, output_ratio=(0.63, 0.63), return_indices=True,
                                      )(x)
        skips.append(x)
        _random_samples = torch.rand(x.size(0), x.size(1), 2, dtype=x.dtype, device=x.device)
        x = self.conv3(x)
        x, a = nn.FractionalMaxPool2d(kernel_size=4, output_ratio=(0.8, 0.8), return_indices=True,
                                      )(x)
        x, a = nn.FractionalMaxPool2d(kernel_size=4, output_ratio=(0.63, 0.63), return_indices=True,
                                      )(x)
        x = self.conv4(x)

        # -----------------------------------------------
        x = nn.functional.interpolate(x, mode='nearest', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.functional.interpolate(x, mode='nearest', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return x



