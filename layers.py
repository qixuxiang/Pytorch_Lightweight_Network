#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import math
import numpy as np

from collections import OrderedDict

CEM_FILTER=245

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_chs, out_chs, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_chs, out_chs, kernel_size, stride, 0, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# helper method
def sconv2d(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', 0)
    if isinstance(padding, str):
        if padding.lower() == 'same':
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        else:
            # 'valid'
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class Conv2D_BN(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=1, stride=1):
        super(Conv2D_BN, self).__init__()
        self.conv = sconv2d(in_chs, out_chs, kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_chs, eps=1e-5, momentum=0.9, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class Conv2D_BN_ReLU(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=1, stride=1):
        super(Conv2D_BN_ReLU, self).__init__()
        self.conv = sconv2d(in_chs, out_chs, kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_chs, eps=1e-5, momentum=0.9, affine=True)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseConv2D_BN(nn.Module):

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, dw_stride=1):
        super(DepthwiseConv2D_BN, self).__init__()
        '''
        self.depthwise_conv2d = nn.Conv2d(in_chs, in_chs,
                                          kernel_size=dw_kernel_size,
                                          stride=dw_stride, padding='same',
                                          groups=in_chs, bias=False)
        '''
        self.depthwise_conv2d = sconv2d(in_chs, out_chs, dw_kernel_size, padding='same', groups=in_chs)
        self.bn = nn.BatchNorm2d(out_chs, eps=1e-5, momentum=0.9, affine=True)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.bn(x)
        return x

class DepthwiseConv2D_BN_POINT(nn.Module):

    def __init__(self, in_chs, out_chs, dw_kernel_size, dw_stride,
                 dw_padding):
        super(DepthwiseConv2D_BN_POINT, self).__init__()

        '''
        self.depthwise_conv2d = nn.Conv2d(in_chs, in_chs,
                                          kernel_size=dw_kernel_size,
                                          stride=dw_stride, padding='same',
                                          groups=in_chs, bias=False)
        '''
        self.depthwise_conv2d = sconv2d(in_chs, out_chs, dw_kernel_size, padding='same', groups=in_chs, **kwargs)
        self.bn = nn.BatchNorm2d(out_chs, eps=1e-5, momentum=0.9, affine=True)
        self.pointwise_conv2d = nn.Conv2d(in_chs, out_chs,
                                          kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.bn(x)
        x = self.pointwise_conv2d(x)
        return x





def main():
    



if __name__ == "__main__":
    main()