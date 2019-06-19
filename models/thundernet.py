import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

from modules import Conv2d, DWConv2d, SEModule

CEM_FILTER=245

def channel_shuffle(x, g):
    n, c, h, w = x.size()
    x = x.view(n, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
    return x
'''
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

'''

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, g=self.groups)



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        pw_conv11 = functools.partial(nn.Conv2d, kernel_size=1, stride=1, padding=0, bias=False)
        dw_conv33 = functools.partial(self.depthwise_conv,
                                      kernel_size=3, stride=self.stride, padding=1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                dw_conv33(inp, inp),
                nn.BatchNorm2d(inp),
                pw_conv11(inp, branch_features),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            pw_conv11(inp if (self.stride > 1) else branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            dw_conv33(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            pw_conv11(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

class CEM(nn.Module):
    """Context Enhancement Module"""
    def __init__(self, in_channels, kernel_size=1, stride=1):
        super(CEM, self).__init__()
        self.conv4 = Conv2d(in_channels, CEM_FILTER, kernel_size, bias=True)
        self.conv5 = Conv2d(in_channels, CEM_FILTER, kernel_size, bias=True)

    def forward(self, inputs):
        # in keras NHWC
        # in torch NCHW
        C4_lat = self.conv4(inputs[0])
        C5_lat = self.conv5(inputs[1])
        C5_lat = nn.UpsamplingBilinear2d(scale_factor=2)
        Cglb_lat = inputs[2].view(-1, CEM_FILTER, 1, 1)
        return C4_lat + C5_lat + Cglb_lat
        
class RPN(nn.Module):
    """region proposal network"""
    def __init__(self, in_channels=245, f_channels=256):
        super(RPN, self).__init__()
        self.num_anchors = 5*5
        self.rpn = DWConv2d(
        in_channels, f_channels, kernel_size=6,
        mid_norm_layer='default', norm_layer='default',
        activation='default')
        self.loc_conv = Conv2d(f_channels, 2*self.num_anchors, kernel_size=1, strides=1,
        padding='valid', bias=True
        )
        self.rpn_cls_pred = Conv2d(2*self.num_anchors, 4*self.num_anchors, kernel_size=1, 
        strides=1, padding='valid', bias=True
        )




class SAM(nn.Module):
    """spatial attention module"""
    def __init__(self, in_channels, kernel_size=1, stride=1):
        super(SAM, self).__init__()
        self.conv1 = Conv2d(
            in_channels, CEM_FILTER, kernel_size, padding="valid",
            norm_layer='default'
        )

    def forward(self, inputs):
        x = self.conv1(inputs[0])
        x = F.softmax(x, dim=1)
        x = x.mul(inputs[1])
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, shuffle_groups=2, with_se=False):
        super().__init__()
        self.with_se = with_se
        channels = in_channels // 2
        self.conv1 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv2 = Conv2d(
            channels, channels, kernel_size=5, groups=channels,
            norm_layer='default',
        )
        self.conv3 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        if with_se:
            self.se = SEModule(channels, reduction=8)
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x = x.contiguous()
        c = x.size(1) // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        if self.with_se:
            x2 = self.se(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shuffle_groups=2, **kwargs):
        super().__init__()
        channels = out_channels // 2
        self.conv11 = Conv2d(
            in_channels, in_channels, kernel_size=5, stride=2, groups=in_channels,
            norm_layer='default',
        )
        self.conv12 = Conv2d(
            in_channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv21 = Conv2d(
            in_channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.conv22 = Conv2d(
            channels, channels, kernel_size=5, stride=2, groups=channels,
            norm_layer='default',
        )
        self.conv23 = Conv2d(
            channels, channels, kernel_size=1,
            norm_layer='default', activation='default',
        )
        self.shuffle = ShuffleBlock(shuffle_groups)

    def forward(self, x):
        x1 = self.conv11(x)

        x1 = self.conv12(x1)

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        return x


class SNet(nn.Module):
    cfg = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }

    def __init__(self, num_classes=CEM_FILTER, version=49, **kwargs):
        super().__init__()
        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels

        self.conv1 = Conv2d(
            3, channels[0], kernel_size=3, stride=2,
            activation='default', **kwargs
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )
        self.stage2 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.stage3 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.stage4 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = Conv2d(
                channels[3], channels[4], kernel_size=1, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = [DownBlock(in_channels, out_channels, **kwargs)]
        for i in range(num_layers - 1):
            layers.append(BasicBlock(out_channels, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if len(self.channels) == 5:
            x = self.conv5(x)
        '''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        '''
        
        return x
    
def main():
    #shape = (10, 320, 320, 3) #NHWC in tf/keras
    shape = (10, 16, 320, 320)
    nx = np.random.rand(*shape).astype(np.float32)
    t = torch.Tensor(nx)
    '''
    g = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], CEM_FILTER)
    x = g(t)
    print(x.shape) #torch.Size([10, 245])
    '''
    #senet_49 = SNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = SNet().to(device)

    summary(model, (3, 320, 320))


if __name__ == "__main__":
    main()