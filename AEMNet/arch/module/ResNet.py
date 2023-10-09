import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo

from arch.module.Basic import BasicBlock, conv3x3

from arch.module.Attention import Variance_Attention,CA_Block

from arch.module.memory import *


class ResEncoder(nn.Module):
    def __init__(self, block, input_channels=3, layers=[2, 2, 2, 2], layer_num=4, neck_planes=64, att_tag=False,
                  memory_module=True,bn_tag=False, last_layer_softmax=False):
        super(ResEncoder, self).__init__()
        self.neck_planes = neck_planes * 2
        self.inplanes = self.neck_planes // 2
        self.layer_num = layer_num
        self.last_layer_softmax = last_layer_softmax

        self.norm = F.normalize

        self.flow = nn.Sequential()

        self.layer0 = []
        self.layer0.append(nn.Conv2d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))
        self.layer0.append(nn.ReLU(inplace=True))

        self.flow.add_module('layer0', nn.Sequential(*self.layer0))

        for layers_idx in range(self.layer_num):
            self.flow.add_module('layer{}_bn'.format(layers_idx + 1), nn.BatchNorm2d(self.inplanes))
            self.flow.add_module('layer{}'.format(layers_idx + 1),
                                 self._make_layer(block, self.neck_planes * (2 ** (layers_idx)), layers[layers_idx],
                                                  stride=2))
            if att_tag:
                self.flow.add_module('att_layer{}'.format(layers_idx + 1),
                                    CA_Block(self.inplanes))

        if bn_tag:
            self.flow.add_module('bn', nn.BatchNorm2d(self.inplanes))
        
        if memory_module:
            self.memory_module = MemoryModule(self.inplanes,self.inplanes)
        if last_layer_softmax:
            self.proj = nn.Sequential()
            self.proj.add_module('last_layer', conv3x3(self.inplanes, self.inplanes))
            self.proj.add_module('last_act', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.flow(x)
        if self.memory_module:
            x = self.memory_module(x)
        if self.last_layer_softmax:
            x = self.proj(x)
            x = self.norm(x, p=1, dim=1)
        return x

    #  定义_make_layer函数，供构建ResNet网络使用
    #  参数说明：block表示采用的基本模块；planes表示输出通道数；blocks表示堆叠基本模块的数量；stride表示步长，可选参数，默认值为1
    def _make_layer(self, block, planes, blocks, stride=1):
        #  初始化downsample
        downsample = None
        #  判断需要不需要进行下采样以匹配维度
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride, padding=1, bias=False),
            )

        #  定义基本模块的堆叠方式
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        #  返回一个序列，其中包含了构建好的所有
        return nn.Sequential(*layers)


class ResDecoder(nn.Module):
    def __init__(self, block, output_channels=3, layers=[2, 2, 2, 2], layer_num=4, neck_planes=64, tanh_tag=False):
        super(ResDecoder, self).__init__()
        self.neck_planes = neck_planes * 2
        self.layer_num = layer_num

        self.flow = nn.Sequential()

        self.layer0 = []
        self.layer0.append(nn.Conv2d(self.neck_planes, output_channels, kernel_size=3, stride=1, padding=1, bias=False))

        layer_channel_num = [512, 256, 128, 64]
        self.inplanes = self.neck_planes * (2 ** (self.layer_num - 1))

        for layers_idx in range(self.layer_num):
            self.flow.add_module('de-layer{}'.format(layers_idx + 1),
                                 self._make_layer(block, self.neck_planes * (2 ** (self.layer_num - 1 - layers_idx)),
                                                  layers[layers_idx], stride=2))

        self.flow.add_module('back-layer', nn.Sequential(*self.layer0))
        if tanh_tag:
            self.flow.add_module('out-act', nn.Tanh())

    def forward(self, x):
        x = self.flow(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [
                nn.PixelShuffle(upscale_factor=stride),
                nn.Conv2d(self.inplanes // (stride ** 2), planes * block.expansion,
                          kernel_size=3, stride=1, padding=1, bias=False),
            ]

            downsample = nn.Sequential(*downsample)

        layers = [block(self.inplanes, planes, -1 * stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ResAE(nn.Module):
    def __init__(self, video_channels_in=3, video_channels_out=3, encoder_layers=[2, 2, 2, 2],
                 decoder_layers=[2, 2, 2, 2], layer_num=4, neck_planes=64, tanh_tag=False):
        super(ResAE, self).__init__()
        self.encoder = ResEncoder(BasicBlock, input_channels=video_channels_in, layers=encoder_layers,
                                  layer_num=layer_num, neck_planes=neck_planes)
        self.decoder = ResDecoder(BasicBlock, output_channels=video_channels_out, layers=decoder_layers,
                                  layer_num=layer_num, neck_planes=neck_planes, tanh_tag=tanh_tag)

    def forward(self, x):
        x = self.encoder(x)
        out_encoder = x
        x = self.decoder(x)
        out_decoder = x
        return out_encoder, out_decoder


if __name__ == "__main__":
    from modules import BasicBlock

    encoder = ResEncoder(BasicBlock)
    decoder = ResDecoder(BasicBlock)
    # print(model)
    input1 = torch.randn(10, 3, 224, 224)
    out_encoder = encoder(input1)
    out_decoder = decoder(out_encoder)
    print(out_encoder.shape)
    print(out_decoder.shape)
