from cmath import log10
# 从cmath模块导入log10函数
import torch
# 导入PyTorch深度学习框架
import torch.nn as nn
# 导入torch.nn模块，用于神经网络的构建
import numpy as np
# 导入numpy库，用于进行数值计算
import cv2


def psnr(pred, gt):
    # 定义PSNR函数，用于计算峰值信噪比
    return 10 * log10(1 / torch.sum((pred - gt) ** 2).item())
    # 返回计算得到的峰值信噪比值


# 假定五帧
'''前四帧为运动部分(motion_in)，第一帧为静态部分(static_in)
    运动目标=最后一帧-第一帧(motion_target)   静态目标=第一帧(static_in)'''
def video_static_motion(frames, img_channel, frames_num):
    motion_in = frames[:, 0 * img_channel:(frames_num - 1) * img_channel, :, :]
    static_in = frames[:, 0 * img_channel:1 * img_channel, :, :]
    motion_target = frames[:, (frames_num - 1) * img_channel:, :, :] - static_in
    static_target = static_in
    return static_in, motion_in, static_target, motion_target



# 假定五帧
'''前四帧为运动部分(motion_in)，最后一帧为静态部分(static_in)
   运动目标：多帧动态图像中的每一帧与静态图像之间的差异(motion_in) 静态目标=最后一帧(static_in) '''
def video_split_static_and_motion_seq(frames, img_channel, frames_num):
    motion_in = frames[:, 0 * img_channel:(frames_num - 1) * img_channel, :, :]
    static_in = frames[:, (frames_num - 1) * img_channel:, :, :]
    motion_target = motion_in - static_in.repeat(1, frames_num - 1, 1, 1)
    static_target = static_in
    return static_in, motion_in, static_target, motion_target


def conv3x3(in_planes, out_planes, stride=1):
    """带有填充的3x3卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_up3x3(in_planes, out_planes, stride=1):
    """带有填充的3x3卷积（上采样）"""

    downsample = [
        nn.PixelShuffle(upscale_factor=stride),
        nn.Conv2d(in_planes // (stride ** 2), out_planes, kernel_size=3, stride=1, padding=1, bias=False),
    ]
    downsample = nn.Sequential(*downsample)

    return downsample


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if stride == -2:
            self.conv1 = conv_up3x3(inplanes, planes, -1 * stride)  # 采用上采样卷积操作
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)  # 采用普通的卷积操作

        self.relu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)  # 定义第二个卷积层

        self.downsample = downsample  # 下采样操作，用于调整残差块的维度匹配
        self.stride = stride  # 定义卷积的步长

    def forward(self, x):
        residual = x  # 将输入特征图保存为残差
        out = self.conv1(x)  # 第一个卷积层
        out = self.relu(out)  # 应用激活函数
        out = self.conv2(out)  # 第二个卷积层

        if self.downsample is not None:
            residual = self.downsample(x)  # 下采样操作，调整残差块的维度匹配

        out += residual  # 将下采样后的特征图与经过卷积的输出特征图相加
        out = self.relu(out)  # 再次应用激活函数
        return out



class BlurFunc(nn.Module): #BlurFunc 是一个模糊函数，用于在图像处理中进行模糊操作
    def __init__(self, ratio=2):
        super(BlurFunc, self).__init__()
        self.down = nn.AvgPool2d(ratio, ratio)  # 定义下采样操作为平均池化
        self.up = nn.Upsample(scale_factor=ratio, mode='bilinear', align_corners=False)  # 定义上采样操作为双线性插值

    def forward(self, x):
        x = self.down(x)  # 对输入特征图进行下采样
        x = self.up(x)  # 对下采样后的特征图进行上采样
        return x


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置可见的GPU设备

    print('#### Test Case ###')
    model = BlurFunc().cuda()  # 创建一个模糊函数的实例，并将其移动到GPU上
    x = torch.rand(2, 12, 256, 256).cuda()  # 创建输入张量，并将其移动到GPU上
    out = model(x)  # 调用模糊函数的前向传播，对输入进行模糊操作
    print(out.shape)  # 打印输出张量的形状
