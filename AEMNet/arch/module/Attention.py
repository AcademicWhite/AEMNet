import sys
import os
# from datasets_sequence import multi_train_datasets, multi_test_datasets
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init




def softmax_normalization(x, func):
    b,c,h,w = x.shape
    x_re = x.view([b,c,-1])
    x_norm = func(x_re)
    x_norm = x_norm.view([ b,c,h,w ])
    return x_norm

class Variance_Attention(nn.Module):
    def __init__(self, depth_in, depth_embedding,maxpool=1):
        super(Variance_Attention, self).__init__()     
        self.flow = nn.Sequential()
        self.flow.add_module('proj_conv', nn.Conv2d(depth_in, depth_embedding, kernel_size=1,padding=False, bias=False))
        self.maxpool = maxpool
        if not maxpool == 1:
            self.flow.add_module('pool', nn.AvgPool2d( kernel_size=maxpool, stride = maxpool ))
            self.unpool = nn.Upsample(scale_factor = maxpool)
        self.norm_func = nn.Softmax(-1)

    def forward(self,x):
        proj_x = self.flow(x)
        mean_x = torch.mean(proj_x, dim=1, keepdim=True)
        variance_x = torch.sum(torch.pow(proj_x - mean_x, 2) , dim = 1 , keepdim=True)
        var_norm = softmax_normalization(variance_x, self.norm_func)
        if not self.maxpool == 1:
            var_norm = self.unpool(var_norm)
        
        return torch.exp(var_norm)*x 
    
class CA_Block(nn.Module):
    def __init__(self,channel,reduction=16):
        super(CA_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel,out_channels=channel//reduction,kernel_size=1,stride=1,bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
        self.F_h = nn.Conv2d(in_channels=channel//reduction,out_channels=channel,kernel_size=1,stride=1,bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self,x):
        # batch_size,c,h,w
        _,_,h,w = x.size()
        # batch_size,c,h,w => batch_size,c,h,1 => batch_size,c,1,h
        x_h = torch.mean(x,dim = 3,keepdim=True).permute(0,1,3,2)
        # batch_size,c,h,w => batch_size,c,1,w
        x_w = torch.mean(x,dim = 2,keepdim=True)
        # batch_size,c,1,w cat batch_size,c,1,h => batch_size,c,1,w+h
        # batch_size,c,1,w+h => batch_size,c/r,1,w+h
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h,x_w),3))))
        # batch_size,c/r,1,w+h => batch_size,c/r,1,h & batch_size,c/r,1,w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h,w],3)
        # batch_size,c/r,1,h => batch_size,c/r,h,1 => batch_size,c,h,1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0,1,3,2)))
        # batch_size,c/r,1,w => batch_size,c,1,w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return  out  
    
if __name__ == "__main__":
    xx = torch.rand(1,8,16,16)
    var_att = Variance_Attention(8,16,maxpool=2)
    yy = var_att(xx)
    print(yy)
    # 测试通道注意力块
    batch_size = 32
    channel = 64
    height = 16
    width = 16

    # 创建随机张量
    x = torch.randn(batch_size, channel, height, width)

    # 实例化通道注意力块
    ca_block = CA_Block(channel)

    # 通过通道注意力块运行输入张量
    out = ca_block(x)

    # 验证输出张量形状是否正确
    assert out.shape == x.shape, "Output tensor shape incorrect"
    print("Output tensor shape correct")