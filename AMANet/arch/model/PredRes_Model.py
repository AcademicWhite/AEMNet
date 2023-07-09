import os

import torch.nn as nn
import torch
import math
import numpy as np
import tqdm

from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

from arch.module.ResNet import ResEncoder, ResDecoder
from arch.module.ResUNet import UResEncoder, UResDecoder
from arch.module.Basic import video_static_motion, BasicBlock, BlurFunc, video_split_static_and_motion_seq
from arch.module.cluster import EuclidDistance_Assign_Module

from arch.module.loss_utils import gradient_loss, gradient_metric
from arch.module.eval_utils import psnr, batch_psnr, l1_metric, l2_metric, min_max_np, calcu_result, reciprocal_metric, \
    log_metric, tpsnr

'''这是一个类的构造函数，用于初始化类的实例化对象。该函数包含了多个参数：

1. static_channel_in: 输入静态图像通道数，默认为3;
2. static_channel_out: 输出静态图像通道数，默认为3;
3. static_layer_struct: 静态图像网络结构，默认为[2, 2, 2, 2];
4. static_layer_nums: 静态图像网络层数，默认为4;
5. motion_channel_in: 输入动态图像通道数，默认为12;
6. motion_channel_out: 输出动态图像通道数，默认为3;
7. motion_layer_struct: 动态图像网络结构，默认为[2, 2, 2, 2];
8. motion_layer_nums: 动态图像网络层数，默认为4;
9. img_channel: 输入图像通道数，默认为3;
10. frame_nums: 序列帧长度，默认为5;
11. cluster_num: k-means聚类数目，默认为128;
12. blur_ratio: 模糊度比例，默认为3;
13. seq_tag: 是否对序列帧进行处理，默认为False;
14. model_type: 神经网络模型类型，默认为'res'。

这些参数用于定义神经网络模型的架构，从而在实例化对象后进行模型训练或预测。'''

class PredRes_AE_Cluster_Model(nn.Module):
    def __init__(self, static_channel_in=3, static_channel_out=3, static_layer_struct=[2, 2, 2, 2], static_layer_nums=4,
                 motion_channel_in=12, motion_channel_out=3, motion_layer_struct=[2, 2, 2, 2], motion_layer_nums=4,
                 img_channel=3, frame_nums=5, cluster_num=128, blur_ratio=3, seq_tag=False, model_type='res'):
        super(PredRes_AE_Cluster_Model, self).__init__()

        # 输入参数的初始化
        self.img_channel = img_channel
        self.frame_nums = frame_nums
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.cluster_num = cluster_num
        self.blur_ratio = blur_ratio
        self.seq_tag = seq_tag

        # 设置中间层的通道数
        self.neck_planes = 32
        self.inter_planes = self.neck_planes * (2 ** static_layer_nums)
        self.bulr_function = BlurFunc(ratio=blur_ratio)  # 初始化模糊函数

        self.model_type = model_type

        # 初始化运动编码器和解码器函数
        self.motion_encoder_func = UResEncoder(BasicBlock, input_channels=motion_channel_in,
                                               layers=motion_layer_struct, layer_num=motion_layer_nums,
                                               neck_planes=self.neck_planes, att_tag=True, memory_module=True,
                                               last_layer_softmax=True)
        self.motion_decoder_func = UResDecoder(BasicBlock, output_channels=motion_channel_out,
                                               layers=motion_layer_struct[::-1], layer_num=motion_layer_nums,
                                               neck_planes=self.neck_planes)

        # 初始化静态编码器和解码器函数
        self.static_encoder_func = ResEncoder(BasicBlock, input_channels=static_channel_in,
                                              layers=static_layer_struct, layer_num=static_layer_nums,
                                              neck_planes=self.neck_planes, att_tag=True, memory_module=True,last_layer_softmax=True)
        self.static_decoder_func = ResDecoder(BasicBlock, output_channels=static_channel_out,
                                              layers=static_layer_struct[::-1], layer_num=static_layer_nums,
                                              neck_planes=self.neck_planes)

        # 初始化聚类模块
        self.cluster = EuclidDistance_Assign_Module(self.inter_planes + self.inter_planes, cluster_num=self.cluster_num,
                                                    soft_assign_alpha=25.0)

        # 将自动编码器、运动部分、静态部分的参数分别存储
        self.ae_par = list(self.motion_encoder_func.parameters()) \
                      + list(self.motion_decoder_func.parameters()) \
                      + list(self.static_encoder_func.parameters()) \
                      + list(self.static_decoder_func.parameters())

        self.motion_par = list(self.motion_encoder_func.parameters()) + list(self.motion_decoder_func.parameters())
        self.static_par = list(self.static_encoder_func.parameters()) + list(self.static_decoder_func.parameters())

        self.cluster_par = list(self.cluster.parameters()) + list(
            self.motion_encoder_func.layer_list[-1].last_layer.parameters()) + list(
            self.static_encoder_func.proj.parameters())

        # 上采样函数
        self.upfunc = nn.Upsample(scale_factor=2 ** motion_layer_nums)

        # 定义损失函数
        self.l2_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

    def forward(self, x, alpha=None, stage=['G']):
        if self.seq_tag:
            static_in, motion_in, static_target, motion_target = video_split_static_and_motion_seq(x, self.img_channel,
                                                                                                   self.frame_nums)
        else:
            static_in, motion_in, static_target, motion_target = video_static_motion(x, self.img_channel,
                                                                                     self.frame_nums)

        if 'S' in stage or 'E' in stage or 'F' in stage:
            # 静态编码器
            static_encoder = self.static_encoder_func(static_in)

            # 静态解码器
            static_decoder = self.static_decoder_func(static_encoder)

            if 'G' in stage:
                # 计算去模糊损失和梯度损失
                loss_deblur = self.l2_criterion(static_decoder, static_in)
                grad_deblur = gradient_loss(static_decoder, static_in)
                return loss_deblur, grad_deblur
            elif 'S' in stage and 'E' in stage:
                # 计算去模糊图像的峰值信噪比
                deblur_psnr = tpsnr(static_decoder, static_in)
                return deblur_psnr

        if 'M' in stage or 'E' in stage or 'F' in stage:
            # 运动编码器
            motion_encoder = self.motion_encoder_func(motion_in)

            # 运动解码器
            motion_decoder = self.motion_decoder_func(motion_encoder)

            if self.seq_tag:
                static_in = static_in.repeat(1, self.frame_nums - 1, 1, 1)
            pred_target = static_in + motion_target

            if 'G' in stage:
                # 计算预测损失和梯度损失
                loss_predict = self.l2_criterion(motion_decoder + static_in, pred_target)
                grad_predict = gradient_loss(motion_decoder + static_in, pred_target)
                return loss_predict, grad_predict
            elif 'M' in stage and 'E' in stage:
                # 计算预测图像的峰值信噪比
                predict_psnr = tpsnr(motion_decoder + static_in, pred_target)
                return predict_psnr

        if 'ini' in stage:
            static_encoder_rep = static_encoder.permute(0, 2, 3, 1).contiguous()
            static_encoder_rep = static_encoder_rep.reshape(-1, static_encoder_rep.shape[-1]).contiguous()
            motion_encoder_rep = motion_encoder[0].permute(0, 2, 3, 1).contiguous()
            motion_encoder_rep = motion_encoder_rep.reshape(-1, motion_encoder_rep.shape[-1]).contiguous()
            cat_rep = torch.cat([static_encoder_rep, motion_encoder_rep], -1)
            return cat_rep

        if 'C' in stage or 'E' in stage:
            # 将静态编码器和运动编码器的特征图进行连接
            cat_encoder = torch.cat([static_encoder, motion_encoder[0]], 1)

            # 进行聚类并计算聚类损失
            rep_dist, softassign = self.cluster(cat_encoder)
            loss_cluster = torch.mean(rep_dist * softassign, [1, 2, 3])

            if 'E' not in stage:
                return loss_cluster, rep_dist, cat_encoder

        if 'F' in stage:
            # 计算去模糊损失和梯度损失
            loss_deblur = self.l2_criterion(static_decoder, static_in)
            grad_deblur = gradient_loss(static_decoder, static_in)

            if self.seq_tag:
                static_decoder = static_decoder.repeat(1, self.frame_nums - 1, 1, 1)

            # 计算预测损失和梯度损失
            loss_predict = self.l2_criterion(motion_decoder + static_in, pred_target)
            grad_predict = gradient_loss(motion_decoder + static_in, pred_target)

            # 计算重构损失和梯度损失
            pred_recon = static_decoder + motion_decoder
            loss_recon = self.l2_criterion(pred_recon, pred_target)
            grad_recon = gradient_loss(pred_recon, pred_target)

            return loss_deblur, loss_predict, loss_recon, grad_deblur, grad_predict, grad_recon

        if 'E' in stage:
            loss_cluster_map = torch.mean(rep_dist * softassign, [3], keepdim=True)
            loss_cluster_map = loss_cluster_map.permute([0, 3, 1, 2]).contiguous()
            loss_cluster_map = self.upfunc(loss_cluster_map)
            return static_decoder, motion_decoder, loss_cluster_map
