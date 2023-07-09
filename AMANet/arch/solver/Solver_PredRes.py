import os                           # 导入os模块，用于操作文件和目录
import gc                           # 导入gc模块，用于垃圾回收
import torch.nn as nn              # 导入torch.nn模块，用于神经网络的构建
import torch                        # 导入PyTorch深度学习框架
import math                         # 导入math模块，用于数学计算
import numpy as np                  # 导入numpy库，用于进行数值计算
from numpy import log10             # 导入log10函数
import tqdm                         # 导入tqdm库，用于显示进度条
import torch.optim as optim         # 导入torch.optim模块，用于优化器的选择
import scipy.io as scio             # 导入scipy库的io模块，用于读取和写入.mat文件
from sklearn.cluster import KMeans  # 从sklearn库中导入KMeans聚类算法
from sklearn.metrics import roc_auc_score, roc_curve
from arch.module.eval_utils import psnr, batch_psnr, l1_metric, l2_metric, min_max_np, calcu_result, reciprocal_metric, \
    log_metric, pairwise_l2_metric, pixel_wise_l2_metric, maxpatch_metric, loss_map, calcu_auc, plot_result,auc_metrics
# 导入自定义模块arch.module.eval_utils中的各种评估指标和计算函数
import matplotlib.pyplot as plt
from arch.module.ResUNet import UResAE  # 导入自定义模块arch.module.ResUNet中的UResAE类
from arch.module.ResNet import ResAE   # 导入自定义模块arch.module.ResNet中的ResAE类
from arch.module.Basic import BlurFunc  # 导入自定义模块arch.module.Basic中的BlurFunc类
from arch.module.Basic import video_static_motion, video_split_static_and_motion_seq  # 导入自定义模块arch.module.Basic中的函数
from arch.model.PredRes_Model import PredRes_AE_Cluster_Model  # 导入自定义模块arch.model.PredRes_Model中的PredRes_AE_Cluster_Model类
from arch.module.loss_utils import gradient_loss, gradient_metric  # 导入自定义模块arch.module.loss_utils中的函数


# 定义 Solver 类
class Solver():
    # 定义 __init__() 方法，用于初始化对象
    def __init__(self, config, cluster_model=PredRes_AE_Cluster_Model, device_idx=0, model_type='res'):
        # 初始化日志目录，并在需要时创建该目录
        self.log_dir = config.log_path
        os.makedirs(self.log_dir, exist_ok=True)
        # 检查点路径为 "model.pth"
        self.checkpoint_path = 'E15000_shanghaitech(new)_model.pth'
        # 是否使用并行模式，默认为 False
        self.para_tag = False
        # CUDA 设备 ID 列表，默认为 [0,1,2]
        self.device_ids = [0, 1, 2]
        # 评估时使用的设备 ID，默认为 [2]
        self.eval_device_idx = [2]
        # 确定所使用的计算设备（GPU 或 CPU）
        self.device = torch.device("cuda:%d" % device_idx if torch.cuda.is_available() else "cpu")
        # 图像通道数
        self.img_channel = config.img_channel
        # 视频序列的帧数
        self.frames_num = config.clips_length
        # 簇数
        self.cluster_num = config.cluster_num
        # 是否为序列数据，默认为 False
        self.seq_tag = False
        # 计算运动信息输出通道数
        if self.seq_tag:
            motion_channel_out = self.img_channel * (self.frames_num - 1)
        else:
            motion_channel_out = self.img_channel
        # 创建聚类 AE 模型对象
        self.model = cluster_model(
            static_channel_in=self.img_channel,
            static_channel_out=self.img_channel,
            static_layer_struct=config.static_layer_struct,
            static_layer_nums=config.static_layer_nums,
            motion_channel_in=self.img_channel * (self.frames_num - 1),
            motion_channel_out=motion_channel_out,
            motion_layer_struct=config.motion_layer_struct,
            motion_layer_nums=config.motion_layer_nums,
            img_channel=self.img_channel,
            frame_nums=self.frames_num,
            cluster_num=self.cluster_num,
            blur_ratio=1,
            seq_tag=self.seq_tag,
            model_type=model_type).to(self.device)
        # 初始化信息
        self.init_info()
        # 定义 L2 损失函数
        self.l2_criterion = nn.MSELoss()
        # 定义 L1 损失函数
        self.l1_criterion = nn.L1Loss()
        # 定义聚类优化器
        self.optimizer_cluster = optim.Adam(self.model.cluster_par, lr=1e-5)
        # 定义全局 AE 优化器
        self.optimizer = optim.Adam(self.model.ae_par, lr=1e-5)
        # 定义静态层优化器
        self.optimizer_static = optim.Adam(self.model.static_par, lr=5e-5)
        # 定义运动层优化器
        self.optimizer_motion = optim.Adam(self.model.motion_par, lr=5e-5)

        # 定义 train_batch_AE() 方法，用于训练 AE 模型
    def train_batch_AE(self, batch_in, alpha=None, loss_appendix=0):
        # 设置为训练模式，并将梯度缓存清零
        self.model.train()
        self.model.zero_grad()

        # 将输入批次数据移动到所选计算设备上
        batch_in = batch_in.to(self.device)
        # 前向传播计算损失和梯度
        loss_deblur, loss_predict, loss_recon, grad_deblur, grad_predict, grad_recon = self.model(batch_in, alpha,
                                                                                                      ['F'])

        # 计算预测、去模糊和重构图像的峰值信噪比
        psnr_predict = 10 * log10(1 / loss_predict.mean().item())
        psnr_deblur = 10 * log10(1 / loss_deblur.mean().item())
        psnr_recon = 10 * log10(1 / loss_recon.mean().item())

            # 计算总体损失
        loss = (loss_deblur + loss_predict + 0.01 * grad_predict)

        # 反向传播并更新网络参数
        loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 将峰值信噪比和损失添加到信息字典中
        self.info['psnr_predict'].append(psnr_predict)
        self.info['psnr_deblur'].append(psnr_deblur)
        self.info['psnr_recon'].append(psnr_recon)
        self.info['total_loss'].append(loss.mean().item())
        # 返回
        return

        # 定义 train_batch_Static() 方法，用于训练静态层
    def train_batch_Static(self, batch_in, alpha=None, loss_appendix=0):
        # 设置为训练模式，并将梯度缓存清零
        self.model.train()
        self.model.zero_grad()

        # 将输入批次数据移动到所选计算设备上
        batch_in = batch_in.to(self.device)
        # 前向传播计算损失和梯度
        loss_deblur, grad_deblur = self.model(batch_in, alpha, ['S', 'G'])

        # 计算去模糊图像的峰值信噪比
        psnr_deblur = 10 * log10(1 / loss_deblur.mean().item())

        # 计算总体损失
        loss = loss_deblur
        loss.mean().backward()
        self.optimizer_static.step()
        self.optimizer_static.zero_grad()

        # 将损失和峰值信噪比添加到信息字典中
        self.info['total_loss'].append(loss.mean().item())
        self.info['psnr_deblur'].append(psnr_deblur)
        # 返回
        return

        # 定义 train_batch_Motion() 方法，用于训练运动层
    def train_batch_Motion(self, batch_in, alpha=None, loss_appendix=0):
        # 设置为训练模式，并将梯度缓存清零
        self.model.train()
        self.model.zero_grad()

        # 将输入批次数据移动到所选计算设备上
        batch_in = batch_in.to(self.device)
        # 前向传播计算损失和梯度
        loss_predict, grad_predict = self.model(batch_in, alpha, ['M', 'G'])

        # 计算预测图像的峰值信噪比
        psnr_predict = 10 * log10(1 / loss_predict.mean().item())

        # 计算总体损失
        loss = loss_predict + grad_predict
        loss.mean().backward()
        self.optimizer_motion.step()
        self.optimizer_motion.zero_grad()

        # 释放内存
        del batch_in
        gc.collect()

        # 将峰值信噪比和损失添加到信息字典中
        self.info['psnr_predict'].append(psnr_predict)
        self.info['total_loss'].append(loss.mean().item())
        # 返回
        return

        # 定义 train_batch_Cluster() 方法，用于训练聚类层
    def train_batch_Cluster(self, batch_in, alpha=None, loss_appendix=0):
        # 设置为训练模式，并将梯度缓存清零
        self.model.train()
        self.model.zero_grad()

        # 将输入批次数据移动到所选计算设备上
        batch_in = batch_in.to(self.device)
        # 前向传播计算损失、重构距离和分类器编码
        loss_cluster, rep_dist, cat_encoder = self.model(batch_in, alpha, ['S', 'M', 'C'])

        # 计算总体损失
        loss = 0.1 * loss_cluster
        loss.sum().backward()
        self.optimizer_cluster.step()
        self.optimizer_cluster.zero_grad()

        # 将聚类损失添加到信息字典中
        self.info['cluster_loss'].append(loss_cluster.mean().item())
        # 返回
        return

        # 定义 init_Cluster() 方法，用于初始化聚类中心
    def init_Cluster(self, training_iter, alpha=None, emmbeding_length=500):
        # 输出提示信息
        print('start initial cluster centers.......')
        # 初始化嵌入向量列表
        embeddings_bank = []
        for iter_idx in range(emmbeding_length):
                # 获取下一个训练批次数据
            batch_in = next(training_iter)

            # 设置为训练模式，并将梯度缓存清零
            self.model.train()
            self.model.zero_grad()

            if self.para_tag:
                # 在多个 GPU 上计算静态表示和运动表示
                static_rep = nn.parallel.data_parallel(self.model, (batch_in, alpha, ['S', 'M', 'ini']),
                                                           device_ids=self.device_ids)
            else:
                # 将输入批次数据移动到所选计算设备上，并计算静态表示和运动表示
                batch_in = batch_in.to(self.device)
                static_rep = self.model(batch_in, alpha, ['S', 'M', 'ini'])

            # 将嵌入向量添加到列表中
            embeddings_bank.append(static_rep.detach().cpu().numpy())
        embeddings_bank = np.concatenate(embeddings_bank, 0)
        # 使用 KMeans 算法对嵌入向量进行聚类
        kmeans_model = KMeans(n_clusters=self.cluster_num, init="k-means++").fit(embeddings_bank)
        # 更新聚类中心参数
        self.model.cluster.cluster_center.data = torch.from_numpy(kmeans_model.cluster_centers_).cuda()
            # 返回
        return

        # 定义 training_info() 方法，用于输出训练信息并保存到日志文件中
    def training_info(self, detail_info):
        # 对每个训练信息键计算平均值，并添加到详细信息字符串中
        for info_keys in self.info.keys():
            if not self.info[info_keys] == []:
                detail_info += ' \t {} : {:.5f} '.format(info_keys, np.stack(self.info[info_keys]).mean())

        detail_info += '\n'

        # 初始化信息字典
        self.init_info()
        # 输出详细信息
        print(detail_info)
        # 将详细信息写入日志文件
        with open(os.path.join(self.log_dir, 'training_log.txt'), 'a+') as f:
            f.writelines(detail_info)

        # 返回
        return

        # 定义 eval_datasets() 方法，用于在评估集上进行评估
    def eval_datasets(self, dataloader, labels_list, epoch=0):
        # 设置为评估模式
        self.model.eval()
        # 初始化评估指标字典
        eval_metric_dict = {}
        eval_metric_dict['inv_recon'] = []

        # 在不更新梯度的情况下进行评估
        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(dataloader.fetch_nums)):
                # 获取下一个评估批次数据
                batch_in = dataloader.fetch()
                batch_in = batch_in.to(self.device)

                if self.seq_tag:
                    static_in, motion_in, static_target, motion_target = video_split_static_and_motion_seq(batch_in,
                                                                                                               self.img_channel,
                                                                                                               self.frames_num)
                else:
                    static_in, motion_in, static_target, motion_target = video_static_motion(batch_in,
                                                                                                 self.img_channel,
                                                                                                 self.frames_num)
                # 将梯度缓存清零
                self.model.zero_grad()
                if self.para_tag:
                    static_decoder, pred_decoder, loss_cluster_map = nn.parallel.data_parallel(self.model,(batch_in, None,['E']),
                                                                                                   device_ids=self.device_ids)
                else:
                    batch_in = batch_in.to(self.device)
                    static_decoder, pred_decoder, loss_cluster_map = self.model(batch_in, None, ['E'])

                if self.seq_tag:
                    static_in = static_in.repeat(1, self.frames_num - 1, 1, 1)
                    static_decoder = static_decoder.repeat(1, self.frames_num - 1, 1, 1)
                pred_recon = pred_decoder + static_decoder
                pred_target = static_in + motion_target
                # loss_recon = l2_metric( pred_recon , pred_target )
                loss_pixelwise_cl_re = loss_map(torch.mean((pred_recon - pred_target) ** 2, [1], keepdim=True),
                                                    loss_cluster_map)
                eval_metric_dict['inv_recon'].append(reciprocal_metric(loss_pixelwise_cl_re))
       
        # 计算 AUC 并添加到列表中
        auc_list = []
        for eval_keys in eval_metric_dict.keys():
            eval_metric = np.concatenate(eval_metric_dict[eval_keys])
            auc = calcu_result(eval_metric, labels_list, converse=False)
            auc_metrics(eval_metric, labels_list)
        # 将标签列表添加到评估指标字典中
        eval_metric_dict['labels'] = labels_list
        # 拼接详细信息字符串
        detail_info = 'Epoches {} \t  auc {:.5f} \n '.format(epoch, auc)

        # 输出详细信息
        print(detail_info)
        # 返回
        return
   
 # 定义 init_info() 方法，用于初始化信息字典
    def init_info(self):
        # 初始化信息字典中各个键对应的值
        self.info = {}
        self.info['total_loss'] = []
        self.info['cluster_loss'] = []
        self.info['psnr_deblur'] = []
        self.info['psnr_predict'] = []
        self.info['psnr_recon'] = []
        # 返回
        return

        # 定义 load_model() 方法，用于加载训练模型的参数
    def load_model(self, ):
        # 拼接模型参数文件路径
        load_path = os.path.join(self.log_dir, self.checkpoint_path)
        # 加载模型参数并更新模型
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        # 输出提示信息
        print("Checkpoint loaded from {}".format(load_path))
        # 返回
        return

        # 定义 save_model() 方法，用于保存训练模型的参数
    def save_model(self, epoch):
        # 构建模型参数字典并保存到文件中
        state = {}
        state['epoches'] = epoch
        state['state_dict'] = self.model.state_dict()
        save_path = os.path.join(self.log_dir, 'E{}_'.format(epoch) + self.checkpoint_path)
        torch.save(state, save_path)
        # 输出提示信息
        print("Checkpoint saved to {}".format(save_path))
        # 返回
        return
