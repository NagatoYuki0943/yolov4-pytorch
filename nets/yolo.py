"""
LeakyReLU
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53

#-------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化,增大感受野
#   池化后和输入数据进行维度堆叠
#   pool_sizes=[1, 5, 9, 13] 1不变,所以不用做了
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        #                                                       stride=1且有padding,所以最终大小不变
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, stride=1, padding=pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]  # ::-1 倒叙
        # maxpool时kernel=1不用做,所以要加上[x]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   1x1卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块 SPP前后的模块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters,      filters_list[0], 1),    # 1 降低通道
        conv2d(filters_list[0], filters_list[1], 3),    # 3 提取特征
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块 PANet中使用
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters,      filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters,         filters_list[0], 3),     # 特征整合
        nn.Conv2d(filters_list[0], filters_list[1], 1),     # 将通道转换为预测
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone   = darknet53(pretrained)

        #---------------------------------------------------#
        #   SPP部分,不同分支池化
        #---------------------------------------------------#
        self.conv1      = make_three_conv([512,1024], 1024)     # 13,13,1024 -> 13,13, 512
        self.SPP        = SpatialPyramidPooling()               # 13,13, 512 -> 13,13,2048
        self.conv2      = make_three_conv([512,1024], 2048)     # 13,13,2048 -> 13,13, 512

        #---------------------------------------------------#
        #   上采样部分 2次    p3,p4要先进行1x1卷积降低维度
        #---------------------------------------------------#
        self.upsample1          = Upsample(512, 256)                # 13,13,512 -> 26,26,256    p5的1x1conv+上采样
        self.conv_for_P4        = conv2d(512, 256, 1)               # 26,26,512 -> 26,26,256    拼接之前的1x1conv
        self.make_five_conv1    = make_five_conv([256, 512], 512)   # 26,26,512 -> 26,26,256

        self.upsample2          = Upsample(256, 128)                # 26,26,256 -> 52,52,128 p4的1x1conv+上采样
        self.conv_for_P3        = conv2d(256, 128, 1)               # 52,52,256 -> 52,52,128 拼接之前的1x1conv
        self.make_five_conv2    = make_five_conv([128, 256], 256)   # 52,52,256 -> 52,52,128

        #---------------------------------------------------#
        #   head部分
        #   下采样部分 2次 重新计算P4 P5    输入数据不需要预先进行1x1卷积
        #---------------------------------------------------#
        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20) = 75  3 每个点有3中框, 4代表位置, 1代表有没有物体 20代表分类数
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

        self.down_sample1       = conv2d(128, 256, 3, stride=2)     # 52,52,128 -> 26,26,256
        self.make_five_conv3    = make_five_conv([256, 512], 512)   # 26,26,512 -> 26,26,256

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20) = 75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        self.down_sample2       = conv2d(256, 512, 3, stride=2)     # 26,26,256 -> 13,13,512
        self.make_five_conv4    = make_five_conv([512, 1024], 1024) # 13,13,1024 -> 13,13,512

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20) = 75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

    def forward(self, x):
        #  backbone
        out3, out4, out5 = self.backbone(x)

        #---------------------------------------------------#
        #   SPP部分,不同分支池化
        #---------------------------------------------------#
        P5 = self.conv1(out5)   # 13,13,1024 -> 13,13, 512
        P5 = self.SPP(P5)       # 13,13, 512 -> 13,13,2048
        P5 = self.conv2(P5)     # 13,13,2048 -> 13,13, 512

        #---------------------------------------------------#
        #   上采样部分 2次
        #---------------------------------------------------#
        P5_upsample = self.upsample1(P5)        # 13,13,512 -> 26,26,256
        P4 = self.conv_for_P4(out4)             # 26,26,512 -> 26,26,256    拼接之前的1x1conv
        P4 = torch.cat([P4, P5_upsample], dim=1)# 26,26,256 + 26,26,256 -> 26,26,512
        P4 = self.make_five_conv1(P4)           # 26,26,512 -> 26,26,256

        P4_upsample = self.upsample2(P4)        # 26,26,256 -> 52,52,128
        P3 = self.conv_for_P3(out3)             # 52,52,256 -> 52,52,128    拼接之前的1x1conv
        P3 = torch.cat([P3, P4_upsample], dim=1)# 52,52,128 + 52,52,128 -> 52,52,256
        P3 = self.make_five_conv2(P3)           # 52,52,256 -> 52,52,128

        #---------------------------------------------------#
        #   下采样部分 2次 重新计算P4 P5
        #---------------------------------------------------#
        P3_downsample = self.down_sample1(P3)       # 52,52,128 -> 26,26,256
        P4 = torch.cat([P3_downsample, P4], dim=1)  # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = self.make_five_conv3(P4)               # 26,26,512 -> 26,26,256

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)  # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = self.make_five_conv4(P5)               # 13,13,1024 -> 13,13,512

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

