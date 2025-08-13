import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expansion=6):
        """
        input_channels: 输入的通道数
        output_channels: 输出的通道数
        stride: 步幅
        expansion: 通道扩展的因子（默认为6，表示输入通道数将扩展为6倍）
        """
        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        # 扩展层（1x1卷积，扩展通道）
        self.expand = nn.Conv2d(input_channels, input_channels * expansion, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(input_channels * expansion)

        # 深度卷积层（3x3卷积，进行特征提取）
        self.depthwise = nn.Conv2d(input_channels * expansion, input_channels * expansion, 
                                    kernel_size=3, stride=stride, padding=1, groups=input_channels * expansion, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(input_channels * expansion)

        # 压缩层（1x1卷积，压缩通道数）
        self.project = nn.Conv2d(input_channels * expansion, output_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        # 扩展层：1x1卷积
        x = self.expand_bn(F.relu(self.expand(x)))

        # 深度卷积层：3x3卷积
        x = self.depthwise_bn(F.relu(self.depthwise(x)))

        # 压缩层：1x1卷积
        x = self.project_bn(self.project(x))

        # 残差连接（如果满足条件）
        if self.use_res_connect:
            return x + self.residual(x)
        else:
            return x

    def residual(self, x):
        # 残差连接的通道数保持不变，只需调整尺寸
        return x
