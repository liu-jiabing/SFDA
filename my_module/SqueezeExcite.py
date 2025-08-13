import torch
import torch.nn as nn

class SqueezeExcite(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        # 这里的fc1和fc2将进行通道数的缩减与恢复
        self.fc1 = nn.Linear(input_channels, input_channels // reduction)
        self.fc2 = nn.Linear(input_channels // reduction, input_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入进行全局平均池化
        squeeze = torch.mean(x, dim=(2, 3), keepdim=True)
        squeeze = squeeze.view(squeeze.size(0), -1)  # 将结果拉平成二维
        excitation = self.fc2(self.fc1(squeeze))    # 两层全连接层
        excitation = self.sigmoid(excitation).view(excitation.size(0), excitation.size(1), 1, 1)
        return x * excitation
