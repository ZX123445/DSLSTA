import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class fusion(nn.Module):
    def __init__(self, dim=64):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个自适应平均池化层，用于全局特征提取
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个顺序模型，用于特征的注意力加权
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),  # 3D卷积层，用于特征的线性变换
            nn.Sigmoid()  # Sigmoid激活函数，用于生成注意力权重
        )

        # 3D卷积层，用于特征的降维
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # 3D卷积层，用于生成空间注意力
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)

        # 另一个3D卷积层，用于生成空间注意力
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)

        # Sigmoid激活函数，用于生成最终的空间注意力权重
        self.nonlin = nn.Sigmoid()

        # ********************************************************
        self.sigmoid = nn.Sigmoid()
        self.conv1_128 = nn.Conv2d(1, 128, 3, 1, 1)

        self.reduce0 = BasicConv2d(64, 64, kernel_size=1)
        self.conv = nn.Conv2d(128, 64, 1)

    def forward(self, x, skip):
        '''
        in: self, x, skip
            x: 输入特征图1
            skip: 输入特征图2（跳跃连接）
        out: output
        '''

        # ****************************************************************************
        """
        修改部分
        对输入x进行Sigmoid,减法操作，乘法操作

        """
        x_1 = self.reduce0(x)
        skip_1 = self.reduce0(skip)

        if skip_1.size()[2:] != x_1.size()[2:]:
            x_1 = F.interpolate(x_1, size=skip_1.size()[-2:], mode='bilinear', align_corners=True)

        if skip.size()[2:] != x.size()[2:]:
            x = F.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip_1], dim=1)
        x = self.conv(x)

        skip = torch.cat([skip, x_1], dim=1)
        skip = self.conv(skip)


        x = x - self.sigmoid(x) * skip  # x[1, 64, 48, 48]
        skip = skip * self.sigmoid(x)  # skip[1, 64, 48, 48]
        # ***************************************************************************


        # 沿通道维度拼接两个特征图
        output = torch.cat([x, skip], dim=1)

        # 使用平均池化和注意力卷积生成全局通道注意力
        att = self.conv_atten(self.avg_pool(output))

        # 将全局通道注意力应用于拼接后的特征图
        output = output * att

        # 使用降维卷积减少特征图的通道数
        output = self.conv_redu(output)

        # 使用两个卷积层分别生成空间注意力
        att = self.conv1(x) + self.conv2(skip)

        # 应用非线性激活函数生成最终的空间注意力权重
        att = self.nonlin(att)

        # 将空间注意力应用于降维后的特征图
        output = output * att

        # 返回最终的输出特征图
        return output


