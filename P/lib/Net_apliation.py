



"""
整体网络骨干
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, transforms
from torch import nn
from lib.pvtv2 import pvt_v2_b2

from lib.feature import LEG_Module
from lib.Fusion import fusion  #特征融合模块
from lib.Skeleton_Encoder import SkeletonEncoder






import time

savepath = r'features_whitegirl_GT'
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))











class SAM(nn.Module):
    def __init__(self, ch_in=64, reduction=16):
        super(SAM, self).__init__()
        self.conv1_h = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        self.conv2_h = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3_h = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv4_h = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=8, dilation=8)
        self.conv5_h = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=16, dilation=16)

        self.conv11_h = nn.Conv2d(ch_in * 5, ch_in, kernel_size=1)

        self.conv1_l = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        self.conv2_l = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3_l = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv4_l = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=8, dilation=8)
        self.conv5_l = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=16, dilation=16)

        self.conv11_l = nn.Conv2d(ch_in * 5, ch_in, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x_h_ori, x_l_ori):
        # print('x_h shape, x_l shape,',x_h.shape, x_l.shape)
        b, c, _, _ = x_h_ori.size()
        # print('self.avg_pool(x_h)',self.avg_pool(x_h).shape)
        x_h = self.conv11_h(torch.cat(
            [self.conv1_h(x_h_ori), self.conv2_h(x_h_ori), self.conv3_h(x_h_ori), self.conv4_h(x_h_ori),
             self.conv5_h(x_h_ori)], dim=1))
        y_h = self.avg_pool(x_h).view(b, c)  # squeeze操作
        # print('***this is Y-h shape',y_h.shape)
        h_weight = self.fc_wight(y_h)
        # print('h_weight',h_weight.shape,h_weight) ##(batch,1)
        y_h = self.fc(y_h).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        # print('y_h.expand_as(x_h)',y_h.expand_as(x_h).shape)
        x_fusion_h = x_h * y_h.expand_as(x_h)
        x_fusion_h = torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
        ##################----------------------------------
        b, c, _, _ = x_l_ori.size()
        x_l = self.conv11_l(torch.cat(
            [self.conv1_l(x_l_ori), self.conv2_l(x_l_ori), self.conv3_l(x_l_ori), self.conv4_l(x_l_ori),
             self.conv5_l(x_l_ori)], dim=1))
        y_l = self.avg_pool(x_l).view(b, c)  # squeeze操作
        l_weight = self.fc_wight(y_l)
        # print('l_weight',l_weight.shape,l_weight)
        y_l = self.fc(y_l).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        # print('***this is y_l shape', y_l.shape)
        x_fusion_l = x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
        #################-------------------------------
        # print('x_fusion_h shape, x_fusion_l shape,h_weight shape',x_fusion_h.shape,x_fusion_l.shape,h_weight.shape)
        x_fusion = x_fusion_h + x_fusion_l

        return x_fusion  # 注意力作用每一个通道上



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


class Network(nn.Module):
    def __init__(self, channels=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path ='/home/xd508/zyx/PVT/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.skeleton = SkeletonEncoder()

        self.leg_module_3 = LEG_Module(channel=64,
                                       stage=3,
                                       drop_path=0.1,
                                       act_layer=nn.GELU,
                                       norm_layer=dict(type='BN', requires_grad=True))
        self.leg_module_2 = LEG_Module(channel=64,
                                       stage=2,
                                       drop_path=0.1,
                                       act_layer=nn.GELU,
                                       norm_layer=dict(type='BN', requires_grad=True))
        self.leg_module_1 = LEG_Module(channel=64,
                                       stage=1,
                                       drop_path=0.1,
                                       act_layer=nn.GELU,
                                       norm_layer=dict(type='BN', requires_grad=True))
        self.leg_module_0 = LEG_Module(channel=64,
                                       stage=0,
                                       drop_path=0.1,
                                       act_layer=nn.GELU,
                                       norm_layer=dict(type='BN', requires_grad=True))

        self.fusion_3 = fusion()
        self.fusion_2 = fusion()
        self.fusion_1 = fusion()
        self.fusion_0 = fusion()

        self.SAM = SAM()

        self.conv64_1 = nn.Conv2d(64, 1, 1)
        self.conv128_64 = nn.Conv2d(128, 64, 1)
        self.conv64_128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv32_64 = nn.Conv2d(32, 64, 3, 1, 1)


        self.reduce4 = BasicConv2d(512, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(320, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(128, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(64, 64, kernel_size=1)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)

        xg, pg = self.skeleton(x)  #对pg进行损失监督  xg [1, 32, 176, 176]   pg[1, 1, 176, 176]
        xg = self.conv32_64(xg)


        x1 = pvt[0]# x1[1, 64, 88, 88]
        x2 = pvt[1]# x2[1, 128, 44, 44]
        x3 = pvt[2]# x3[1, 320, 22, 22]
        x4 = pvt[3]# x4[1, 512, 11, 11]

        x1 = self.reduce1(x1)      #r1[1, 64, 96, 96]
        x2 = self.reduce2(x2)      #r2[1, 64, 48, 48]
        x3 = self.reduce3(x3)      #r3[1, 64, 24, 24]
        x4 = self.reduce4(x4)      #r4[1, 64, 12, 12]

        #模块处理阶段

        x4_t = self.leg_module_3(x4, x4)  #[1, 128, 14. 14]
        x3_t = self.leg_module_2(x3, x4)   #[1, 128, 14, 14]
        x2_t = self.leg_module_1(x2, x3)   #[1, 128, 14, 14]
        cim_feature = self.leg_module_0(x1, x2)   #[1, 128, 14, 14]

        x4_t = self.conv128_64(x4_t)
        x3_t = self.conv128_64(x3_t)
        x2_t = self.conv128_64(x2_t)
        cim_feature = self.conv128_64(cim_feature)



        """
        对特征进行融合，得到最终的预测结果
        """
        gt4 = self.fusion_3(x4_t, xg)

        if gt4.size()[2:] !=x3_t.size()[2:]:
            gt4 = F.interpolate(gt4, size=x3_t.size()[2:], mode='bilinear', align_corners=True)
        x3_t = gt4 + x3_t
        gt3 = self.fusion_2(x3_t, xg)

        if gt3.size()[2:] !=x2_t.size()[2:]:
            gt3 = F.interpolate(gt3, size=x2_t.size()[2:], mode='bilinear', align_corners=True)
        x2_t = gt3 + x2_t
        gt2 = self.fusion_1(x2_t, xg)

        if gt2.size()[2:] !=cim_feature.size()[2:]:
            gt2 = F.interpolate(gt2, size=cim_feature.size()[2:], mode='bilinear', align_corners=True)
        cim_feature = gt2 + cim_feature

        gt1 = self.fusion_0(cim_feature, xg)

        gt2 = F.interpolate(gt2, size=gt1.size()[2:], mode='bilinear', align_corners=True)
        sam_feature = self.SAM(gt1, gt2)  #[1, 64, 192, 192]

        """
        gt1[1, 64, 192, 192]
        gt2[1, 64, 192, 192]
        gt3[1, 64, 24, 24]
        gt4[1, 64, 12, 12]  
        """


        x4 = self.conv64_1(gt4)
        x3 = self.conv64_1(gt3)
        x2 = self.conv64_1(gt2)
        x1 = self.conv64_1(gt1)
        x0 = self.conv64_1(sam_feature)


        gt4 = F.interpolate(x4, size=image_shape, mode='bilinear', align_corners=True)
        gt3 = F.interpolate(x3, size=image_shape, mode='bilinear', align_corners=True)
        gt2 = F.interpolate(x2, size=image_shape, mode='bilinear', align_corners=True)
        gt1 = F.interpolate(x1, size=image_shape, mode='bilinear', align_corners=True)
        gt0 = F.interpolate(x0, size=image_shape, mode='bilinear', align_corners=True)
        pg = F.interpolate(pg, size=image_shape, mode='bilinear', align_corners=True)

        return gt0, gt1, gt2, gt3, gt4, pg


# if __name__ =='__main__':
#     from thop import profile
#     net = Network().cuda()
#     data = torch.randn(1, 3, 384, 384).cuda()
#     flops, params = profile(net, (data,))
#     print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))
#
#


model = Network().cuda()
# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model.eval()
img = cv2.imread('2.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = transform(img).cuda()
img = img.unsqueeze(0)

with torch.no_grad():
    start = time.time()
    out = model(img)
    print("total time:{}".format(time.time() - start))
    result = out.cpu().numpy()
    # ind=np.argmax(out.cpu().numpy())
    ind = np.argsort(result, axis=1)
    for i in range(5):
        print("predict:top {} = cls {} : score {}".format(i + 1, ind[0, 1000 - i - 1], result[0, 1000 - i - 1]))
    print("done")
