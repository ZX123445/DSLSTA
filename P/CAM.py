import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm

from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet152_Weights

from torchvision.transforms.functional import to_pil_image

img_path = 'C:/Users/Lenovo/Desktop/xunlian_2 _PVT/00383.jpg'     # 输入图片的路径
save_path = 'C:/Users/Lenovo/Desktop/xunlian_2 _PVT/00383.png'    # 类激活图保存路径

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



#************模型更替*****************************

# 使用 ResNet50_Weights.DEFAULT 加载预训练权重
weights = ResNet152_Weights.DEFAULT
net = models.resnet152(weights=weights).cuda()   # 导入模型

# 打印模型结构
# print(net)

# 建立列表容器，用于盛放输出特征图
feature_map = []

# 构造前向传播钩子
def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入列表 feature_map

# 对 net.layer4 这一层注册前向传播钩子
net.layer4.register_forward_hook(forward_hook)

# 建立列表容器，用于盛放特征图的梯度
grad = []

# 构造后向传播钩子
def backward_hook(module, inp, outp):    # 定义hook
    grad.append(outp)    # 把输出装入列表 grad

# 注册反向传播钩子
net.layer4.register_full_backward_hook(backward_hook)

# 打开图片并转换为RGB模式
orign_img = Image.open(img_path).convert('RGB')

# 图片预处理
img = preprocess(orign_img)
img = torch.unsqueeze(img, 0)     # 增加batch维度 [1, 3, 224, 224]

# 前向传播
out = net(img.cuda())

# 获取预测类别编码
cls_idx = torch.argmax(out).item()

# 获取预测类别分数
score = out[:, cls_idx].sum()

# 梯度清零
net.zero_grad()

# 由预测类别分数反向传播
score.backward(retain_graph=True)

# 获得权重
weights = grad[0][0].squeeze(0).mean(dim=(1, 2))

# 计算 Grad-CAM
grad_cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)

# CAM 归一化
def _normalize(cams: Tensor) -> Tensor:
    """CAM normalization"""
    cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
    return cams

grad_cam = _normalize(F.relu(grad_cam, inplace=True)).cpu()
mask = to_pil_image(grad_cam.detach().numpy(), mode='F')

# 叠加掩码和原图
def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
    return overlayed_img

# 生成结果并保存
result = overlay_mask(orign_img, mask)
result.show()
result.save(save_path)