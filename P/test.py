import torch, time
from thop import profile
import torch.nn.functional as F
import numpy as np
import os, argparse, cv2
from scipy import misc
from lib.Net_apliation import Network
from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/xd508/zyx/P/weight/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
    data_path = '/home/xd508/zyx/COD/COD/TestDataset/{}/'.format(_data_name)
    save_path = './testmaps/{}/'.format(_data_name)
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    total_time = 0
    count = 0

    for i in range(test_loader.size):
        image, gt, name,group_name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start_time = time.perf_counter()
        preds= model(image)
        end_time = time.perf_counter()
        count += 1
        total_time += end_time-start_time
        res = preds[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {} '.format(_data_name, name))
        cv2.imwrite(save_path+name, res*255)
    fps = count/total_time
    print('FPS:', fps)



       
        # save_path_with_group = os.path.join(save_path, group_name + '_' + name)
        # cv2.imwrite(save_path_with_group, res*255)
        #misc.imsave(save_path+name, res)
        # cv2.imwrite(save_path+name, res*255)
