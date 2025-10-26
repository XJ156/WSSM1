# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict


import glob
from time import time

import copy
import math
import argparse
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

import torch
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
# import ttach as tta

from dataset.dataset import Dataset

from net import Unet
# from net import Unet,sepnet,MultiResUnet,dense_UNet,segnet,UNetplusplus,cbam_Unet
from utilities.utils import str2bool, count_params
from net.unet_model import UNet
from net.unet_DCN import DCNUNet
from net.MyModel import MyModelUnet
import pandas as pd

# import ttach as tta
palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14],[15], [16]]
test_ct_path = '/root/imagesTs'  # 需要预测的CT图像
seg_result_path = '/root/labelsTs'  # mask

pred_path = '/root/autodl-tmp/Dataset/WORD/pred_result_Mymodel_0.8467'

if not os.path.exists(pred_path):
    os.mkdir(pred_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--training', type=bool, default=False,
                        help='whthere dropout or not')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if (val != 1.0):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # print("进入hook_fn")
        self.features = output

    def close(self):
        self.hook.remove()


def getheatmap(feature):
    cam = feature.features.squeeze(0).detach().cpu().numpy()
    cam = np.mean(cam, axis=0)
    # cam = np.sum(cam,axis = 0)
    # print(cam.shape)
    # print(type(cam), cam.shape)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = cam * 255
    cam = cam.astype(np.uint8)
    cam = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return heatmap

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = colour_codes[x]
    return x.astype(np.uint8)

def main():
    log = pd.DataFrame(index=[], columns=[
        'file_index', 'file', 'iou','liver', 'spleen', 'left_kidney', 'right_kidney', 'stomach', 'gallbladder', 'esophagus', 'pancreas', 'duodenum', 'colon',
                         'intestine', 'adrenal', 'rectum', 'bladder', 'Head_of_femur_L', 'Head_of_femur_R','testmean'
    ])

    # val_args = parse_args()
    #
    # args = joblib.load('/disk/sdc/liumenghao/workspace/xyregistration/Word/models/WORD_Unet_lym/2023-12-07-18-03-15/args.pkl')
    # print('Config -----')
    # for arg in vars(args):
    #     print('%s: %s' % (arg, getattr(args, arg)))
    # print('------------')
    # # joblib.dump(args,'/disk/sdb/liumenghao/pycharm_workspace/registration/Main1/models/LiTS_Unet_lym/2023-07-17-11-54-49/args.pkl')

    # create model
    print("=> creating model ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = UNet(n_channels=3, n_classes=17)  # todo edit input_channels n_classes
    # model = DCNUNet(n_channels=3, n_classes=17)  # todo edit input_channels n_classes
    model = MyModelUnet(n_channels=3, n_classes=17)  # todo edit input_channels n_classes
    # 将网络拷贝到deivce中
    model.to(device=device)

    model.load_state_dict(torch.load('/root/autodlworkspace/WORD/models/WORD_Unet_lym/2023-12-14-20-15-55/epoch27-0.8467_model.pth'))
    model.eval()
    in_feature = SaveFeatures(model.inc.double_conv)
    down1_feature = SaveFeatures(model.down1.maxpool_conv)
    down2_feature = SaveFeatures(model.down2.maxpool_conv)
    down3_feature = SaveFeatures(model.down3.maxpool_conv)
    up1_feature = SaveFeatures(model.up1.conv)
    up2_feature = SaveFeatures(model.up2.conv)
    up3_feature = SaveFeatures(model.up3.conv)
    up4_feature = SaveFeatures(model.up4.conv)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    for file_index, file in enumerate(os.listdir(test_ct_path)):

        # 生成特征图所需要的路径
        savepath = "/root/Mymodel_results_JET/" + file.replace(".nii.gz", "")
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        savepathin = os.path.join(savepath, "in")
        savepathdown1 = os.path.join(savepath, "down1")
        savepathdown2 = os.path.join(savepath, "down2")
        savepathdown3 = os.path.join(savepath, "down3")
        savepathup1 = os.path.join(savepath, "up1")
        savepathup2 = os.path.join(savepath, "up2")
        savepathup3 = os.path.join(savepath, "up3")
        savepathup4 = os.path.join(savepath, "up4")
        allpath = [savepathin, savepathdown1, savepathdown2, savepathdown3, savepathup1, savepathup2, savepathup3,savepathup4]
        for path in allpath:
            if not os.path.exists(path):
                os.mkdir(path)

        losses = AverageMeter()
        ious = AverageMeter()
        dices_1s = AverageMeter()
        dices_2s = AverageMeter()
        dices_3s = AverageMeter()
        dices_4s = AverageMeter()
        dices_5s = AverageMeter()
        dices_6s = AverageMeter()
        dices_7s = AverageMeter()
        dices_8s = AverageMeter()
        dices_9s = AverageMeter()
        dices_10s = AverageMeter()
        dices_11s = AverageMeter()
        dices_12s = AverageMeter()
        dices_13s = AverageMeter()
        dices_14s = AverageMeter()
        dices_15s = AverageMeter()
        dices_16s = AverageMeter()

        start = time()


        # 将CT读入内存
        ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        mask = sitk.ReadImage(os.path.join(seg_result_path, file), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        print('start predict file:', file)

        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        start_slice = max(0, start_slice - 1)
        end_slice = min(mask_array.shape[0] - 1, end_slice + 2)

        ct_crop = ct_array[start_slice:end_slice, 32:480, 32:480]
        mask_crop = mask_array[start_slice + 1:end_slice - 1, 32:480, 32:480]

        # print("ct_crop,mask_crop.shape",ct_crop.shape,mask_crop.shape)

        slice_predictions = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)

        with torch.no_grad():
            for n_slice in range(mask_crop.shape[0]):
                npmask = mask_crop[n_slice, :, :]
                npmask = npmask.reshape(npmask.shape[0], npmask.shape[1], 1)
                npmask = mask_to_onehot(npmask, palette)
                npmask = npmask.transpose((2, 0, 1))
                # npmask = npmask.astype("float32")
                mask_tensor = torch.FloatTensor(npmask).cuda()
                target = mask_tensor.unsqueeze(dim=0)


                ct_tensor = torch.FloatTensor(ct_crop[n_slice: n_slice + 3, :, :]).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print('ct_tensor',ct_tensor.shape,n_slice)
                output = model(ct_tensor)

                # # 输出特征图
                # inheatmap = getheatmap(in_feature)
                # down1heatmap = getheatmap(down1_feature)
                # down2heatmap = getheatmap(down2_feature)
                # down3heatmap = getheatmap(down3_feature)
                # up1heatmap = getheatmap(up1_feature)
                # up2heatmap = getheatmap(up2_feature)
                # up3heatmap = getheatmap(up3_feature)
                # up4heatmap = getheatmap(up4_feature)
                # allheatmap = [inheatmap, down1heatmap, down2heatmap, down3heatmap, up1heatmap, up2heatmap, up3heatmap,
                #               up4heatmap]
                # for i in range(len(allheatmap)):
                #     cv2.imwrite(allpath[i] + "/" + str(n_slice) + ".png", allheatmap[i])

                # 求dice
                iou = iou_score(output, target)
                all_dice = dice_coef(output, target)
                print(all_dice)
                ious.update(iou, ct_tensor.shape[0])
                dices_1s.update(all_dice[0], ct_tensor.shape[0])
                dices_2s.update(all_dice[1], ct_tensor.shape[0])
                dices_3s.update(all_dice[2], ct_tensor.shape[0])
                dices_4s.update(all_dice[3], ct_tensor.shape[0])
                dices_5s.update(all_dice[4], ct_tensor.shape[0])
                dices_6s.update(all_dice[5], ct_tensor.shape[0])
                dices_7s.update(all_dice[6], ct_tensor.shape[0])
                dices_8s.update(all_dice[7], ct_tensor.shape[0])
                dices_9s.update(all_dice[8], ct_tensor.shape[0])
                dices_10s.update(all_dice[9], ct_tensor.shape[0])
                dices_11s.update(all_dice[10], ct_tensor.shape[0])
                dices_12s.update(all_dice[11], ct_tensor.shape[0])
                dices_13s.update(all_dice[12], ct_tensor.shape[0])
                dices_14s.update(all_dice[13], ct_tensor.shape[0])
                dices_15s.update(all_dice[14], ct_tensor.shape[0])
                dices_16s.update(all_dice[15], ct_tensor.shape[0])

                output = torch.sigmoid(output).data.cpu().numpy()
                probability_map = onehot_to_mask(np.array(output.squeeze()).transpose([1, 2, 0]), palette)
                probability_map = probability_map.transpose([2, 0, 1])
                # for idx in range(output.shape[2]):
                #     for idy in range(output.shape[3]):
                #         if (output[0, 12, idx, idy] > 0.35):
                #             probability_map[0, idx, idy] = 12

                # # 预测值拼接回去
                # # i = 0
                # for idz in range(output.shape[1]):
                #     for idx in range(output.shape[2]):
                #         for idy in range(output.shape[3]):
                #             if (output[0, 0, idx, idy] > 0.65):
                #                 probability_map[0, idx, idy] = 1
                #             if (output[0, 1, idx, idy] > 0.5):
                #                 probability_map[0, idx, idy] = 2

                slice_predictions[n_slice + start_slice + 1, 32:480, 32:480] = probability_map
            all_avg = (dices_1s.avg + dices_2s.avg + dices_3s.avg + dices_4s.avg + dices_5s.avg + dices_6s.avg + dices_7s.avg + dices_8s.avg + dices_9s.avg + dices_10s.avg + dices_11s.avg + dices_12s.avg + dices_13s.avg + dices_14s.avg + dices_15s.avg + dices_16s.avg) / 16
            print("测试集的平均值-----------------", dices_1s.avg, dices_2s.avg, dices_3s.avg, dices_4s.avg, dices_5s.avg,
                  dices_6s.avg, dices_7s.avg,
                  dices_8s.avg, dices_9s.avg, dices_10s.avg, dices_11s.avg, dices_12s.avg, dices_13s.avg, dices_14s.avg,
                  dices_15s.avg, dices_16s.avg)
            print("测试集所有器官的平均值为：--------------", all_avg)
            testlog = OrderedDict([
                ('iou', ious.avg),
                ('dice_1', dices_1s.avg), ('dice_2', dices_2s.avg), ('dice_3', dices_3s.avg), ('dice_4', dices_4s.avg),
                ('dice_5', dices_5s.avg), ('dice_6', dices_6s.avg),('dice_7', dices_7s.avg), ('dice_8', dices_8s.avg), ('dice_9', dices_9s.avg),
                ('dice_10', dices_10s.avg),('dice_11', dices_11s.avg), ('dice_12', dices_12s.avg),
                ('dice_13', dices_13s.avg), ('dice_14', dices_14s.avg), ('dice_15', dices_15s.avg),('dice_16', dices_16s.avg),
                ('avg_dice', all_avg)
            ])

            tmp = pd.Series([
                file_index,
                file,
                testlog['iou'],
                testlog['dice_1'], testlog['dice_2'], testlog['dice_3'], testlog['dice_4'], testlog['dice_5'],
                testlog['dice_6'], testlog['dice_7'], testlog['dice_8'],
                testlog['dice_9'], testlog['dice_10'], testlog['dice_11'], testlog['dice_12'],
                testlog['dice_13'], testlog['dice_14'], testlog['dice_15'], testlog['dice_16'],
                testlog['avg_dice'],
            ], index=['file_index', 'file', 'iou','liver', 'spleen', 'left_kidney', 'right_kidney', 'stomach', 'gallbladder', 'esophagus', 'pancreas', 'duodenum', 'colon',
                         'intestine', 'adrenal', 'rectum', 'bladder', 'Head_of_femur_L', 'Head_of_femur_R','testmean'])

            log = log._append(tmp, ignore_index=True)
            log.to_csv(os.path.join(pred_path, 'testlog.csv'), index=False)

            pred_seg = slice_predictions
            pred_seg = pred_seg.astype(np.uint8)

            pred_seg = sitk.GetImageFromArray(pred_seg)

            pred_seg.SetDirection(ct.GetDirection())
            pred_seg.SetOrigin(ct.GetOrigin())
            pred_seg.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('word', 'pred-word')))

            speed = time() - start

            print(file, 'this case use {:.3f} s'.format(speed))
            print('-----------------------')

            torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()



