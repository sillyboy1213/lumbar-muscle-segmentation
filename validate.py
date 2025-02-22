import os
import cv2
import torch
import shutil
# import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
# import utils.transforms as extended_transforms
# import Bones
from utils.loss import *
from networks.u_net import Baseline
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import Dataset_gen
from torchvision import transforms

from utils.metrics import diceCoeffv2
import segmentation_models_pytorch as smp
from utils.loss import *
from utils import misc
crop_size = 256
val_path = r'./Dataset'
# center_crop = joint_transforms.CenterCrop(crop_size)
center_crop  = None
val_input_transform = transforms.Compose([
    transforms.ToTensor(),])
target_transform = transforms.Compose([
    transforms.ToTensor(),])


val_set = Dataset_gen.Dataset(val_path, 'val', 1,
                              joint_transform=None, transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette = [[0, 0, 0], [170, 0, 0], [0, 85, 0], [170, 0, 127], 
           [0, 85, 127], [170, 0, 255], [0, 85, 255], [170, 255, 0]]  # one-hot的颜色表
num_classes = 8  # 分类数,不包含L和R

net = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=num_classes,                      # model output channels (number of classes in your dataset)
    )
net.to('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load("./checkpoint/unet_depth=2_fold_1_dice_39817.pth"))
net.eval()


def auto_val(net):
    # 效果展示图片数
    dices = 0
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)

    save_path = './results'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    gt_path = os.path.join(save_path, 'gt')
    os.makedirs(img_path)
    os.makedirs(pred_path)
    os.makedirs(gt_path)

    val_dice_arr = []
    for (input, mask), file_name in tqdm(val_loader):
        file_name = file_name[0].split('.')[0]

        X = input.cuda()
        pred = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()

        # pred[pred < 0.5] = 0
        # pred[np.logical_and(pred > 0.5, pred == 0.5)] = 1

        # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(np.expand_dims(m1, 2))

        # gt
        print(np.array(mask).shape)
        gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = helpers.array_to_img(gt)

        # pred
        save_pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = helpers.array_to_img(save_pred)
        # plt.subplot(1, 2, 1)
        # plt.imshow(gt)
        # plt.subplot(1, 2, 2)
        # plt.imshow(save_pred_png)
        # plt.show()
        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))  
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))

        class_dice = []
        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

        mean_dice = sum(class_dice) / len(class_dice)
        val_dice_arr.append(class_dice)
        dices += mean_dice
        class_dices += np.array(class_dice)

        # 按照上面类似的输出样式更新
        print('Val mean: {:.4} - L1: {:.4} - R1: {:.4} - L2: {:.4} - R2: {:.4} - L3: {:.4} - R3: {:.4} - S: {:.4}'
            .format(mean_dice, class_dice[0], class_dice[1], class_dice[2], class_dice[3],
                    class_dice[4], class_dice[5], class_dice[6]))

        val_mean_dice = dices / (len(val_loader) / 1)
        val_class_dice = class_dices / (len(val_loader) / 1)

        # 按照上面类似的输出样式更新
        print('Val mean: {:.4} - L1: {:.4} - R1: {:.4} - L2: {:.4} - R2: {:.4} - L3: {:.4} - R3: {:.4} - S: {:.4}'
            .format(val_mean_dice, val_class_dice[0], val_class_dice[1], val_class_dice[2], val_class_dice[3],
                    val_class_dice[4], val_class_dice[5], val_class_dice[6]))


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)
    auto_val(net)


      # print(output.detach().cpu().numpy().shape)
            # gt = helpers.onehot_to_mask(output[0].detach().cpu().numpy().transpose([1, 2, 0]), Bones.palette)
            # gt = helpers.array_to_img(gt)
            # plt.imshow(gt)
            # plt.show(block=False)
            # plt.pause(0.2)
            # plt.close()