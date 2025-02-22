import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import utils.helpers as helpers
import numpy as np
import matplotlib.pyplot as plt

# 图像标签的颜色和名称定义
'''
label:
- color: '#000000'
  name: __background__
- color: '#aa0000'
  name: L1
- color: '#005500'
  name: R1
- color: '#aa007f'
  name: L2
- color: '#00557f'
  name: R2
- color: '#aa00ff'
  name: L3
- color: '#0055ff'
  name: R3
- color: '#55ff00'
  name: L
- color: '#ffff7f'
  name: R
- color: '#aaff00'
  name: S
'''
palette = [[0, 0, 0], [170, 0, 0], [0, 85, 0], [170, 0, 127], 
           [0, 85, 127], [170, 0, 255], [0, 85, 255], [170, 255, 0]]  # one-hot的颜色表
num_classes = 8  # 分类数,不包含L和R

# 用于根据给定的数据集路径创建数据项
def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']  # 检查模式是否为 'train'、'val' 或 'test'
    items = []
    if mode == 'train':  # 如果是训练集
        img_path = os.path.join(root, 'Images')  # 图像路径
        mask_path = os.path.join(root, 'Labels')  # 标签路径

        if 'Augdata' in root:  # 当使用增广后的训练集
            data_list = os.listdir(os.path.join(root, 'Labels'))  # 获取标签文件列表
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]  # 从文件读取训练数据列表
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))  # 将图像和标签的路径作为一项添加
            items.append(item)
    elif mode == 'val':  # 如果是验证集
        img_path = os.path.join(root, 'Images')  # 图像路径
        mask_path = os.path.join(root, 'Labels')  # 标签路径
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]  # 从文件读取验证数据列表
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))  # 将图像和标签的路径作为一项添加
            items.append(item)
    else:  # 如果是测试集
        img_path = os.path.join(root, 'Images')  # 图像路径
        try:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]  # 从文件读取测试数据列表
        except:
            raise FileNotFoundError(f"文件test.txt不存在!")  # 如果找不到文件则报错
        for it in data_list:
            item = (os.path.join(img_path,it))  # 只有图像路径
            items.append(item)
    return items

# 引入torchvision的transforms库进行图像处理
from torchvision import transforms

# 定义数据集类
class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)  # 加载数据集
        self.palette = palette  # 定义颜色表
        self.mode = mode  # 数据集模式（训练、验证、测试）
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')  # 如果没有图片数据，则抛出异常
        self.joint_transform = joint_transform  # 图像联合变换
        self.center_crop = center_crop  # 中心裁剪
        self.transform = transform  # 输入图像的转换
        self.target_transform = target_transform  # 目标图像的转换

    def __getitem__(self, index):
        # 获取图像和标签路径
        img_path, mask_path = self.imgs[index]
        file_name = mask_path.split('\\')[-1]  # 获取文件名

        img = Image.open(img_path)  # 打开图像
        mask = Image.open(mask_path)  # 打开标签
        img = img.convert('L')  # 转换为灰度图像
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)  # 如果有联合变换，进行处理
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)  # 如果有中心裁剪，进行处理
        img = np.array(img)  # 转换为NumPy数组
        mask = np.array(mask)  # 转换为NumPy数组
        mask = helpers.mask_to_onehot(mask, self.palette)  # 将标签转为one-hot编码
        # print(img.shape, mask.shape) (512, 512) (512, 512, 8)
        # shape from (H, W, C) to (C, H, W)
        # img = img.transpose([2, 0, 1])  # 如果是彩色图像，调整维度
        # mask = mask.transpose([2, 0, 1])  # 如果是彩色标签，调整维度
        # print(img.shape, mask.shape)
        if self.transform is not None:
            img = self.transform(img)  # 进行输入图像的转换
        if self.target_transform is not None:
            mask = self.target_transform(mask)  # 进行目标图像的转换
        # print(img.shape, mask.shape)    #torch.Size([1, 512, 512]) torch.Size([8, 512, 512])
        return (img, mask), file_name  # 返回图像、标签和文件名

    def __len__(self):
        return len(self.imgs)  # 返回数据集的长度

# 导入DataLoader来加载数据
from torch.utils.data import DataLoader

if __name__ == '__main__':
  np.set_printoptions(threshold=9999999)  # 设置NumPy数组打印选项

  # 测试加载数据类
  def demo():
      train_path = r'.\Dataset'  # 训练集路径
      val_path = r'.\Dataset'  # 验证集路径
      test_path = r'.\Dataset'  # 测试集路径

      # center_crop = joint_transforms.CenterCrop(256)
      center_crop = None  # 没有使用中心裁剪
      test_center_crop = transforms.CenterCrop(256)  # 使用torchvision的CenterCrop进行裁剪
      train_input_transform = transforms.Compose([transforms.ToTensor()])  # 输入图像转换为Tensor
      target_transform = transforms.Compose([transforms.ToTensor()])  # 标签转换为Tensor

      # 创建数据集实例
      train_set = Dataset(train_path, 'train', 1,
                            joint_transform=None, center_crop=center_crop,
                            transform=train_input_transform, target_transform=target_transform)
      train_loader = DataLoader(train_set, batch_size=1, shuffle=False)  # 使用DataLoader加载数据

      for (input, mask), file_name in train_loader:
          # 打印图像和标签的形状
          # print(input.shape)
          # print(mask.shape)
#         torch.Size([1, 1, 512, 512])
# torch.Size([1, 8, 512, 512])
          img = np.array(input.squeeze())  # 将输入图像转换为NumPy数组
          # print(img.shape)
          plt.imshow(img)  # 显示图像
          plt.show()  # 展示图像
          img = helpers.array_to_img(np.expand_dims(input.squeeze(), 2))  # 将Tensor转为PIL图像并展示
          plt.imshow(img)
          plt.show()
          # 将gt反one-hot回去以便进行可视化
          # palette = [[0, 0, 0], [246, 16, 16], [16, 136, 246]]
          palette = [[0, 0, 0], [170, 0, 0], [0, 85, 0], [170, 0, 127], 
          [0, 85, 127], [170, 0, 255], [0, 85, 255], [170, 255, 0]]
          gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose(1, 2, 0), palette)  # 将one-hot标签转为图像
          plt.imshow(gt)  # 展示标签图像
          plt.show()  # 展示标签图像
      
  demo()
