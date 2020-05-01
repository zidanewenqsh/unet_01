'''
@Descripttion: 
@version: 
@Author: QsWen
@Date: 2020-04-29 18:33:58
@LastEditors: QsWen
@LastEditTime: 2020-04-29 18:33:59
'''
'''
@Descripttion: 
@version: 
@Author: QsWen
@Date: 2020-04-27 22:45:03
@LastEditors: QsWen
@LastEditTime: 2020-04-27 22:45:22
'''

import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import time
import numpy as np
from torchvision import transforms

from torch import optim
from torch.nn import functional
from torch.utils import data


# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, padding=0, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Unet, self).__init__()
        basechannel = 32
        self.conv1 = DoubleConv(in_ch, basechannel)
        self.pool1 = DownSample(basechannel, basechannel)
        self.conv2 = DoubleConv(basechannel, basechannel * 2)
        self.pool2 = DownSample(basechannel * 2, basechannel * 2)
        self.conv3 = DoubleConv(basechannel * 2, basechannel * 4)
        self.pool3 = DownSample(basechannel * 4, basechannel * 4)
        self.conv4 = DoubleConv(basechannel * 4, basechannel * 8)
        self.pool4 = DownSample(basechannel * 8, basechannel * 8)
        self.conv5 = DoubleConv(basechannel * 8, basechannel * 16)
        # 逆卷积，也可以使用上采样
        self.up6 = UpSample()  # nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(basechannel * (16 + 8), basechannel * 8)
        self.up7 = UpSample()
        self.conv7 = DoubleConv(basechannel * (8 + 4), basechannel * 4)
        self.up8 = UpSample()
        self.conv8 = DoubleConv(basechannel * (4 + 2), basechannel * 2)
        self.up9 = UpSample()
        self.conv9 = DoubleConv(basechannel * (2 + 1), basechannel)
        self.conv10 = nn.Conv2d(basechannel, out_ch, 1)

    def mysigmoid(self, x):
        '''
        压缩数据输出
        :param x:
        :return:
        '''
        ratio = 10
        return 1 / (1 + torch.exp(-ratio * (x - 0.0)))

    def forward(self, x):
        c1 = self.conv1(x)
        # print(c1.shape)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        # print(c2.shape)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        # print(c3.shape)
        p3 = self.pool3(c3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        # print(c4.shape)
        p4 = self.pool4(c4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        # print(c5.shape)
        up_6 = self.up6(c5)
        # print(up_6.shape)
        # print(c4.shape)
        merge6 = torch.cat([up_6, c4], dim=1)
        # print(merge6.shape)
        c6 = self.conv6(merge6)
        # print(c6.shape)
        up_7 = self.up7(c6)
        # print(up_7.shape,c3.shape)
        merge7 = torch.cat([up_7, c3], dim=1)
        # print(merge7.shape)
        c7 = self.conv7(merge7)
        # print(c7.shape)
        up_8 = self.up8(c7)
        # print(up_8.shape, c2.shape)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        # print(c8.shape)
        up_9 = self.up9(c8)
        # print(up_9.shape)
        merge9 = torch.cat([up_9, c1], dim=1)
        # print(merge9.shape)
        c9 = self.conv9(merge9)
        # print(c9.shape)
        c10 = self.conv10(c9)
        # print(c10.shape)
        # out = nn.Sigmoid()(c10)
        # print("c10",c10)
        out = self.mysigmoid(c10)
        return out


class UnetDataset(data.Dataset):
    def __init__(self, pic_dir='../datas/data1/data', mask_dir='../datas/data1/label'):  # transform要有，因为图片需要transform
        super().__init__()
        # self.label_path = label_path
        self.pic_dir = pic_dir
        self.mask_dir = mask_dir
        self.dataset = os.listdir(pic_dir)
        self.dataset.sort(key=lambda x: int(x.split('.')[0]))

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with Image.open(os.path.join(self.pic_dir, self.dataset[index])) as image:
            image_data = self.transformer(image)
        with Image.open(os.path.join(self.mask_dir, self.dataset[index])) as label:
            label_data = self.transformer(label)
        return image_data, label_data


class Trainer:
    def __init__(self, net_savefile=r"./net.pth", param_savefile=r"./net.pt"):
        self.module = Unet(1, 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_savefile = net_savefile
        self.param_savefile = param_savefile

        if os.path.exists(self.net_savefile):
            self.module = torch.load(self.net_savefile)
            print("net load successfully")
        elif os.path.exists(self.param_savefile):
            self.module.load_state_dict(torch.load(self.param_savefile))
            print("net params load successfully")
        self.module.to(self.device)
        # self.optimizer = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}])
        self.optimizer = optim.Adam(self.module.parameters())
        self.loss_fn = nn.MSELoss(reduction="sum")

    def train(self, epoch=1, batchsize=2, pic_dir="../datas/data1/data", mask_dir="../datas/data1/label", savebatch=100):
        # epoch = 1
        # batchsize = 2
        dataset = UnetDataset(pic_dir, mask_dir)
        dataloader = data.DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            self.module.train()
            for j, (img_data, label_data) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                label_data = label_data.to(self.device)
                output = self.module(img_data)
                loss = self.loss_fn(output, label_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if j % savebatch == 0:
                #     print("loss: {0}".format(loss.detach().item()))
                if j % savebatch == 0:
                    print("epoch: %d, batch:%d, loss: %.4f" % (i, j, loss.detach().item()))
                    torch.save(self.module, self.net_savefile)
                    print("net save successfully")
                    torch.save(self.module.state_dict(),self.param_savefile)

        return 0


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, precision=8)
    print(0)
    # trainer = Trainer()
    # trainer.train()
