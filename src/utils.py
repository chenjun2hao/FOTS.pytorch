#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import collections
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import cv2
import os
import PIL
from tools.data_gen import generate_rbox, generate_rbox2
from tools.data_gen import load_gt_annoataion
from tools.data_gen import get_images
from tools.data_gen import random_rotation, random_perspective
import torchvision.transforms as transforms


from rroi_align.modules.rroi_align import _RRoiAlign

with open('./data/alphabet.txt', 'r') as f:
    alphabet = f.readlines()[0]


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts



class strLabelConverterForCTC(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self.sep = sep
        self.alphabet = alphabet.split(sep)
        self.alphabet.append('-')  # for `-1` index

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = text.split(self.sep)
            text = [self.dict[item] for item in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s.split(self.sep)) for s in text]
            text = self.sep.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        else:
            v = v
            count = 1

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


class halo():
    '''
    u:高斯分布的均值
    sigma:方差
    nums:在一张图片中随机添加几个光点
    prob:使用halo的概率
    '''

    def __init__(self, nums, u=0, sigma=0.2, prob=0.5):
        self.u = u  # 均值μ
        self.sig = math.sqrt(sigma)  # 标准差δ
        self.nums = nums
        self.prob = prob

    def create_kernel(self, maxh=32, maxw=50):
        height_scope = [10, maxh]  # 高度范围     随机生成高斯
        weight_scope = [20, maxw]  # 宽度范围

        x = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*height_scope))
        y = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*weight_scope))
        Gauss_map = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Gauss_map[i, j] = np.exp(-((x[i] - self.u) ** 2 + (y[j] - self.u) ** 2) / (2 * self.sig ** 2)) / (
                        math.sqrt(2 * math.pi) * self.sig)

        return Gauss_map

    def __call__(self, img):
        if random.random() < self.prob:
            Gauss_map = self.create_kernel(32, 60)  # 初始化一个高斯核,32为高度方向的最大值，60为w方向
            img1 = np.asarray(img)
            img1.flags.writeable = True  # 将数组改为读写模式
            nums = random.randint(1, self.nums)  # 随机生成nums个光点
            img1 = img1.astype(np.float)
            # print(nums)
            for i in range(nums):
                img_h, img_w = img1.shape
                pointx = random.randint(0, img_h - 10)  # 在原图中随机找一个点
                pointy = random.randint(0, img_w - 10)

                h, w = Gauss_map.shape  # 判断是否超限
                endx = pointx + h
                endy = pointy + w

                if pointx + h > img_h:
                    endx = img_h
                    Gauss_map = Gauss_map[1:img_h - pointx + 1, :]
                if img_w < pointy + w:
                    endy = img_w
                    Gauss_map = Gauss_map[:, 1:img_w - pointy + 1]

                # 加上不均匀光照
                img1[pointx:endx, pointy:endy] = img1[pointx:endx, pointy:endy] + Gauss_map * 255.0
            img1[img1 > 255.0] = 255.0  # 进行限幅，不然uint8会从0开始重新计数
            img = img1
        return Image.fromarray(np.uint8(img))


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


class GBlur(object):
    def __init__(self, radius=2, prob=0.5):
        radius = random.randint(0, radius)
        self.blur = MyGaussianBlur(radius=radius)
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = img.filter(self.blur)
        return img


class RandomBrightness(object):
    """随机改变亮度
        pil:pil格式的图片
    """

    def __init__(self, prob=1.5):
        self.prob = prob

    def __call__(self, pil):
        rgb = np.asarray(pil)
        if random.random() < self.prob:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 0.7, 0.9, 1.2, 1.5, 1.7])  # 随机选择一个
            # adjust = random.choice([1.2, 1.5, 1.7, 2.0])      # 随机选择一个
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(np.uint8(rgb)).convert('L')


class randapply(object):
    """随机决定是否应用光晕、模糊或者二者都用

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def process_crnn(im_data, gtso, lbso, net, ctc_loss, converter, training):
    num_gt = len(gtso)
    rrois = []
    labels = []
    for kk in range(num_gt):
        gts = gtso[kk]
        lbs = lbso[kk]
        if len(gts) != 0:
            gt = np.asarray(gts)
            center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4        # 求中心点
            dw = gt[:, 2, :] - gt[:, 1, :]
            dh =  gt[:, 1, :] - gt[:, 0, :] 
            poww = pow(dw, 2)
            powh = pow(dh, 2)
            w = np.sqrt(poww[:, 0] + poww[:,1])
            h = np.sqrt(powh[:,0] + powh[:,1])  + random.randint(-2, 2)
            angle_gt = ( np.arctan2((gt[:,2,1] - gt[:,1,1]), gt[:,2,0] - gt[:,1,0]) + np.arctan2((gt[:,3,1] - gt[:,0,1]), gt[:,3,0] - gt[:,0,0]) ) / 2        # 求角度
            angle_gt = -angle_gt / 3.1415926535 * 180                                   # 需要加个负号

            # 10. 对每个rroi进行判断是否用于训练
            for gt_id in range(0, len(gts)):
                
                gt_txt = lbs[gt_id]                       # 文字判断
                if gt_txt.startswith('##'):
                    continue
                
                gt = gts[gt_id]                           # 标注信息判断
                if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(2) or gt.min() < 0:
                    continue
                
                rrois.append([kk, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle_gt[gt_id]])   # 将标注的rroi写入
                labels.append(gt_txt)

    # labels = labels[0]
    # rrois = [rrois[0]]

    text, label_length = converter.encode(labels)

    # 13.rroi_align, 特征前向传播，并求ctcloss
    rois = torch.tensor(rrois).to(torch.float).cuda()
    pooled_height = 32
    maxratio = rois[:, 4] / rois[:, 3]
    maxratio = maxratio.max().item()
    pooled_width = math.ceil(pooled_height * maxratio)

    roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)  # 声明类
    pooled_feat = roipool(im_data, rois.view(-1, 6))

    # 13.1 显示所有的crop区域
    alldebug = 0
    if alldebug:
        for i in range(pooled_feat.shape[0]):

            x_d = pooled_feat.data.cpu().numpy()[i]
            x_data_draw = x_d.swapaxes(0, 2)
            x_data_draw = x_data_draw.swapaxes(0, 1)

            # x_data_draw += 1
            # x_data_draw *= 128
            x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
            x_data_draw = x_data_draw[:, :, ::-1]
            cv2.imshow('crop %d' % i, x_data_draw)
            cv2.imwrite('./data/tshow/crop%d.jpg' % i, x_data_draw)
            # cv2.imwrite('./data/tshow/%s.jpg' % labels[i], x_data_draw)

        for j in range(im_data.size(0)):
            img = im_data[j].cpu().numpy().transpose(1,2,0)
            img = (img + 1) * 128
            img = np.asarray(img, dtype=np.uint8)
            img = img[:, :, ::-1]
            cv2.imshow('img%d'%j, img)
            cv2.imwrite('./data/tshow/img%d.jpg' % j, img)
        cv2.waitKey(100)

    if training:
        preds = net.ocr_forward(pooled_feat)

        preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))       # 求ctc loss
        res = ctc_loss(preds, text, preds_size, label_length) / preds.size(1)    # 求一个平均
    else:
        labels_pred = net.ocr_forward(pooled_feat)

        _, labels_pred = labels_pred.max(2)
        # labels_pred = labels_pred.contiguous().view(-1)
        labels_pred = labels_pred.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([labels_pred.size(0)]))
        res = converter.decode(labels_pred.data, preds_size.data, raw=False)
        res = (res, labels)
    return res


class ImgDataset(Dataset):
    def __init__(self, root=None, csv_root=None, transform=None, target_transform=None):
        self.root = root
        with open(csv_root) as f:
            self.data = f.readlines()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        per_label = self.data[idx].rstrip().split('\t')
        imgpath = os.path.join(self.root, per_label[0])
        srcimg = cv2.imread(imgpath)
        img = srcimg[:, :, ::-1].copy()

        if self.transform:
            img = self.transform(img)
            # img = torch.tensor(img, dtype=torch.float)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img = img.float()
        
        temp = [[int(x) for x in per_label[2:6]]]

        roi = []
        for i in range(len(temp)):
            temp1 = np.asarray([[temp[i][0], temp[i][3]], [temp[i][0],temp[i][1]], [temp[i][2],temp[i][1]], [temp[i][2],temp[i][3]]])
            roi.append(temp1)

        # for debug show
        #     cv2.rectangle(srcimg, (temp1[1][0], temp1[1][1]), (temp1[3][0], temp1[3][1]), (255, 0, 0), thickness=2)
        # #     temp1 = temp1.reshape(-1,1,2)
        # #     cv2.polylines(srcimg,[temp1],False,(0,255,255), thickness=3)
        # plt.imshow(srcimg)
        # plt.show()

        text = [per_label[1].lstrip(), per_label[6].lstrip()]


        return img, roi, text          # gt_box的标注信息为x1,y1,x2,y2, 返回一个名字


class ImgDataset2(Dataset):
    def __init__(self, root=None, csv_root=None, transform=None, target_transform=None):
        self.root = root
        with open(csv_root) as f:
            self.data = f.readlines()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        per_label = self.data[idx].rstrip().split('\t')
        imgpath = os.path.join(self.root, per_label[0])
        srcimg = cv2.imread(imgpath)
        img = srcimg[:, :, ::-1].copy()

        if self.transform:
            img = self.transform(img)
            # img = torch.tensor(img, dtype=torch.float)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float()

        temp = [[int(x) for x in per_label[2:6]],
                [int(x) for x in per_label[7:11]]]

        roi = []
        for i in range(len(temp)):
            temp1 = np.asarray([[temp[i][0], temp[i][3]], [temp[i][0], temp[i][1]], [temp[i][2], temp[i][1]],
                                [temp[i][2], temp[i][3]]])
            roi.append(temp1)

        # cv2.rectangle(srcimg, (temp[0][0], temp[0][1]), (temp[0][2], temp[0][3]), (255, 0, 0), thickness=2)
        #     temp1 = temp1.reshape(-1,1,2)
        #     cv2.polylines(srcimg,[temp1],False,(0,255,255), thickness=3)
        # plt.imshow(srcimg)
        # plt.show()

        text = [per_label[1].lstrip(), per_label[6].lstrip()]

        return img, roi, text  # gt_box的标注信息为x1,y1,x2,y2, 返回一个名字


def own_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    img = []
    gt_boxes = []
    texts = []
    for per_batch in batch:
        img.append(per_batch[0])
        gt_boxes.append(per_batch[1])
        texts.append(per_batch[2])

    return torch.stack(img, 0), gt_boxes, texts


class E2Edataset(Dataset):
    def __init__(self, train_list, input_size=512, in_train = True):
        super(E2Edataset, self).__init__()
        self.image_list = np.array(get_images(train_list))

        print('{} training images in {}'.format(self.image_list.shape[0], train_list))

        self.transform = transforms.Compose([
                    transforms.ColorJitter(.3,.3,.3,.3),
                    transforms.RandomGrayscale(p=0.1)  ])
        self.input_size = input_size
        self.in_train = in_train

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_name = self.image_list[index]

        im = cv2.imread(im_name)                # 图片

        txt_fn = im_name.replace(os.path.basename(im_name).split('.')[1], 'txt')
        base_name = os.path.basename(txt_fn)
        txt_fn_gt = '{0}/gt_{1}'.format(os.path.dirname(im_name), base_name)

        # 载入标注信息
        text_polys, text_tags, labels_txt = load_gt_annoataion(txt_fn_gt, txt_fn_gt.find('/icdar-2015-Ch4/') != -1)

        pim = PIL.Image.fromarray(np.uint8(im))
        if self.transform:
            pim = self.transform(pim)
        im = np.array(pim)

        text_polys, text_tags, labels_txt = load_gt_annoataion(txt_fn_gt, txt_fn_gt.find('/icdar-2015-Ch4/') != -1)

        new_h, new_w, _ = im.shape
        score_map, geo_map, training_mask, gt_idx, gt_out, labels_out = generate_rbox(im, (new_h, new_w), text_polys, text_tags, labels_txt, vis=False)

        im = np.asarray(im, dtype=np.float)
        im /= 128
        im -= 1
        im = torch.from_numpy(im).permute(2,0,1).float()

        return im, score_map, geo_map, training_mask, gt_idx, gt_out, labels_txt

    def preprocess(self, im, text_polys):
        """
            图片预处理函数，就是图像增强
        """
        if random.uniform(0, 100) < 50 or im.shape[0] < 600 or im.shape[1] < 600:         # 随机在周边填充
            top = int(random.uniform(300, 500))
            bottom = int(random.uniform(300, 500))
            left = int(random.uniform(300, 500))
            right = int(random.uniform(300, 500))
            im = cv2.copyMakeBorder(im, top , bottom, left, right, cv2.BORDER_CONSTANT)
        if len(text_polys) > 0:
            text_polys[:, :, 0] += left
            text_polys[:, :, 1] += top

        if random.uniform(0, 100) < 30:
            im = random_rotation(im, text_polys)            # 随机旋转
        if random.uniform(0, 100) < 30:
            im = random_perspective(im, text_polys)         # ？？？随机干哈

          #im = random_crop(im, text_polys, vis=False)

        scalex = random.uniform(0.5, 2)                   # 宽度和高度方向上随机比例
        scaley = scalex * random.uniform(0.8, 1.2)
        im = cv2.resize(im, dsize=(int(im.shape[1] * scalex), int(im.shape[0] * scaley)))
        text_polys[:, :, 0] *= scalex
        text_polys[:, :, 1] *= scaley

        if random.randint(0, 100) < 10:
            im = np.invert(im)

        return im, text_polys


def E2Ecollate(batch):
    img = []
    gt_boxes = []
    texts = []
    for per_batch in batch:
        img.append(per_batch[0])
        gt_boxes.append(per_batch[5])
        texts.append(per_batch[6])

    return torch.stack(img, 0), gt_boxes, texts


if __name__ == '__main__':
    llist = './data/ICDAR2015.txt'

    data = E2Edataset(train_list=llist)

    E2Edataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, collate_fn=E2Ecollate)

    for index, data in enumerate(E2Edataloader):
        im = data