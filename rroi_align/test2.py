'''
采用rroi_align对旋转的文字进行矫正和crop
data：2019-6-24
author:yibao2hao
注意:
    1. im_data和rois都要是cuda
    2. roi为[index, x, y, h, w, theta]
    3. 增加了batch操作支持
    4. 
'''
from modules.rroi_align import _RRoiAlign
import torch
import cv2
import numpy as np
import math
import random
from math import sin, cos, floor, ceil
import matplotlib.pyplot as plt
from torch.autograd import Variable


if __name__=='__main__':

    path = './rroi_align/data/timg.jpeg'
    # path = './data/grad.jpg'
    im_data = cv2.imread(path)
    img = im_data.copy()
    im_data = torch.from_numpy(im_data).unsqueeze(0).permute(0,3,1,2)
    im_data = im_data
    im_data = im_data.to(torch.float).cuda()
    im_data = Variable(im_data, requires_grad=True)

    # plt.imshow(img)
    # plt.show()

    # 参数设置
    debug = True
    # 居民身份证的坐标位置
    gt3 = np.asarray([[200,218],[198,207],[232,201],[238,210]])      # 签发机关
    gt1 = np.asarray([[205,150],[202,126],[365,93],[372,111]])     # 居民身份证
    # # gt2 = np.asarray([[205,150],[202,126],[365,93],[372,111]])     # 居民身份证
    gt2 = np.asarray([[206,111],[199,95],[349,60],[355,80]])       # 中华人民共和国
    gt4 = np.asarray([[312,127],[304,105],[367,88],[374,114]])       # 份证
    gt5 = np.asarray([[133,168],[118,112],[175,100],[185,154]])      # 国徽
    # gts = [gt1, gt2, gt3, gt4, gt5]
    gts = [gt2, gt4, gt5]
    
    
    roi = []
    for i,gt in enumerate(gts):
        center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4        # 求中心点

        dw = gt[2, :] - gt[1, :]
        dh =  gt[1, :] - gt[0, :] 
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])                    # 宽度和高度
        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])  + random.randint(-2, 2)

        angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
        angle_gt = -angle_gt / 3.1415926535 * 180                       # 需要加个负号

        roi.append([0, center[0], center[1], h, w, angle_gt])           # roi的参数

    rois = torch.tensor(roi)  
    rois = rois.to(torch.float).cuda()

    pooled_height = 44
    maxratio = rois[:,4] / rois[:,3]
    maxratio = maxratio.max().item()
    pooled_width = math.ceil(pooled_height * maxratio)

    roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)
    # 执行rroi_align操作
    pooled_feat = roipool(im_data, rois.view(-1, 6))

    res = pooled_feat.pow(2).sum()
    # res = pooled_feat.sum()
    res.backward()

    if debug:
        for i in range(pooled_feat.shape[0]):
            x_d = pooled_feat.data.cpu().numpy()[i]
            x_data_draw = x_d.swapaxes(0, 2)
            x_data_draw = x_data_draw.swapaxes(0, 1)
        
            x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
            cv2.imshow('im_data_gt %d' % i, x_data_draw)
            cv2.imwrite('./rroi_align/data/res%d.jpg' % i, x_data_draw)
            
        cv2.imshow('img', img)

        # 显示梯度
        im_grad = im_data.grad.data.cpu().numpy()[0]
        im_grad = im_grad.swapaxes(0, 2)
        im_grad = im_grad.swapaxes(0, 1)
        
        im_grad = np.asarray(im_grad, dtype=np.uint8)
        cv2.imshow('grad', im_grad)
        cv2.imwrite('./rroi_align/data/grad.jpg',im_grad)

        # 
        grad_img = img + im_grad
        cv2.imwrite('./rroi_align/data/grad_img.jpg', grad_img)
    cv2.waitKey(100)
    print(pooled_feat.shape)
    
