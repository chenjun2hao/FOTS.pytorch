'''
Created on 2019-6-30

@author: chenjun2hao
'''

import torch, os
import numpy as np
import cv2

import tools.net_utils
import tools.data_gen
from tools.data_gen import draw_box_points
import timeit

import math
import random

from tools.models import ModelResNetSep2, OwnModel
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

# from torch_baidu_ctc import ctc_loss, CTCLoss
from warpctc_pytorch import CTCLoss
from tools.ocr_test_utils import print_seq_ext
from rroi_align.modules.rroi_align import _RRoiAlign
from src.utils import strLabelConverter
from src.utils import alphabet
from src.utils import averager


import unicodedata as ud
import tools.ocr_gen
from torch import optim
import argparse


lr_decay = 0.99
momentum = 0.9
weight_decay = 0
batch_per_epoch = 1000
disp_interval = 10

norm_height = 44


def intersect(a, b):
  '''Determine the intersection of two rectangles'''
  rect = (0,0,0,0)
  r0 = max(a[0],b[0])
  c0 = max(a[1],b[1])
  r1 = min(a[2],b[2])
  c1 = min(a[3],b[3])
  # Do we have a valid intersection?
  if r1 > r0 and  c1 > c0: 
      rect = (r0,c0,r1,c1)
  return rect

def union(a, b):
  r0 = min(a[0],b[0])
  c0 = min(a[1],b[1])
  r1 = max(a[2],b[2])
  c1 = max(a[3],b[3])
  return (r0,c0,r1,c1)

def area(a):
  '''Computes rectangle area'''
  width = a[2] - a[0]
  height = a[3] - a[1]
  return width * height
  
def process_boxes(images, im_data, iou_pred, roi_pred, angle_pred, score_maps, gt_idxs, gtso, lbso, features, net, ctc_loss, opts, converter, debug = False):
  '''iou_pred:类别  roi_pred:上，下，左，右  angle_pred:预测的角度  score_maps:目标分类label'''
  gt_good = 0
  gt_proc = 0

  # 0. 对预测的rroi进行筛选，选出满足要求的rroi，求误差//对一个batch进行操作
  rrois = []
  labels = []
  for bid in range(iou_pred.size(0)):
    
    gts = gtso[bid]               # 文字区域
    lbs = lbso[bid]               # 文字
    
    gt_proc = 0
    gt_good = 0
    
    gts_count = {}                # 每个rroi区域的计数，避免一个rroi多次用于训练

    iou_pred_np = iou_pred[bid].data.cpu().numpy()
    iou_map = score_maps[bid]
    to_walk = iou_pred_np.squeeze(0) * iou_map * (iou_pred_np.squeeze(0) > 0.5)           # 预测为文本区域且实际也为文本的索引点
    
    roi_p_bid = roi_pred[bid].data.cpu().numpy()      # 预测出来的roi
    gt_idx = gt_idxs[bid]                             # 第i个文字区域的label
    
    if debug:
      img = images[bid]
      img += 1
      img *= 128
      img = np.asarray(img, dtype=np.uint8)
    
    xy_text = np.argwhere(to_walk > 0)                          # 返回的顺序为：（行，列）
    random.shuffle(xy_text)
    xy_text = xy_text[0:min(xy_text.shape[0], 100)]             # 选出100个点
    
    # 对预测的点进行crop
    for i in range(0, xy_text.shape[0]):
      if opts.geo_type == 1:
        break
      pos = xy_text[i, :]
      
      gt_id = gt_idx[pos[0], pos[1]]                # 每个点对应的目标文字区域label
      
      if not gt_id in gts_count:
        gts_count[gt_id] = 0
      
      # 1. 一个文字区域最多用2次
      if gts_count[gt_id] > 2:                      # 一个目标文字区域最多可以用几次
        continue

      # 2. 文字是‘##’
      gt = gts[gt_id]                               # 当前点预测区域对应的label区域，和label文本
      gt_txt = lbs[gt_id]
      if gt_txt.startswith('##'):
        continue

      # 3. 目标文字区域的高度
      dhgt =  gt[1] - gt[0]                         # 
      h_gt = math.sqrt(dhgt[0] * dhgt[0] + dhgt[1] * dhgt[1])       # 标注label短边值
      if h_gt < 10:
        continue

      # 4. 标注的区域超出了图像范围
      if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(3):
        continue 
      
      # 5. 预测角度和真实角度相差太大
      angle_sin = angle_pred[bid, 0, pos[0], pos[1]] 
      angle_cos = angle_pred[bid, 1, pos[0], pos[1]]
      angle = math.atan2(angle_sin, angle_cos)            # 预测的角度和真实的角度
      angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
      if math.fabs(angle_gt - angle) > math.pi / 16:      # 预测角度和真实角度相差11.25度以上,原始为16
        continue

      # 6. 求倾斜4边形4条边的中点——4个角点
      offset = roi_p_bid[:, pos[0], pos[1]]               # 得到当前点的rroi，对应的含义为：上下左右
      posp = pos + 0.25                                   # 顺序为h,w——y,x
      pos_g = np.array([(posp[1] - offset[0] * math.sin(angle)) * 4, (posp[0] - offset[0] * math.cos(angle)) * 4 ])         # 求出的是x,y。xy_test,返回的是（行，列），转换到图像坐标系是（y，x）
      pos_g2 = np.array([ (posp[1] + offset[1] * math.sin(angle)) * 4, (posp[0] + offset[1] * math.cos(angle)) * 4 ])
      pos_r = np.array([(posp[1] - offset[2] * math.cos(angle)) * 4, (posp[0] - offset[2] * math.sin(angle)) * 4 ])
      pos_r2 = np.array([(posp[1] + offset[3] * math.cos(angle)) * 4, (posp[0] + offset[3] * math.sin(angle)) * 4 ])
      
      center = (pos_g + pos_g2 + pos_r + pos_r2) / 2 - [4*pos[1], 4*pos[0]]    
      #center = (pos_g + pos_g2 + pos_r + pos_r2) / 4   # 求中心
      dw = pos_r - pos_r2                               # 长边
      dh =  pos_g - pos_g2                              # 短边
      w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])      # 长边值
      h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])      # 短边值
    
      rect = ( (center[0], center[1]), (w, h), angle * 180 / math.pi )        # 预测的矩形，网络输出的是sin和cos，angle为弧度
      pts = cv2.boxPoints(rect)                                               # 4边形4个角点的值
      
      # 7. 求倾斜4边形和目标矩形的IOU// 这里是用矩形的方式求的
      pred_bbox = cv2.boundingRect(pts)                                       # 倾斜4边形的外接矩形，[x,y,w,h]
      pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]]    # 返回的是
      pred_bbox[2] += pred_bbox[0]
      pred_bbox[3] += pred_bbox[1]
      gt_bbox = [gt[:, 0].min(), gt[:, 1].min(), gt[:, 0].max(), gt[:, 1].max()]    # 目标box的外接矩形
      
      inter = intersect(pred_bbox, gt_bbox)             # 交集
      uni = union(pred_bbox, gt_bbox)                   # 并集
      ratio = area(inter) / float(area(uni))            # 求两个矩形的交并比
      
      if ratio < 0.9:                                  # 交并比小于0.9则舍弃
        continue
      hratio = min(h, h_gt) / max(h, h_gt)              # 高度相差太多
      if hratio < 0.5:
        continue
      
      # 8. 将rroi按rroi_align的要求进行整理
      angle = -angle / 3.1415926535 * 180
      rrois.append([bid, center[0], center[1], h, w, angle])   # 将多个rroi添加在一起
      labels.append(gt_txt)
      gts_count[gt_id] += 1
      gt_proc += 1

    # 8.1. for debug: 自己读入图片进行测试// 以上为对预测的rroi进行筛选
    # img = cv2.imread('./rroi_align/data/timg.jpeg')
    # gts = [[[206,111],[199,95],[349,60],[355,80]]]
    # im_data = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # 显示测试
    # im_data = im_data.to(torch.float).cuda()

    # 9. 为了引导误差的收敛方向，每张图片都将标注的rroi crop出来进行训练
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
        
        rrois.append([bid, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle_gt[gt_id]])   # 将标注的rroi写入
        labels.append(gt_txt)
        gt_good +=1
      

    # 11. debug显示标注的区域
    if debug:
      rois = torch.tensor(rrois).to(torch.float).cuda()
      pooled_height = 44
      maxratio = rois[:,4] / rois[:,3]
      maxratio = maxratio.max().item()
      pooled_width = math.ceil(pooled_height * maxratio)

      roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)        # 声明类
      pooled_feat = roipool(im_data, rois.view(-1, 6))

      for i in range(pooled_feat.shape[0]):

        x_d = pooled_feat.data.cpu().numpy()[i]
        x_data_draw = x_d.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)

        x_data_draw += 1
        x_data_draw *= 128
        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        x_data_draw = x_data_draw[:, :, ::-1]
        cv2.imshow('crop %d' % i, x_data_draw)
        cv2.imwrite('./data/tshow/crop%d.jpg' % i, x_data_draw)
            
      cv2.imshow('img', img)
      cv2.waitKey(100)


  # 12. 进行ctc label的转换 // 以上都是为了求rrois和labels // 这里是求的一个batch内的rroi
  if len(labels) > 32:
    labels = labels[:32]
    rrois = rrois[:32]
  text, label_length = converter.encode(labels)

  # 13.rroi_align, 特征前向传播，并求ctcloss
  rois = torch.tensor(rrois).to(torch.float).cuda()
  pooled_height = 11                                      # 从feature上crop出来特征
  maxratio = rois[:, 4] / rois[:, 3]
  maxratio = maxratio.max().item()
  pooled_width = math.ceil(pooled_height * maxratio)

  # 12.1 直接从原图中进行crop
  roipool = _RRoiAlign(pooled_height, pooled_width, 1.0 / 4)  # 声明类
  pooled_feat = roipool(features[1], rois.view(-1, 6))

  # 13.1 显示所有的crop区域
  alldebug = 0
  if alldebug:
      for i in range(pooled_feat.shape[0]):

        x_d = pooled_feat.data.cpu().numpy()[i]
        x_data_draw = x_d.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)

        x_data_draw += 1
        x_data_draw *= 128
        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        x_data_draw = x_data_draw[:, :, ::-1]
        cv2.imshow('crop %d' % i, x_data_draw)
        cv2.imwrite('./data/tshow/crop%d.jpg' % i, x_data_draw)

      for j in range(im_data.size(0)):
        img = im_data[j].cpu().numpy().transpose(1,2,0)
        img = (img + 1) * 128
        img = np.asarray(img, dtype=np.uint8)
        img = img[:, :, ::-1]
        cv2.imshow('img%d'%j, img)
        cv2.imwrite('./data/tshow/img%d.jpg' % j, img)
      cv2.waitKey(100)
      
  # ocr_features = net.forward_features(pooled_feat)
  preds = net.forward_ocr(pooled_feat)
  preds = preds.permute(2, 0, 1)

  # preds = net.ocr_forward(pooled_feat)

  preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))       # 求ctc loss
  loss1 = ctc_loss(preds, text, preds_size, label_length) / preds.size(1)    # 求一个平均

  return loss1, gt_good , gt_proc
