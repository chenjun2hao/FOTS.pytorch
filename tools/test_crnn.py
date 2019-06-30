'''
Created on Sep 3, 2017

@author: Michal.Busta at gmail.com
'''

import torch, os
import numpy as np
import cv2

import net_utils
import data_gen
from data_gen import draw_box_points
import timeit

import math
import random

from models import ModelResNetSep2, OwnModel, CRNN
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

# from torch_baidu_ctc import ctc_loss, CTCLoss
from warpctc_pytorch import CTCLoss
from ocr_test_utils import print_seq_ext
from rroi_align.modules.rroi_align import _RRoiAlign
from src.utils import strLabelConverter
from src.utils import alphabet
from src.utils import process_crnn
from src.utils import ImgDataset
from src.utils import own_collate

import unicodedata as ud
import ocr_gen
from torch import optim
import argparse


lr_decay = 0.99
momentum = 0.9
weight_decay = 0
batch_per_epoch = 1000
disp_interval = 5

norm_height = 44

f = open('codec.txt', 'r')
codec = f.readlines()[0]
#codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[codec[i]] = index
  index += 1
f.close()

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
  
     
def main(opts):
  alphabet = '0123456789.'
  nclass = len(alphabet) + 1
  model_name = 'crnn'
  net = CRNN(nclass)
  print("Using {0}".format(model_name))

  if opts.cuda:
    net.cuda()
  learning_rate = opts.base_lr
  optimizer = torch.optim.Adam(net.parameters(), lr=opts.base_lr, weight_decay=weight_decay)

  if os.path.exists(opts.model):
    print('loading model from %s' % args.model)
    step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)

  ## 数据集
  converter = strLabelConverter(alphabet)
  dataset = ImgDataset(
      root='/home/yangna/deepblue/OCR/mech_demo2/dataset/imgs/image',
      csv_root='/home/yangna/deepblue/OCR/mech_demo2/dataset/imgs/train_list.txt',
      transform=None,
      target_transform=converter.encode
  )
  ocrdataloader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, collate_fn=own_collate
  )

  num_count = 0
  net = net.eval()

  converter = strLabelConverter(alphabet)
  ctc_loss = CTCLoss()

  for step in range(len(dataset)):

    try:
        data = next(data_iter)
    except:
        data_iter = iter(ocrdataloader)
        data = next(data_iter)

    im_data, gt_boxes, text = data
    im_data = im_data.cuda()

    try:
      res = process_crnn(im_data, gt_boxes, text, net, ctc_loss, converter, training=False)

      pred, target = res
      if pred == target[0]:
        num_count += 1
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      pass


    print('correct/total:%d/%d'%(num_count, len(dataset)))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-train_list', default='./data/small_train.txt')
  parser.add_argument('-ocr_feed_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-save_path', default='backup')
  parser.add_argument('-model', default='./backup/crnn_2000.h5')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-batch_size', type=int, default=1)
  parser.add_argument('-ocr_batch_size', type=int, default=256)
  parser.add_argument('-num_readers', type=int, default=1)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-input_size', type=int, default=512)
  parser.add_argument('-geo_type', type=int, default=0)
  parser.add_argument('-base_lr', type=float, default=0.001)
  parser.add_argument('-max_iters', type=int, default=300000)
  
  args = parser.parse_args()

  main(args)
  
