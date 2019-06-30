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
from src.utils import E2Ecollate,E2Edataset

import unicodedata as ud
import ocr_gen
from torch import optim
import argparse


lr_decay = 0.99
momentum = 0.9
weight_decay = 0.9
batch_per_epoch = 10
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
  # alphabet = '0123456789.'
  nclass = len(alphabet) + 1
  model_name = 'E2E-CRNN'
  net = OwnModel(attention=True, nclass=nclass)
  print("Using {0}".format(model_name))

  if opts.cuda:
    net.cuda()
  learning_rate = opts.base_lr
  optimizer = torch.optim.Adam(net.parameters(), lr=opts.base_lr, weight_decay=weight_decay)
  optimizer = optim.Adam(net.parameters(), lr=opts.base_lr, betas=(0.5, 0.999))
  step_start = 0

  ### 第一种：只修改conv11的维度
  # model_dict = net.state_dict()
  # if os.path.exists(opts.model):
  #     print('loading pretrained model from %s' % opts.model)
  #     pretrained_model = OwnModel(attention=True, nclass=12)
  #     pretrained_model.load_state_dict(torch.load(opts.model)['state_dict'])
  #     pretrained_dict = pretrained_model.state_dict()
  #
  #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'rnn' not in k and 'conv11' not in k}
  #     model_dict.update(pretrained_dict)
  #     net.load_state_dict(model_dict)

  if os.path.exists(opts.model):
    print('loading model from %s' % args.model)
    step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)

  ## 数据集
  e2edata = E2Edataset(train_list=opts.train_list)
  e2edataloader = torch.utils.data.DataLoader(e2edata, batch_size=opts.batch_size, shuffle=True, collate_fn=E2Ecollate, num_workers=4)

  # 电表数据集
  # converter = strLabelConverter(alphabet)
  # dataset = ImgDataset(
  #     root='/home/yangna/deepblue/OCR/mech_demo2/dataset/imgs/image',
  #     csv_root='/home/yangna/deepblue/OCR/mech_demo2/dataset/imgs/train_list.txt',
  #     transform=None,
  #     target_transform=converter.encode
  # )
  # ocrdataloader = torch.utils.data.DataLoader(
  #     dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=own_collate
  # )

  net.train()

  converter = strLabelConverter(alphabet)
  ctc_loss = CTCLoss()

  for step in range(step_start, opts.max_iters):

    for index, date in enumerate(e2edataloader):
      im_data, gtso, lbso = date
      im_data = im_data.cuda()

      try:
        loss= process_crnn(im_data, gtso, lbso, net, ctc_loss, converter, training=True)

        net.zero_grad()
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      except:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        pass


      if index % disp_interval == 0:
        try:
          print('epoch:%d || step:%d || loss %.4f' % (step, index, loss))
        except:
          import sys, traceback
          traceback.print_exc(file=sys.stdout)
          pass

    if step > step_start and (step % batch_per_epoch == 0):
      save_name = os.path.join(opts.save_path, '{}_{}.h5'.format(model_name, step))
      state = {'step': step,
               'learning_rate': learning_rate,
              'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict()}
      torch.save(state, save_name)
      print('save model: {}'.format(save_name))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-train_list', default='./data/ICDAR2015.txt')
  parser.add_argument('-ocr_feed_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-save_path', default='backup')
  parser.add_argument('-model', default='./backup/E2E-CRNN_210.h5')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-batch_size', type=int, default=8)
  parser.add_argument('-ocr_batch_size', type=int, default=256)
  parser.add_argument('-num_readers', type=int, default=1)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-input_size', type=int, default=512)
  parser.add_argument('-geo_type', type=int, default=0)
  parser.add_argument('-base_lr', type=float, default=0.001)
  parser.add_argument('-max_iters', type=int, default=300000)
  
  args = parser.parse_args()

  main(args)
  
