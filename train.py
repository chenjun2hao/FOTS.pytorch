import torch, os
import numpy as np
import cv2
import math
import random
import time

import tools.net_utils as net_utils
import tools.data_gen as data_gen
from tools.data_gen import draw_box_points

from tools.models import ModelResNetSep2, OwnModel
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from warpctc_pytorch import CTCLoss
from tools.ocr_test_utils import print_seq_ext
from rroi_align.modules.rroi_align import _RRoiAlign
from src.utils import strLabelConverter
from src.utils import alphabet
from src.utils import averager
from src.ocr_process import process_boxes

import unicodedata as ud
import tools.ocr_gen
from torch import optim
import argparse
     

def main(opts):

  ## 1. 初始化模型
  nclass = len(alphabet) + 1            # 训练ICDAR2015
  model_name = 'E2E-MLT'
  net = ModelResNetSep2(attention=True, nclass=nclass)
  print("Using {0}".format(model_name))

  learning_rate = opts.base_lr
  # optimizer = torch.optim.Adam(net.parameters(), lr=opts.base_lr, weight_decay=weight_decay)
  optimizer = optim.Adam(net.parameters(), lr=opts.base_lr, betas=(0.5, 0.999))
  step_start = 0

  ### //预训练模型初始化，第一种：只修改conv11的维度 
  model_dict = net.state_dict()
  if os.path.exists(opts.model):
      print('loading pretrained model from %s' % opts.model)
      pretrained_model = ModelResNetSep2(attention=True, nclass=7500)                 # pretrained model from:https://github.com/MichalBusta/E2E-MLT
      pretrained_model.load_state_dict(torch.load(opts.model)['state_dict'])
      pretrained_dict = pretrained_model.state_dict()
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'conv11' not in k and 'rnn' not in k}
      model_dict.update(pretrained_dict)
      net.load_state_dict(model_dict)
  ### 第二种：直接接着前面训练
  # if os.path.exists(opts.model):
  #   print('loading model from %s' % args.model)
  #   step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)
  ### 
  if opts.cuda:
    net.cuda()
  net.train()


  ## 2. 定义数据集
  converter = strLabelConverter(alphabet)
  ctc_loss = CTCLoss()
  data_generator = data_gen.get_batch(num_workers=opts.num_readers, 
           input_size=opts.input_size, batch_size=opts.batch_size, 
           train_list=opts.train_list, geo_type=opts.geo_type)
  # dg_ocr = ocr_gen.get_batch(num_workers=2,
  #         batch_size=opts.ocr_batch_size, 
  #         train_list=opts.ocr_feed_list, in_train=True, norm_height=norm_height, rgb=True)            # 训练OCR识别的数据集

  ## 3. 变量初始化
  bbox_loss = averager(); seg_loss = averager(); angle_loss = averager()
  loss_ctc = averager(); train_loss = averager()


  ## 4. 开始训练
  for step in range(step_start, opts.max_iters):
  
    # 读取数据
    images, image_fns, score_maps, geo_maps, training_masks, gtso, lbso, gt_idxs = next(data_generator)
    im_data = net_utils.np_to_variable(images.transpose(0, 3, 1, 2), is_cuda=opts.cuda)
    start = time.time()
    try:
      seg_pred, roi_pred, angle_pred, features = net(im_data)
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      continue
    
    # for EAST loss
    smaps_var = net_utils.np_to_variable(score_maps, is_cuda=opts.cuda)
    training_mask_var = net_utils.np_to_variable(training_masks, is_cuda=opts.cuda)
    angle_gt = net_utils.np_to_variable(geo_maps[:, :, :, 4], is_cuda=opts.cuda)
    geo_gt = net_utils.np_to_variable(geo_maps[:, :, :, [0, 1, 2, 3]], is_cuda=opts.cuda)
    try:
      loss = net.loss(seg_pred, smaps_var, training_mask_var, angle_pred, angle_gt, roi_pred, geo_gt)
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      continue
    
    bbox_loss.add(net.box_loss_value.item()); seg_loss.add(net.segm_loss_value.item()); angle_loss.add(net.angle_loss_value.item())
    
    
    # 训练ocr的部分
    try:
      # 10000步之前都是用文字的标注区域训练的//E2E-MLT中采用的这种策略
      if step > 10000 or True:            #this is just extra augumentation step ... in early stage just slows down training
        ctcl, gt_target , gt_proc = process_boxes(images, im_data, seg_pred[0], roi_pred[0], angle_pred[0], score_maps, gt_idxs, gtso, lbso, features, net, ctc_loss, opts, converter, debug=opts.debug)
        loss_ctc.add(ctcl)
        loss = loss + ctcl.cuda()
        train_loss.add(loss.item())

      net.zero_grad()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      pass

    if step % opts.disp_interval == 0:
      end = time.time()            # 计算耗时
      ctc_loss_val2 = 0.0
      print('epoch %d[%d], loss: %.3f, bbox_loss: %.3f, seg_loss: %.3f, ang_loss: %.3f, ctc_loss: %.3f, time %.3f' % (
          step / 1000 * opts.batch_size, step, train_loss.val(), bbox_loss.val(), seg_loss.val(), angle_loss.val(), loss_ctc.val(), end-start))

    # for save mode
    if step > step_start and (step % ((1000 / opts.batch_size)*20) == 0):               # 20代保存一次
      save_name = os.path.join(opts.save_path, '{}_{}.h5'.format(model_name, step))
      state = {'step': step,
               'learning_rate': learning_rate,
              'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict()}
      torch.save(state, save_name)
      print('save model: {}'.format(save_name))
      train_loss.reset(); bbox_loss.reset(); seg_loss.reset(); angle_loss.reset(); loss_ctc.reset()               # 避免超出了范围



if __name__ == '__main__': 
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-train_list', default='./data/ICDAR2015.txt')
  parser.add_argument('-ocr_feed_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-save_path', default='backup')
  parser.add_argument('-model', default='./weights/e2e-mlt.h5')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-batch_size', type=int, default=2)
  parser.add_argument('-ocr_batch_size', type=int, default=256)
  parser.add_argument('-num_readers', type=int, default=4, help='it is faster')
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-input_size', type=int, default=512)
  parser.add_argument('-geo_type', type=int, default=0)
  parser.add_argument('-base_lr', type=float, default=0.001)
  parser.add_argument('-max_iters', type=int, default=300000)
  parser.add_argument('-disp_interval', type=int, default=5)
  
  args = parser.parse_args()

  main(args)
  
