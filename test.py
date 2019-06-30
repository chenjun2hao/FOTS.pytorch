'''
Created on 2019-06-30

@author: chenjun2hao
'''

import os
import cv2
import glob
import numpy as np
import torch
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from nms import get_boxes

from tools.models import ModelResNetSep2, OwnModel
import tools.net_utils as net_utils
from src.utils import strLabelConverter, alphabet
from tools.ocr_utils import ocr_image, align_ocr
from tools.data_gen import draw_box_points

def resize_image(im, max_size = 1585152, scale_up=True):

  if scale_up:
    image_size = [im.shape[1] * 3 // 32 * 32, im.shape[0] * 3 // 32 * 32]
  else:
    image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
  while image_size[0] * image_size[1] > max_size:
    image_size[0] /= 1.2
    image_size[1] /= 1.2
    image_size[0] = int(image_size[0] // 32) * 32
    image_size[1] = int(image_size[1] // 32) * 32

  resize_h = int(image_size[1])
  resize_w = int(image_size[0])

  scaled = cv2.resize(im, dsize=(resize_w, resize_h))
  return scaled, (resize_h, resize_w)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='./weights/FOTS_280000.h5')
  # parser.add_argument('-model', default='./weights/e2e-mlt.h5')
  parser.add_argument('-segm_thresh', default=0.5)
  parser.add_argument('-test_folder', default=r'./data/example_image/')
  parser.add_argument('-output', default='./data/ICDAR2015')

  font2 = ImageFont.truetype("./tools/Arial-Unicode-Regular.ttf", 18)

  args = parser.parse_args()

  # net = ModelResNetSep2(attention=True, nclass=len(alphabet)+1)
  net = ModelResNetSep2(attention=True, nclass=len(alphabet)+1)
  net_utils.load_net(args.model, net)
  net = net.eval()

  converter = strLabelConverter(alphabet)

  if args.cuda:
    print('Using cuda ...')
    net = net.cuda()

  test_path = os.path.realpath(args.test_folder)
  test_path = test_path + '/*.jpg'
  imagelist = glob.glob(test_path)
  with torch.no_grad():
    for path in imagelist:
      # path = '/home/yangna/deepblue/OCR/data/ICDAR2015/ch4_test_images/img_405.jpg'
      im = cv2.imread(path)

      im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
      images = np.asarray([im_resized], dtype=np.float)
      images /= 128
      images -= 1
      im_data = net_utils.np_to_variable(images.transpose(0, 3, 1, 2), is_cuda=args.cuda)
      seg_pred, rboxs, angle_pred, features = net(im_data)

      rbox = rboxs[0].data.cpu()[0].numpy()                   # 转变成h,w,c
      rbox = rbox.swapaxes(0, 1)
      rbox = rbox.swapaxes(1, 2)

      angle_pred = angle_pred[0].data.cpu()[0].numpy()

      segm = seg_pred[0].data.cpu()[0].numpy()
      segm = segm.squeeze(0)

      draw2 = np.copy(im_resized)
      boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

      img = Image.fromarray(draw2)
      draw = ImageDraw.Draw(img)

      out_boxes = []
      for box in boxes:

        pts  = box[0:8]
        pts = pts.reshape(4, -1)

        # det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
        det_text, conf, dec_s = align_ocr(net, converter, im_data, box, features, debug=0)
        if len(det_text) == 0:
          continue

        width, height = draw.textsize(det_text, font=font2)
        center =  [box[0], box[1]]
        draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
        out_boxes.append(box)
        print(det_text)

      im = np.array(img)
      for box in out_boxes:
        pts  = box[0:8]
        pts = pts.reshape(4, -1)
        draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

      cv2.imshow('img', im)
      basename = os.path.basename(path)
      cv2.imwrite(os.path.join(args.output, basename), im)
      cv2.waitKey(1000)

        


