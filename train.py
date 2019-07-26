import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset, fots_collotfn
from model import EAST
from model2 import FOTS
from loss import Loss
import os
import time
import numpy as np
from util import roi_process
from util import strLabelConverter
from util import averager


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, ocr_branch):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=fots_collotfn())

    with open('./ICDAR_2015/icdar2015_alphabet.txt', 'r') as f:
        alphabet = f.readlines()[0]
    converter = strLabelConverter(alphabet=alphabet)
    nclass = len(alphabet) + 1

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FOTS(nclass=nclass)

    pretrained = './pths/model_epoch_60.pth'
    if pretrained:
        print('loading pretrained model from %s' % pretrained)
        model.load_state_dict(torch.load(pretrained))

    # ### //预训练模型初始化，第一种：只修改conv11的维度
    # model_dict = model.state_dict()
    # pretrained = './pths/east_vgg16.pth'
    # if pretrained:
    # 	print('loading pretrained model from %s' % pretrained)
    # 	pretrained_model = EAST()
    # 	pretrained_model.load_state_dict(torch.load(pretrained))
    # 	pretrained_dict = pretrained_model.state_dict()
    # 	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 	model_dict.update(pretrained_dict)
    # 	model.load_state_dict(model_dict)

    ### //不训练某些层
    # frozen_layers = [model.extractor, model.merge, model.output]
    # for layer in frozen_layers:
    # 	for name, value in layer.named_parameters():
    # 		value.requires_grad = False
    # params = filter(lambda p: p.requires_grad, model.parameters())

    data_parallel = False
    # if torch.cuda.device_count() > 1:
    # 	model = nn.DataParallel(model)
    # 	data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)

    seg_loss = averager();
    angle_loss = averager();
    iou_loss = averager()
    ocr_loss = averager()

    model.train()
    for epoch in range(epoch_iter):

        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map, ocr_img, text_polys, texts) in enumerate(train_loader):
            try:
                start_time = time.time()
                img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(
                    device), ignored_map.to(device)
                pred_score, pred_geo = model(img)
                loss, seg, angle, iou = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
                seg_loss.add(seg)
                angle_loss.add(angle)
                iou_loss.add(iou)

                if ocr_branch:
                    ocr_img = ocr_img.to(device)
                    rrois, labels = roi_process(ocr_img, text_polys, texts)
                    encode_labels, label_length = converter.encode(labels)
                    # shared_feature = model.merge(model.extractor(ocr_img))
                    shared_feature = model.ocr_shared(ocr_img)
                    loss1 = model.ocr_forward(shared_feature, rrois, encode_labels, label_length)
                    ocr_loss.add(loss1)

                    loss = loss + loss1.cuda()

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    'epoch %d[%d], loss: %.3f, seg_loss: %.3f, angle_loss: %.3f, iou_loss: %.3f, ctc_loss: %.3f,  time %.3f' % (
                        epoch + 1, i, loss.item(), seg_loss.val(), angle_loss.val(), iou_loss.val(), loss1.item(),
                        time.time() - start_time))
            except:
                import sys, traceback
                traceback.print_exc(file=sys.stdout)
                continue

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / batch_size),
                                                                  time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    train_img_path = os.path.abspath('./ICDAR_2015/ch4_training_images')
    train_gt_path = os.path.abspath('./ICDAR_2015/ch4_training_localization_transcription_gt')
    pths_path = './pths'
    batch_size = 2
    lr = 1e-4
    num_workers = 8
    epoch_iter = 600
    save_interval = 2
    ocr_branch = True
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, ocr_branch)

