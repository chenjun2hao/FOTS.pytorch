from modules.roi_pool import _RoIPooling
import torch
import cv2
import numpy as np
import math
import random
from math import sin, cos, floor, ceil


if __name__=='__main__':
    roipool = _RoIPooling(44, 328, 1.0)           #  类的初始化

    path = './data/timg.jpeg'
    im_data = cv2.imread(path)
    img = im_data.copy()
    im_data = torch.from_numpy(im_data).unsqueeze(0).permute(0,3,1,2)
    im_data = im_data
    im_data = im_data.to(torch.float)

    # 参数设置
    norm_height = 44
    debug = True
    # 居民身份证的坐标位置
    # gt = np.asarray([[200,218],[198,207],[232,201],[238,210]])
    gt = np.asarray([[205,150],[202,126],[365,93],[372,111]])
    
    center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4        # 求中心点

    dw = gt[2, :] - gt[1, :]
    dh =  gt[1, :] - gt[0, :] 
    w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])                    # 宽度和高度
    h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])  + random.randint(-2, 2)

    angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
    angle_gt = angle_gt / 3.1415926535 * 180

    rois = torch.tensor([0, center[0], center[1], h, w, angle_gt])  
    rois = rois.to(torch.float)

    # rroi_align传入的参数为roi的中心，w，h和arctan(theta),theta为角度
    # 参数设置
    pooled_width = 328
    pooled_height = 44
    channels = 3
    spatial_scale = 1.0
    index = pooled_height * pooled_width * channels
    imageHeight, imageWidth, channel = img.shape
    height, width = imageHeight, imageWidth
    output = torch.zeros(index)
    for i in range(index):
        n = i;
        pw = n % pooled_width;
        n /= pooled_width;
        ph = n % pooled_height;
        n /= pooled_height;
        c = n % channels;
        n /= channels;

        offset_bottom_rois = rois
        roi_batch_ind = offset_bottom_rois[0];
        cx = offset_bottom_rois[1];
        cy = offset_bottom_rois[2];
        h = offset_bottom_rois[3];
        w = offset_bottom_rois[4];
        angle = - offset_bottom_rois[5]/180.0*3.1415926535;

        # //TransformPrepare
        dx = -pooled_width/2.0;
        dy = -pooled_height/2.0;
        Sx = w*spatial_scale/pooled_width;
        Sy = h*spatial_scale/pooled_height;
        Alpha = cos(angle);
        Beta = sin(angle);
        Dx = cx*spatial_scale;
        Dy = cy*spatial_scale;

        M =[[0 for col in range(3)] for row in range(2)]
        M[0][0] = Alpha*Sx;
        M[0][1] = Beta*Sy;
        M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
        M[1][0] = -Beta*Sx;
        M[1][1] = Alpha*Sy;
        M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

        # float P[8];
        P =[0 for col in range(8)]
        P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
        P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
        P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
        P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
        P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
        P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
        P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
        P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

        leftMost = (max(torch.round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
        rightMost= (min(torch.round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
        topMost= (max(torch.round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
        bottomMost= (min(torch.round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));

        # //float maxval = 0;
        # //int maxidx = -1;
        # offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
        offset_bottom_data = im_data.view(-1)

        bin_cx = (leftMost + rightMost) / 2.0;     # // shift
        bin_cy = (topMost + bottomMost) / 2.0;

        bin_l = int(floor(bin_cx));
        bin_r = int(ceil(bin_cx));
        bin_t = int(floor(bin_cy));
        bin_b = int(ceil(bin_cy));

        lt_value = 0.0;
        if (bin_t > 0 and bin_l > 0 and bin_t < height and bin_l < width):
            lt_value = offset_bottom_data[bin_t * width + bin_l];
        rt_value = 0.0;
        if (bin_t > 0 and bin_r > 0 and bin_t < height and bin_r < width):
            rt_value = offset_bottom_data[bin_t * width + bin_r];
        lb_value = 0.0;
        if (bin_b > 0 and bin_l > 0 and bin_b < height and bin_l < width):
            lb_value = offset_bottom_data[bin_b * width + bin_l];
        rb_value = 0.0;
        if (bin_b > 0 and bin_r > 0 and bin_b < height and bin_r < width):
            rb_value = offset_bottom_data[bin_b * width + bin_r];

        rx = bin_cx - floor(bin_cx);
        ry = bin_cy - floor(bin_cy);

        wlt = (1.0 - rx) * (1.0 - ry);
        wrt = rx * (1.0 - ry);
        wrb = rx * ry;
        wlb = (1.0 - rx) * ry;

        inter_val = 0.0;

        inter_val += lt_value * wlt;
        inter_val += rt_value * wrt;
        inter_val += rb_value * wrb;
        inter_val += lb_value * wlb;

        output[i] = inter_val
    
    res = output.view(channels, pooled_height, pooled_width)

    if debug:
        x_d = res.data.cpu().numpy()
        x_data_draw = x_d.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)
      
        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        # x_data_draw = x_data_draw[:, :, ::-1]
        cv2.imshow('im_data_gt', x_data_draw)
        cv2.imwrite('res.jpg', x_data_draw)                 # 这个效果很正呀
        cv2.imshow('src_img', img)
        cv2.waitKey(100)
    temp = 1


    # pooled_feat = roipool(im_data, rois.view(-1, 6))

    # if debug:
    #     x_d = pooled_feat.data.cpu().numpy()[0]
    #     x_data_draw = x_d.swapaxes(0, 2)
    #     x_data_draw = x_data_draw.swapaxes(0, 1)
      
    #     x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
    #     # x_data_draw = x_data_draw[:, :, ::-1]
    #     cv2.imshow('im_data_gt', x_data_draw)
    #     cv2.imshow('src_img', img)
    #     cv2.waitKey(100)

    # print(pooled_feat.shape)