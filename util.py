
from dataset import draw_box_points
import torchvision.transforms as transforms
import torch
import collections
import math
import random
import numpy as np
import cv2
from rroi_align.modules.rroi_align import _RRoiAlign
from torch.autograd import Variable


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


def roi_process(im_data, text_polys, texts, debug_show=False):
    rrois = []
    labels = []
    for bid in range(len(im_data)):
        gts = text_polys[bid]
        lbs = texts[bid]

        # for debug show
        # x_d = transforms.ToPILImage()(im_data[bid])
        # im = np.array(x_d)
        # for box in gts:
        #     pts = box[0:8]
        #     pts = pts.reshape(4, -1)
        #     draw_box_points(im, pts, color=(0, 0, 255), thickness=1)
        # # cv2.imshow('img', im)
        # cv2.imwrite('./res%d.jpg'%bid, im)

        if len(gts) != 0:
            gt = gts.reshape(-1, 4, 2)
            center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4
            dh = gt[:, 2, :] - gt[:, 1, :]; dh1 = gt[:, 3, :] - gt[:, 0, :]
            dw =  gt[:, 1, :] - gt[:, 0, :]; dw1 = gt[:, 2, :] - gt[:, 3, :]
            h = (np.sqrt(np.sum(dh**2,axis=1)) + np.sqrt(np.sum(dh1**2,axis=1))) / 2.0 + random.random()*2 - 1
            w = (np.sqrt(np.sum(dw ** 2, axis=1)) + np.sqrt(np.sum(dw1 ** 2, axis=1))) / 2.0 + random.random()*2 - 1

            angle_gt = (np.arctan2((gt[:, 1, 1] - gt[:, 0, 1]), gt[:, 1, 0] - gt[:, 0, 0]) + np.arctan2(
                (gt[:, 2, 1] - gt[:, 3, 1]), gt[:, 2, 0] - gt[:, 3, 0])) / 2  # 求角度
            angle_gt = -angle_gt / 3.1415926535 * 180  # 需要加个负号
            
        for gt_id in range(0, len(gts)):
                per_text = lbs[gt_id]
                per_gt = gt[gt_id]
                if per_gt[:, 0].max() > im_data[bid].shape[2] or per_gt[:, 1].max() > im_data[bid].shape[1] or per_gt.min() < 0:
                    continue
                if '###' in per_text:
                    continue
                if h[gt_id] > 2 * w[gt_id] and len(per_text) > 2:
                    continue

                rrois.append([bid, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle_gt[gt_id]])
                labels.append(per_text)

    # for debug show
    debug_show = 1
    if debug_show:
        rois = torch.tensor(rrois).to(torch.float).cuda()
        pooled_height = 44
        maxratio = rois[:,4] / rois[:,3]
        maxratio = maxratio.max().item()
        pooled_width = math.ceil(pooled_height * maxratio)

        roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)        # 声明类
        im_data = im_data.cuda()
        pooled_feat = roipool(im_data, rois.view(-1, 6))

        for i in range(pooled_feat.shape[0]):
            x_d = pooled_feat.data.cpu().numpy()[i]
            x_data_draw = x_d.swapaxes(0, 2)
            x_data_draw = x_data_draw.swapaxes(0, 1)
            # x_data_draw += 1
            # x_data_draw *= 128
            x_data_draw *= 255
            x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
            x_data_draw = x_data_draw[:, :, ::-1]
            # cv2.imshow('crop %d' % i, x_data_draw)
            cv2.imwrite('./outputs/tshow/crop%d.jpg' % i, x_data_draw)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)

    return rrois, labels


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