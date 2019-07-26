import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image, ImageDraw
from model import EAST
from model2 import FOTS
# from efficient_model import FOTS
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
import cv2
import random
import math
from dataset import draw_box_points
from dataset import resize_32
from rroi_align.modules.rroi_align import _RRoiAlign
from util import strLabelConverter
from PIL import ImageFont


def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
		   res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.8, nms_thresh=0.5):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def plot_boxes_texts(img, boxes, texts, font2):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box,det_text in zip(boxes, texts):
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
		draw.text((box[6], box[7]), det_text, fill = (255,0,0),font=font2)
	return img


def ocr_branch(net, img, rroi, converter):
	# resized_img, ratio_h, ratio_w = resize_img(img)
	resized_img, rroi = resize_32(img, rroi)
	with torch.no_grad():
		# shared_feature = net.merge(net.extractor(load_pil(resized_img).to('cuda')))
		shared_feature = net.ocr_shared(load_pil(resized_img).to('cuda'))
	gts = rroi[:,:8]

	## for debug show
	# im = np.array(resized_img)
	# for box in gts:
	#     pts = box[0:8]
	#     pts = pts.reshape(4, -1)
	#     draw_box_points(im, pts, color=(0, 0, 255), thickness=1)
	# # cv2.imshow('img', im)
	# cv2.imwrite('./res%d.jpg'%0, im)

	if len(gts) != 0:
		gt = gts.reshape(-1, 4, 2)
		center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4
		dh = gt[:, 2, :] - gt[:, 1, :]; dh1 = gt[:, 3, :] - gt[:, 0, :]
		dw =  gt[:, 1, :] - gt[:, 0, :]; dw1 = gt[:, 2, :] - gt[:, 3, :]
		h = (np.sqrt(np.sum(dh**2,axis=1)) + np.sqrt(np.sum(dh1**2,axis=1))) / 2.0 
		w = (np.sqrt(np.sum(dw ** 2, axis=1)) + np.sqrt(np.sum(dw1 ** 2, axis=1))) / 2.0 

		angle_gt = (np.arctan2((gt[:, 1, 1] - gt[:, 0, 1]), gt[:, 1, 0] - gt[:, 0, 0]) + np.arctan2(
			(gt[:, 2, 1] - gt[:, 3, 1]), gt[:, 2, 0] - gt[:, 3, 0])) / 2  # 求角度
		angle_gt = -angle_gt / 3.1415926535 * 180  # 需要加个负号

	texts = []
	for gt_id in range(0, len(gts)):
		
		rois = [0, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle_gt[gt_id]]
		rois = [float(x) for x in rois]
		
		pooled_height = 8
		rois = torch.tensor(rois).to(torch.float).cuda()
		maxratio = rois[4] / rois[3]
		maxratio = maxratio.max().item()
		pooled_width = math.ceil(pooled_height * maxratio)
		roipool2 = _RRoiAlign(pooled_height, pooled_width, 1.0 / 4)  # 声明类
		pooled_feat = roipool2(shared_feature, rois.view(-1, 6))

		conv = net.cnn(pooled_feat)
		b, c, _h, _w = conv.size()
		assert _h == 1, "the height of conv must be 1"
		conv = conv.squeeze(2)                  # 64×512×26
		conv = conv.permute(2, 0, 1)  # [w, b, c] 26×64×512
		conv = net.dropout(conv)
		labels_pred = net.rnn(conv)

		labels_pred = labels_pred.permute(0,2,1)

		_, labels_pred = labels_pred.max(1)
		labels_pred = labels_pred.transpose(1, 0).contiguous().view(-1)
		preds_size = Variable(torch.IntTensor([labels_pred.size(0)]))
		sim_preds = converter.decode(labels_pred.data, preds_size.data, raw=False)

		texts.append(sim_preds)
	
	return texts


def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect(Image.open(img_file), model, device)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)


if __name__ == '__main__':
	test_img_path = './ICDAR_2015/ch4_test_images/'
	submit_path = './submit'
	model_path  = './pths/model_epoch_20.pth'
	# model_path = './pths/giou_east/model_epoch_1.pth'
	# model_path = './pths/east_vgg16.pth'

	with open('./ICDAR_2015/icdar2015_alphabet.txt', 'r') as f:
		alphabet = f.readlines()[0]
	converter = strLabelConverter(alphabet=alphabet)
	font2 = font2 = ImageFont.truetype("./src/Arial-Unicode-Regular.ttf", 18)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = FOTS(nclass=len(alphabet)+1).to(device)
	# model = EAST().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])

	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')		
		img = Image.open(img_file)
		
		boxes = detect(img, model, device)
		if boxes is not None:
			texts = ocr_branch(model, img, boxes.copy(), converter)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + ',%s\n'%text for box,text in zip(boxes,texts)])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)
		plot_img = plot_boxes_texts(img, boxes, texts, font2)
		res_img = './outputs/ocr/' + 'res_' + os.path.basename(img_file)
		plot_img.save(res_img)


