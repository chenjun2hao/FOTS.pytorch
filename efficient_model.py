from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn
# from warpctc_pytorch import CTCLoss
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import math


class efficient(nn.Module):
	def __init__(self, model_name):
		super(efficient, self).__init__()
		self.model = EfficientNet.from_pretrained(model_name)
		self.model_name = model_name
		self.choose_index = {'efficientnet-b4': [31, 21, 9, 5], 'efficientnet-b0': [2, 4, 10, 15],
							 'efficientnet-b2': [4, 7, 15, 22]}

	def forward(self, inputs):
		
		index = self.choose_index[self.model_name]
		out = []
		x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

		# Blocks
		for idx, block in enumerate(self.model._blocks):
			drop_connect_rate = self.model._global_params.drop_connect_rate
			if drop_connect_rate:
				drop_connect_rate *= float(idx) / len(self.model._blocks)
			x = block(x, drop_connect_rate=drop_connect_rate)
			if idx in index:
				out.append(x)

		# # for debug show
		# for index, t in enumerate(out):
		# 	print(index, t.shape)
		## Head
		# x = relu_fn(self.model._bn1(self.model._conv_head(x)))

		return out


class merge(nn.Module):
	def __init__(self, model_name):
		super(merge, self).__init__()

		self.choose_channel = {'efficientnet-b4': [608, 184, 96], 'efficientnet-b0': [432, 168, 88],
							   'efficientnet-b2': [472, 176, 88]}
		channels = self.choose_channel[model_name]

		self.conv1 = nn.Conv2d(channels[0], 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(channels[1], 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(channels[2], 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		y = self.relu7(self.bn7(self.conv7(y)))
		return y

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, model_name, pretrained=True):
		super(EAST, self).__init__()
		self.extractor = efficient(model_name=model_name)
		self.merge     = merge(model_name=model_name)
		self.output    = output()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

		
class FOTS(nn.Module):
	def __init__(self, model_name, nclass=666, height=8):
		super(FOTS, self).__init__()
		self.extractor = efficient(model_name=model_name)
		self.merge     = merge(model_name=model_name)
		self.output    = output()
		self.pooled_height = height

		## 测试ocr识别网络
		nh = 256          # LSTM隐含层的节点数
		ks = [3,  3,  3,   3,   3,   3, 2]
		ps = [1,  1,  1,   1,   1,   1, 0]
		ss = [1,  1,  1,   1,   1,   1, 1]
		nm = [64, 64, 128, 128, 256, 256, 256]

		cnn = nn.Sequential()

		def convRelu(i, batchNormalization=False, relu=True):
			nIn = 32 if i == 0 else nm[i - 1]
			nOut = nm[i]
			cnn.add_module('conv{0}'.format(i),
						nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
			if batchNormalization:
				cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
			if not relu:
				cnn.add_module('relu{0}'.format(i),
								nn.LeakyReLU(0.2, inplace=True))
			else:
				cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

		convRelu(0, True)
		convRelu(1, True)
		cnn.add_module('pooling{0}'.format(0),
						nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4xseq
		convRelu(2, True)
		convRelu(3, True)
		cnn.add_module('pooling{0}'.format(1),
						nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x2xseq
		convRelu(4, True)
		convRelu(5, True)
		# cnn.add_module('pooling{0}'.format(2),
		#                 nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
		convRelu(6, True)  # 512x1xseq

		self.cnn = cnn
		self.rnn = nn.Sequential(
			# BidirectionalLSTM(256, nh, nh),
			BidirectionalLSTM(nh, nh, nclass))
		self.dropout = nn.Dropout2d(0.2)
		# self.ocr_crition = CTCLoss()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))

	def ocr_forward(self, x, rois, text, label_length):
		shared_feature = self.merge(self.extractor(x))

		rois = torch.tensor(rois).to(torch.float).cuda()
		maxratio = rois[:, 4] / rois[:, 3]
		maxratio = maxratio.max().item()
		pooled_width = math.ceil(self.pooled_height * maxratio)
		roipool2 = _RRoiAlign(self.pooled_height, pooled_width, 1.0 / 4)  # 声明类
		pooled_feat = roipool2(shared_feature, rois.view(-1, 6))

		conv = self.cnn(pooled_feat)
		b, c, h, w = conv.size()
		assert h == 1, "the height of conv must be 1"
		conv = conv.squeeze(2)                  # 64×512×26
		conv = conv.permute(2, 0, 1)  # [w, b, c] 26×64×512
		conv = self.dropout(conv)
		preds = self.rnn(conv)

		if self.training:
			preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))  # 求ctc loss
			loss_ocr = self.ocr_crition(preds, text, preds_size, label_length) / preds.size(1)
			return loss_ocr
		else:
			pass


if __name__ == '__main__':
	m = FOTS()
	x = torch.randn(1, 32, 8, 8)
	out = m.ocr_forward(x)
	print(out.shape)


if __name__ == '__main__':
	m = EAST('efficientnet-b2')
	x = torch.randn(1, 3, 256, 256)
	score, geo = m(x)
	print(score.shape)
	print(geo.shape)


# if __name__=='__main__':
# 	model = efficient('efficientnet-b2')
# 	x = torch.randn(1,3,512,512)
# 	y = model(x)
# 	print(y.shape)

