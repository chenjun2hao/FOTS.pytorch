import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from rroi_align.modules.rroi_align import _RRoiAlign
# from warpctc_pytorch import CTCLoss
from torch.autograd import Variable


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]


class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
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
	def __init__(self, pretrained=True, nclass=666, height=8):
		super(FOTS, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge     = merge()
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
			nIn = 128 if i == 0 else nm[i - 1]
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

	def ocr_shared(self, x):
		for i in range(14):
			m = self.extractor.features[i]
			x = m(x)
		return x

	def ocr_forward(self, shared_feature, rois, text, label_length):
		
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
