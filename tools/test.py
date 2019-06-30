import cv2
import matplotlib.pyplot as plt

path = './data/tshow/crop0.jpg'
im = cv2.imread(path)
plt.imshow(im)
plt.show()
# temp = 1

# import torch
# from torch.autograd import Variable
# import numpy as np

# def csum(input):
#     # res = input.sum()
#     npinput = input.detach().numpy()
#     res = np.sum(npinput)
#     res = torch.FloatTensor([res])
#     return res

# data = torch.randn((5, 5), requires_grad=True)

# res = csum(data)
# res.backward()
# print(data.grad)


# import torch
# HW = 7
# N = 2
# x = torch.rand(N,3,HW,HW)

# # 求解最大值位置：
# temp = torch.mean(x, dim=1).view(N, HW*HW)
# points = torch.argmax(temp,dim=1)
# points

# # 将最大值位置转成坐标：
# x_p = points / HW
# print(x_p)
# y_p = torch.fmod(points,HW)
# print(y_p)


# # 联合坐标
# z_p = torch.cat((y_p.view(2,1),x_p.view(2,1)),dim=1).float() # 注意在F.grid_sample中我们计算的y_p才是x轴

# # 对坐标缩至-1，1之间：
# z_p = ((z_p+1)-(HW+1)/2)/((HW-1)/2)
# grid = z_p.unsqueeze(1).unsqueeze(1)


# # 生成通用裁剪区域：此处生成大小3*3
# step = 2/(HW-1)
# BOX_LEFT = 1
# BOX = 2*BOX_LEFT+1
# # torch.Size([Box, Box, 1])
# direct = torch.linspace(-(BOX_LEFT)*step,(BOX_LEFT)*step,BOX).unsqueeze(0).repeat(BOX,1).unsqueeze(-1)
# direct_trans = direct.transpose(1,0)
# full = torch.cat([direct,direct_trans],dim=2).unsqueeze(0).repeat(N,1,1,1)


# # 将通用区域和最大值坐标对应起来，注意grid_sample要求flow field在-1到1之间：
# full[:,:,:,0] = torch.clamp(full[:,:,:,0] + grid[:,:,:,0],-1,1)
# full[:,:,:,1] = torch.clamp(full[:,:,:,1] + grid[:,:,:,1],-1,1)


# # 将通用区域和最大值坐标对应起来，注意grid_sample要求flow field在-1到1之间：
# full[:,:,:,0] = torch.clamp(full[:,:,:,0] + grid[:,:,:,0],-1,1)
# full[:,:,:,1] = torch.clamp(full[:,:,:,1] + grid[:,:,:,1],-1,1)
# full


# # 裁剪feature map
# torch.nn.functional.grid_sample(x,full)

