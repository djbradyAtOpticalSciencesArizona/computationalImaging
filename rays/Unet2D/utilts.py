import scipy.io as sio
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import logging
from ssim_torch import ssim
import torch.nn as nn

def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

def torch_psnr(img, ref):      #input [28,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:] - ref[i,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):   #input [28,256,256]
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))

def init_params(net, init_type='kn'):
    print('use init scheme: %s' %init_type)
    if init_type != 'edsr':
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                if init_type == 'kn':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if init_type == 'ku':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if init_type == 'xn':
                    nn.init.xavier_normal_(m.weight)
                if init_type == 'xu':
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
#            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)):
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


