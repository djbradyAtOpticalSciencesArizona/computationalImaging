from math import gamma
import os
import sys
import time
import glob
import numpy as np
import torch

import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as io
from torch.autograd import Variable
from model_Unet import UNet
from utilts import torch_psnr,init_params
import imageio

Nl=64
Nx=145
# io.savemat('input.mat',{'H':H})
data = io.loadmat('Indian_pines.mat')['indian_pines']


x = data[100,:,100:(100+Nl)]
x = x/x.max()
ny, nx = x.shape
inputs = io.loadmat('./Unet2D/input.mat')
H = inputs['H']
y = inputs['y']
n = np.random.normal(0, 0., (ny, nx)) # noise


#%%
### collect training and validation data
seed = 19
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)

all_data = np.delete(data, 100, 0)[:,:,100:(100+Nl)]
all_data = all_data/all_data.max()

print(all_data.shape)
data_train = torch.zeros((104,145,64))
data_train[0:80,:,:] = torch.from_numpy(all_data[0:80,:,:])
data_train[80:,:,:] = torch.from_numpy(all_data[120:,:,:])
print(data_train.shape)
data_val = torch.from_numpy(all_data[95:105,:,:])
print(data_val.shape)

y_train = torch.zeros((data_train.shape[0],H.shape[0]))
for i in range(data_train.shape[0]):
    n_i = torch.from_numpy(np.random.normal(0, 0., (ny, nx)))
    y_train[i,:] = torch.matmul(torch.from_numpy(H),(data_train[i,:,:].flatten()+n.flatten()))

y_val = torch.zeros((data_val.shape[0],H.shape[0]))
for i in range(data_val.shape[0]):
    n_i = torch.from_numpy(np.random.normal(0, 0., (ny, nx)))
    y_val[i,:] = torch.matmul(torch.from_numpy(H),(data_val[i,:,:].flatten()+n.flatten()))
print(y_train.shape)

#%%
criterion = nn.MSELoss(size_average=False)
criterion = criterion.cuda()
model = UNet(1, 1)
model = model.cuda()
max_epoch = 50
optimizer = torch.optim.Adam(model.parameters(), lr= 4e-4, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.97, last_epoch=-1) # CosineAnnealingLR(optimizer, float(max_epoch), eta_min=1e-5)
epoch_sam_num = 5000
batch_size = 32
batch_num = int(np.floor(epoch_sam_num/batch_size))


#%%

def gen_HTy(x,H=None):
    '''To generate H.T*y as input'''
    HT = torch.transpose(torch.from_numpy(H),0,1)
    y = torch.zeros((x.shape[0],H.shape[0]))
    for i in range(x.shape[0]):
        n_i = torch.from_numpy(np.random.normal(0, 0., (ny, nx)))
        y[i,:] = torch.matmul(torch.from_numpy(H),(x[i,:,:].flatten()+n_i.flatten()))

    HTy = torch.zeros((y.shape[0],HT.shape[0]))
    for i in range(y.shape[0]):
        HTy[i,:] = torch.matmul(HT.double(),y[i,:].double())
    return HTy

def random_choose(train_data, batch_size):
    '''To random choose data'''
    patch_size = 145 
    index = np.random.choice(range(train_data.shape[0]), batch_size)
    processed_data = np.zeros((batch_size, patch_size,train_data.shape[2]), dtype=np.float32)
    
    for i in range(batch_size):     
        processed_data[i, :,:] = train_data[index[i],:,:]
    gt_batch = torch.from_numpy(processed_data)
    return gt_batch

def train(train_set, model, criterion, optimizer, lr, epoch,batch_num):

    gt = train_set
    for i in range(batch_num):
        gt_batch = random_choose(train_set, batch_size)
        yHT = gen_HTy(gt_batch,H).float().cuda()
        gt = Variable(gt_batch).cuda().float()
        optimizer.zero_grad()
        logits = model(yHT)
        loss = criterion(logits, gt)/(gt.size()[0]*2) # 1 145 64

        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()        
        total_loss = loss.item()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i , batch_num,
                100. * i , total_loss))
    return total_loss

def infer(dataset_val, model,epoch=0,H=None):
    model.eval()
    psnr_val = 0
    test_gt = dataset_val
    yHT = gen_HTy(test_gt,H).float().cuda()
    test_gt = test_gt.cuda()

    model_out = model(yHT)
    
    for k in range(len(dataset_val)):
        psnr_val += torch_psnr(model_out[k,:,:], test_gt[k,:,:])
    print("[epoch %d] PSNR_val_mean: %.4f" % (epoch, psnr_val/len(dataset_val)))

    
    return psnr_val/len(dataset_val),model_out


#%%
if __name__=='__main__':
    psnr_max = 0

    for epoch in range(max_epoch):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_loss = train(data_train, model, criterion, optimizer, lr, epoch,batch_num)
        logging.info('train_loss %f', train_loss)

        # validation
        psnr_mean,model_out = infer(data_val, model,epoch,H)
        
        logging.info('valid_acc %f \n', psnr_mean)

        psnr_mean = psnr_mean.cpu().detach().numpy()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 27:
                torch.save(model.state_dict(), os.path.join('./Unet2D/', ('net'+str(epoch)+'.pth')))
                # result = model_out[0,:,:].cpu().detach().numpy()
                # io.savemat('./Unet2D/result_Unet.mat',{'result_Unet':result})
                # imageio.imwrite('./Unet2D/result.png',result)

        