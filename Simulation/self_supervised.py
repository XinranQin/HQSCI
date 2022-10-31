import torch
import torch.utils.data as tud
import argparse
import numpy as np
from Simulation.Dataset_simulation import dataset
from torch.autograd import Variable
import time
from Simulation.network import Network
from torch import optim
import scipy.io as scio
import torch.nn as nn
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument('--data_path', default='./Data/Testing_data/', type=str,help='path of data')
parser.add_argument('--mask_path', default='./Data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=256, type=int, help='the size of trainset image')
parser.add_argument("--train_num", default=100000, type=int, help='total number of iterations')
parser.add_argument("--test_num", default=10, type=int, help='total number of inference')
parser.add_argument("--sample_num", default=1, type=int, help='total number of inference')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
opt = parser.parse_args()
print(opt)

def ForwardA(x,mask):
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)
    temp = mask * x
    temp_shift = torch.zeros((B,H, W + (28 - 1) * 2, 28)).cuda()
    temp_shift[:,:,0:opt.size, :] = temp
    for t in range(28):
        temp_shift[ :,:,:, t] = torch.roll(temp_shift[:,:,:,t], 2 * t, dims=2)
    meas = torch.sum(temp_shift, axis=3)
    y = meas / 28 * 2
    return y.unsqueeze(1)

def prepare_data_test(path, file_num):
    HR_HSI = np.zeros((((256,256,28,file_num))))
    for idx in range(file_num):
        path1 = os.path.join(path)  + 'scene%02d.mat' % (idx+1)
        data = scio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['img']
    HR_HSI[HR_HSI < 0.] = 0.
    HR_HSI[HR_HSI > 1.] = 1.
    return HR_HSI

HR_HSI = prepare_data_test(opt.data_path,  10)

dataset = dataset(opt, HR_HSI)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)
criterion = nn.MSELoss()



inference_num = opt.test_num
Iteration = opt.train_num
psnr_total = 0
k = 0
for f, (input, label,mask) in enumerate(loader_train):

    model = Network(3,28).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    for i in range(Iteration):
        input, label = Variable(input), Variable(label)
        input, label,mask3D_torch = input.cuda(), label.cuda(),mask.cuda()
        mask3D = mask3D_torch.permute(0,3,1,2)
        b,c,w,h = input.shape
        mask = mask.cuda()
        start = time.time()
        out = model(input,mask3D)
        loss = criterion(ForwardA(out,mask),input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result = out
        result = result.clamp(min=0.,max=1.)

        if i % 1000 == 0:
            savepath = "./Checkpoint/self-supervised/" + str(f) + '/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            torch.save(model, os.path.join(savepath, 'single_model_%03d.pth' % (i)))
            with torch.no_grad():
                for num in range(inference_num-1):
                    out = model(input,mask3D)
                    result = result + out
            result = (result/10)
            scio.savemat(savepath+str(f),{"result":result.detach().cpu().numpy()})
            res = result.cpu().permute(2,3,1,0).squeeze(3).detach().numpy()
            save_path = './Results/' + str(f + 1) + '.mat'
            scio.savemat(save_path, {'res':res})



