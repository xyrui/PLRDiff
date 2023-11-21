import torch as torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as nF
import h5py as h5py
import scipy.io as sio
import torch.optim as optim
from core1 import imresize
import os as os

def blur_kernel(shape, var):
    assert shape%2==1
    mu = int((shape - 1)/2) 
    XX, YY = torch.meshgrid(torch.arange(shape), torch.arange(shape), indexing='ij')
    out = torch.exp((-(XX.to(var.device)-mu)**2-(YY.to(var.device)-mu)**2)/(2*var**2))
    return out/out.sum()

class Para(nn.Module):
    def __init__(self, ks, B):
        super(Para, self).__init__()
        self.kernel = Parameter(torch.ones((ks*ks), dtype=torch.float32))
        self.R = Parameter(torch.ones((B), dtype=torch.float32))
        self.ks = ks

    def forward(self,):
        return nF.softmax(self.kernel, dim=0).reshape(self.ks,self.ks),  nF.softmax(self.R, dim=0)
    
class estKR:
    def __init__(self, ms, pan, ks):
        self.device = torch.device("cuda")
        # load data

        self.pan = pan.to(self.device)
        self.ms = ms.to(self.device)
        _,C,_,_ = self.ms.shape

        self.ks = ks
        self.C = C

    def start_est(self, scale_factor = None, ite=None, degrade=None, lr=None):
        para = Para(self.ks, self.C).cuda()

        para.train()
        optimizer = optim.SGD(para.parameters(), lr=lr)

        for ii in range(ite):
           # kernel, R = para() # linear
            kernel, R = para()

            kernel = kernel.unsqueeze(0).unsqueeze(0)
            R = R.unsqueeze(0).unsqueeze(0).unsqueeze(-1)   # linear
            
            if degrade == 0:
                pd = int((self.ks - 1)/2) 
                blurpan = nF.interpolate(nF.conv2d(nF.pad(self.pan, [pd, pd, pd, pd], mode='circular'), kernel, padding=0, groups=1), scale_factor=scale_factor, mode='nearest')
            elif degrade == 1:
                blurpan = imresize(nF.conv2d(self.pan, weight=kernel, padding=int((self.ks - 1)/2), groups=1), scale=scale_factor)
            elif degrade == 2:
                pd = int((self.ks - 1)/2) 
                blurpan = imresize(nF.conv2d(nF.pad(self.pan, [pd, pd, pd, pd], mode='circular'), kernel, padding=0, groups=1), scale=scale_factor)
            elif degrade == 3:
                pd = int((self.ks - 1)/2) 
                blurpan = nF.interpolate(nF.conv2d(self.pan, weight=kernel, padding=int((self.ks - 1)/2), groups=1), scale_factor=scale_factor, mode='nearest')

            pms = torch.matmul(self.ms.permute(0,2,3,1), R).permute(0,3,1,2) # linear

            optimizer.zero_grad()
            loss = nF.mse_loss(blurpan, pms)
            loss.backward()
            optimizer.step()
            
            if ii%500 == 0:
                print(f"Iter: [{ii+1:4d}] -- Loss: {loss.item():.3e}")
        
        print(f"Iter: [{ii+1:4d}] -- Loss: {loss.item():.3e}")
        R = R.detach().cpu()
        kernel = kernel.detach().cpu().squeeze(0).squeeze(0)
        return kernel, R

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device("cuda")
    # load data
    dn = ""
    ms = ""
    pan = ""
    Est = estKR(ms, pan, ks=11) 
    kernel, R = Est.start_est(scale_factor = 1/4, ite=15000, degrade=0, lr=500) 
    print(R.sum())
    sio.savemat("./KR_"+dn+".mat", {'kernel':kernel.numpy(), 'R':R.numpy()})
    
    
