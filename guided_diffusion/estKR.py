import torch as torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as nF
import h5py as h5py
import scipy.io as sio
import torch.optim as optim
from .core import imresize

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

    def start_est(self, scale_factor = 1/4, ite=15000):
        para = Para(self.ks, self.C).cuda()
        scale = int(1/scale_factor)

        para.train()
        optimizer = optim.SGD(para.parameters(), lr=500)
        for ii in range(ite):
            kernel, R = para()

            kernel = kernel.unsqueeze(0).unsqueeze(0)
            R = R.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            pd = int((self.ks - 1)/2) 
            blurpan = nF.conv2d(self.pan, kernel, None, (1, 1), (pd, pd), groups=1)
            blurpan = blurpan[:, :, ::scale , :: scale]

            pms = torch.matmul(self.ms.permute(0,2,3,1), R).permute(0,3,1,2)

            optimizer.zero_grad()
            loss = nF.mse_loss(blurpan, pms)
            loss.backward()
            optimizer.step()
            
            if ii%500 == 0:
                print(f"Iter: [{ii+1:4d}] -- Loss: {loss.item():.3e}")
        
        print(f"Iter: [{ii+1:4d}] -- Loss: {loss.item():.3e}")
        R = R.detach().cpu()
        print(R.sum())
        kernel = kernel.detach().cpu().squeeze(0).squeeze(0)
        return kernel, R
