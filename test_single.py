import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as nF
from pathlib import Path

from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
import scipy.io as sio
from collections import OrderedDict
from os.path import join
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from guided_diffusion.core import imresize
from math import sqrt, log

def blur_kernel(shape, var):
    assert shape%2==1
    mu = int((shape - 1)/2) 
    XX, YY = np.meshgrid(np.arange(shape), np.arange(shape))
    out = np.exp((-(XX-mu)**2-(YY-mu)**2)/(2*var**2))
    return np.float32(out/out.sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-dr', '--dataroot', type=str, default='') # dataroot with 
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-sr', '--savedir', type=str, default='./results')
    parser.add_argument('-eta1', '--eta1', type=float, default=1)  # trade off parameters 1
    parser.add_argument('-eta2', '--eta2', type=float, default=2)  # trade off parameters 2
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-dn', '--dataname', type=str, default='') # dataname: used for save output
    parser.add_argument('-step', '--step', type=int, default=1000) # diffusion steps
    parser.add_argument('-scale', '--scale', type=int, default=4)  # downsample scale
    parser.add_argument('-ks', '--kernelsize', type=int, default=9) # kernel size
    parser.add_argument('-sig', '--sig', type=float, default=None) # kernel variance
    parser.add_argument('-sn', '--samplenum', type=int, default=1) # sample number 
    parser.add_argument('-rs', '--resume_state', type=str, default='')  # path: pretrained model

    ## parse configs
    args = parser.parse_args()

    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)
    opt['diffusion']['diffusion_steps'] = opt['step']
    
    device = th.device("cuda")
    
    ## create model and diffusion process
    model, diffusion = create_model_and_diffusion_RS(opt)
    
    ## seed
    seeed = opt['seed']
    print(seeed)
    np.random.seed(seeed)
    th.manual_seed(seeed)
    th.cuda.manual_seed(seeed)

    ## load model
    load_path = opt['resume_state']
    gen_path = '{}_gen.pth'.format(load_path)
    cks = th.load(gen_path)
    new_cks = OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model.load_state_dict(new_cks, strict=False)
    model.to(device)
    model.eval()
 
    ## params
    param = dict()
    param['scale'] = opt['scale']
    param['eta1'] = opt['eta1']
    param['eta2'] = opt['eta2']

    k_s = opt['kernelsize']
    if opt['sig'] is None:
        sig = sqrt(4**2/(8*log(2)))
    else:
        sig = opt['sig']

    ## img
    data = sio.loadmat(opt['dataroot'])
    HRMS = th.from_numpy(np.float32(data['HRMS']))
    ms, Ch = HRMS.shape[0], HRMS.shape[-1]
    HRMS = HRMS.permute(2,0,1).unsqueeze(0)

    param['k_s'] = k_s
    kernel = blur_kernel(k_s, sig)
    kernel = th.from_numpy(kernel).repeat(Ch,1,1,1)
    param['kernel'] = kernel.to(device)


    if opt['dataname'] == 'Chikusei':
        param['Band'] = th.Tensor([31,63,95]).type(th.int).to(device)   
    elif opt['dataname'] == 'Houston':
        param['Band'] = th.Tensor([35,71,107]).type(th.int).to(device)   
    elif opt['dataname'] == 'Pavia':
        param['Band'] = th.Tensor([25,51,77]).type(th.int).to(device)  


    PH = th.from_numpy(np.float32(data['H'])).unsqueeze(0).unsqueeze(-1)
    param['PH'] = PH.to(device)
    PAN = th.from_numpy(np.float32(data['PAN'])).unsqueeze(0).unsqueeze(0)

    LRMS = nF.conv2d(HRMS, kernel.to(HRMS.device), padding=int((k_s - 1)/2), groups=Ch)
    LRMS = imresize(LRMS, 1/opt['scale'])

    model_condition = {'LRMS': LRMS.to(device), 'PAN': PAN.to(device)}

    out_path = Path(join(opt['savedir'], str(opt['scale'])+"_"+str(opt['kernelsize'])+"_"+str(sig)))
    out_path.mkdir(parents=True, exist_ok=True)

    Rr = 3  # spectral dimensironality of subspace

    dname = opt['dataname']
    for j in range(opt['samplenum']):
        sample,E = diffusion.p_sample_loop(
            model,
            (1, Ch, ms, ms),
            Rr = Rr,
            clip_denoised=True,
            model_condition=model_condition,
            param=param,
            save_root=out_path,
            progress=True,
        )
        
        sample = (sample + 1)/2
        im_out = th.matmul(E, sample.reshape(opt['batch_size'], Rr, -1)).reshape(opt['batch_size'], Ch, ms, ms)
        im_out = im_out.cpu().squeeze(0).permute(1,2,0).numpy() 
        HRMS = HRMS.squeeze(0).permute(1,2,0).numpy()
        LRMS = LRMS.squeeze(0).permute(1,2,0).numpy() 
        PAN = PAN.squeeze(0).permute(1,2,0).numpy() 
        A = sample.cpu().squeeze(0).permute(1,2,0).numpy()
        E = E.cpu().squeeze(0).numpy()
        
        psnr = 0
        for i in range(Ch):
            psnr += PSNR(HRMS, im_out)
        psnr /= Ch
        
        ssim = SSIM(HRMS, im_out)
        
        ## save output
        sio.savemat(join(out_path, opt['dataname']+str(opt['step'])+".mat"), {'Rec': im_out, 'HRMS': HRMS, 'LRMS': LRMS, 'PAN':PAN, 'E':E, 'A':A})

        print(f"{dname:s} \t PSNR: \t {psnr:.2f} \t SSIM: \t {ssim:.4f} \n") 

