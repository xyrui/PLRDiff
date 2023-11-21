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
from math import sqrt, log, log10
from torch.utils.data import DataLoader
import torch.utils.data as uData

def my_psnr(X,Y):
    ch = X.shape[-1]
    psnr = 0
    for i in range(ch):
        psnr = psnr + 10*log10(1/np.mean(np.power(X[:,:,i] - Y[:,:,i], 2)))
    return psnr/ch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="2")
    parser.add_argument('-sr', '--savedir', type=str, default='./results/')
    parser.add_argument('-eta1', '--eta1', type=float, default=2)
    parser.add_argument('-eta2', '--eta2', type=float, default=2)
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-dn', '--dataname', type=str, default="Chikusei")
    parser.add_argument('-step', '--step', type=int, default=500)
    parser.add_argument('-accstep', '--accstep', type=int, default=500)
    parser.add_argument('-krtype', '--krtype', type=int, default=1)  
    parser.add_argument('-sn', '--samplenum', type=int, default=1)
    parser.add_argument('-sample_method', '--sample_method', type=str, default='ddpm')

    parser.add_argument('-rs', '--resume_state', type=str, default='/resume_state/I190000_E97')

    ## parse configs
    args = parser.parse_args()

    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)
    opt['diffusion']['diffusion_steps'] = args.step
    opt['diffusion']['acce_steps'] = args.accstep
    
    device = th.device("cuda")
    dname = opt['dataname']
    
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
    param['scale'] = 4
    param['eta1'] = opt['eta1']
    param['eta2'] = opt['eta2']

    if ("Chikusei" in dname) or ("Houston" in dname):
        param['degrade'] = 1
        param['k_s'] = 9
    else:
        param['degrade'] = 0
        param['k_s'] = 11

    ## img
    dataroot = join('./data', dname+'.mat')
    data = sio.loadmat(dataroot)
    HRMS = th.from_numpy(np.float32(data['HRMS']))
    ms, Ch = HRMS.shape[0], HRMS.shape[-1]
    HRMS = HRMS.permute(2,0,1).unsqueeze(0)

    ## select bands
    inters = int((Ch+1)/4)
    selected_bands = [inters-1, 2*inters-1, 3*inters-1]
    param['Band'] = th.Tensor(selected_bands).type(th.int).to(device)  

    PAN = th.from_numpy(np.float32(data['PAN'])).unsqueeze(0).unsqueeze(0)
    
    LRMS = th.from_numpy(np.float32(data['LRMS'])).permute(2,0,1).unsqueeze(0)

    model_condition = {'LRMS': LRMS.to(device), 'PAN': PAN.to(device)}

    out_path = Path(opt['savedir'])
    out_path.mkdir(parents=True, exist_ok=True)

    Rr = 3  # spectral dimensironality of subspace

    if opt['krtype'] == 0: # estimate kr by optimization
        from guided_diffusion.estKR import estKR
        Estkr = estKR(LRMS.to(device), PAN.to(device), param['k_s'])
        kernel, PH = Estkr.start_est(degrade=param['degrade'])

        sio.savemat("./estKR/KR_"+dname+".mat", {'kernel':kernel.numpy(), 'R':PH.squeeze(0).squeeze(0).numpy()})  # save the KR for the first time. Next time you may choose krtype == 1

        kernel = kernel.repeat(Ch,1,1,1).to(device)
        PH = PH.to(device)
    elif opt['krtype'] == 1: # load kr from somewhere
        kr = sio.loadmat("./estKR/KR_"+dname+".mat")
        kernel = th.from_numpy(kr['kernel']).repeat(Ch,1,1,1).to(device)
        PH = th.from_numpy(kr['R']).unsqueeze(0).unsqueeze(0).to(device)
    
    param['kernel'] = kernel.to(device)
    param['PH'] = PH.to(device)

    for j in range(opt['samplenum']):
        sample,E = diffusion.sample_loop(
            model,
            (1, Ch, ms, ms),
            Rr = Rr,
            noise = None,
            clip_denoised=True,
            model_condition=model_condition,
            param=param,
            save_root=out_path,
            sample_method=args.sample_method,
            start_t = None
        )
        
        sample = (sample + 1)/2
        im_out = th.matmul(E, sample.reshape(1, Rr, -1)).reshape(1, Ch, ms, ms)
        im_out = im_out.cpu().squeeze(0).permute(1,2,0).numpy() 
        HRMS = HRMS.squeeze(0).permute(1,2,0).numpy()
        LRMS = LRMS.squeeze(0).permute(1,2,0).numpy() 
        PAN = PAN.squeeze(0).permute(1,2,0).numpy() 
        A = sample.cpu().squeeze(0).permute(1,2,0).numpy()
        E = E.cpu().squeeze(0).numpy()
        
        nf = np.max(HRMS, axis=(0,1), keepdims=True)
        psnr = my_psnr(HRMS/nf, im_out/nf)
        
        ssim = SSIM(HRMS/nf, im_out/nf, data_range=1)
        
        # save output
        sio.savemat(join(out_path, dname+"_Ours.mat"), {'R_Ours': im_out,'E':E, 'A':A})

        print(f"{dname:s} \t PSNR: \t {psnr:.2f} \t SSIM: \t {ssim:.4f} \n") 

