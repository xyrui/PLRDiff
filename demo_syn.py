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

from math import sqrt, log, log10
from torch.utils.data import DataLoader
import torch.utils.data as uData
import time

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
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")    
    parser.add_argument('-sr', '--savedir', type=str, default='./results')   # where to save the restored images
    parser.add_argument('-eta1', '--eta1', type=float, default=2)            # parameter eta_1
    parser.add_argument('-eta2', '--eta2', type=float, default=2)            # parameter eta_2
    parser.add_argument('-rank', '--rank', type=int, default=3)              # subspace dimension; low rank parameter s
    parser.add_argument('-seed', '--seed', type=int, default=0)          
    parser.add_argument('-dn', '--dataname', type=str, default="Chikusei")   
    parser.add_argument('-step', '--step', type=int, default=500)            # Original total sampling step (divisible by accstep)
    parser.add_argument('-accstep', '--accstep', type=int, default=500)      # Actual sampling step (less than step)
    parser.add_argument('-krtype', '--krtype', type=int, default=1)          # how to get the kernel and srf: '0' for estimate, '1' for download
    parser.add_argument('-sn', '--samplenum', type=int, default=1)  
    parser.add_argument('-scale', '--scale', type=int, default=4)            # downsampling scale
    parser.add_argument('-ks', '--ks', type=int, default=11)                 # kernel size
    parser.add_argument('-res', '--res', type=str, default="no")             # how to set residual: 'no' for no residual, 'opt' for estimating residual
    parser.add_argument('-sample_method', '--sample_method', type=str, default='ddpm')

    parser.add_argument('-rs', '--resume_state', type=str, default='/blabla/I190000_E97')  # where you put the loaded diffusion model 

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

    ## Load diffusion model
    load_path = opt['resume_state']
    gen_path = '{}_gen.pth'.format(load_path)
    cks = th.load(gen_path)
    new_cks = OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model.load_state_dict(new_cks, strict=False)
    for param in model.parameters():
        param.requires_grad=False
    model.to(device)
    model.eval()
 
    ## params
    param = dict()
    param['scale'] = opt['scale'] # downsampling scale
    param['eta1'] = opt['eta1']   # parameter eta_1
    param['eta2'] = opt['eta2']   # parameter eta_2

    param['k_s'] = opt['ks']      # kernel size

    ## load img
    dataroot = join('./data', dname+'.mat')
    data = sio.loadmat(dataroot)
    HRHS = th.from_numpy(np.float32(data['HRMS']))
    ms, Ch = HRHS.shape[0], HRHS.shape[-1]
    HRHS = HRHS.permute(2,0,1).unsqueeze(0) # [1, Ch, ms, ms]
    
    Rr = opt['rank']  # spectral dimensironality of subspace

    # select bands
    inters = int((Ch+1)/(Rr+1)) # interval
    selected_bands = [(t+1)*inters-1 for t in range(Rr)]
    param['Band'] = th.Tensor(selected_bands).type(th.int).to(device)  

    PAN = th.from_numpy(np.float32(data['PAN'])).unsqueeze(0).unsqueeze(0) # [1,1,ms,ms]
    
    LRHS = th.from_numpy(np.float32(data['LRMS'])).permute(2,0,1).unsqueeze(0) # [1, Ch, ms/scale, ms/scale]

    model_condition = {'LRHS': LRHS.to(device), 'PAN': PAN.to(device)}

    out_path = Path(opt['savedir'])
    out_path.mkdir(parents=True, exist_ok=True)

    ## Get Kernel and srf
    if opt['krtype'] == 0: # estimate kr by optimization
        from guided_diffusion.estKR import estKR
        Estkr = estKR(LRHS, PAN, param['k_s'])
        kernel, PH = Estkr.start_est()
        
        # save the kernel and srf so that next time you can use opt['krtype']==1 to directly load them
        sio.savemat("./estKR/KR_"+dname+".mat", {'kernel':kernel.numpy(), 'R':PH.squeeze(0).squeeze(0).numpy()})
        
        kernel = kernel.repeat(Ch,1,1,1).to(device)
        PH = PH.to(device)
    elif opt['krtype'] == 1: # load kr from somewhere
        kr = sio.loadmat("./estKR/KR_"+dname+".mat")
        kernel = th.from_numpy(kr['kernel']).repeat(Ch,1,1,1).to(device)
        PH = th.from_numpy(kr['R']).unsqueeze(0).unsqueeze(0).to(device)
    
    param['kernel'] = kernel.to(device) # kernel
    param['PH'] = PH.to(device)         # srf
    
    start_time = time.time()
    # sample: base tensor A     E: coefficient matrix E      add_res: R
    sample,E,add_res = diffusion.sample_loop(
        model,
        (1, Ch, ms, ms),
        Rr = Rr,
        noise = None,
        clip_denoised=True,
        model_condition=model_condition,
        param=param,
        save_root=out_path,
        sample_method=args.sample_method,
        res = args.res  # opt, itp
    )
    
    sample = (sample + 1)/2 # base tensor A: rescale from range [-1,1] to [0,1]
    ## im_out is the final restored HS image
    im_out = th.matmul(E, sample.reshape(1, Rr, -1)).reshape(1, Ch, ms, ms) + add_res # Ax_3 E + R
    
    Ours_time = time.time() - start_time
    im_out = im_out.cpu().squeeze(0).permute(1,2,0).numpy() # [ms, ms, Ch]

    A = sample.cpu().squeeze(0).permute(1,2,0).numpy()  # base tensor A: to numpy
    E = E.cpu().squeeze(0).numpy()  # coefficient matrix E: to numpy
    
    nf = np.max(HRHS.squeeze(0).permute(1,2,0).numpy(), axis=(0,1), keepdims=True)
    psnr = my_psnr(HRHS.squeeze(0).permute(1,2,0).numpy()/nf, im_out/nf)
    
    ssim = SSIM(HRHS.squeeze(0).permute(1,2,0).numpy()/nf, im_out/nf, data_range=1)
    
    ## save output
    sio.savemat(join(out_path, dname+"_"+args.res+"_Ours.mat"), {'R_Ours': im_out,'E':E, 'A':A})

    print(f"{dname:s} \t PSNR: \t {psnr:.2f} \t SSIM: \t {ssim:.4f} \t Time: {Ours_time:.2f}\n") 


