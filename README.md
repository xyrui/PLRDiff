# PLRDiff
Official codes of "**Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model**" 

[paper (arxiv)](https://arxiv.org/pdf/2305.10925.pdf)

## Load pretrained Model 
Pretrained diffusion model can be downloaded from

[https://github.com/wgcban/ddpm-cd#arrow_forwardpre-trained-models--trainvaltest-logs](https://github.com/wgcban/ddpm-cd#arrow_forwardpre-trained-models--trainvaltest-logs)

## Download Dataset

Chikusei: [https://naotoyokoya.com/Download.html](https://naotoyokoya.com/Download.html)

Houston: [https://hyperspectral.ee.uh.edu/?page id=459](https://hyperspectral.ee.uh.edu/?page_id=459)

Pavia: [https://github.com/liangjiandeng/HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection)

## Prepare test dataset
Use data/generate_data.m to generate test data for Chikusei and Houston. Pavia can be directly downloaded for use. 

## Testing
### Single HSI testing
run ``python3 demo_syn.py -res opt``

there are several options you can set:

-gpu: int

-dn: dataname,str. e.g. 'Chikusei_test'

-krtype: int. Set 0 for the first time in order to estimate kernel and srf. Set 1 if you have already save them in './estKR'.

-res: str. Set 'opt' for estimating the residual and 'no' for R=0.

Other options include eta1, eta2, scale, ks, step, accstep. Please refer to demo_syn.py.

## Connections
<a href="mailto:xyrui.aca@gmail.com">xyrui.aca@gmail.com</a>

<head> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/all.js"></script> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/v4-shims.js"></script> 
</head> 
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.0.13/css/all.css">

