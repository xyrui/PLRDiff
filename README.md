# Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model (Information Fusion 2024)
<p align="center">
    Xiangyu Rui, <a href="https://github.com/xiangyongcao">Xiangyong Cao</a>, <a href="https://github.com/LiPang">Li Pang</a>, <a href="https://github.com/Zeyu-Zhu">Zeyu Zhu</a>, <a href="https://github.com/zsyOAOA">Zongsheng Yue</a>, <a href="https://gr.xjtu.edu.cn/web/dymeng">Deyu Meng</a>
</p>

<p align="center">

[paper (arxiv draft)](https://arxiv.org/pdf/2305.10925.pdf)

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

