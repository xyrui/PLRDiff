# PLRDiff
Official codes of "**Unsupervised Pansharpening via Low-rank Diffusion Model**" 

[paper (arxiv)](https://arxiv.org/pdf/2305.10925.pdf)

## Load pretrained Model 
Pretrained diffusion model can be downloaded from

[https://github.com/wgcban/ddpm-cd#arrow_forwardpre-trained-models--trainvaltest-logs](https://github.com/wgcban/ddpm-cd#arrow_forwardpre-trained-models--trainvaltest-logs)

## Download Dataset

Chikusei: [https://naotoyokoya.com/Download.html](https://naotoyokoya.com/Download.html)

Houston: [https://hyperspectral.ee.uh.edu/?page id=459](https://hyperspectral.ee.uh.edu/?page_id=459)

Pavia: [https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

WV3: [https://github.com/liangjiandeng/PanCollection#1--the-training-and-testing-datasets-for-worldview-3](https://github.com/liangjiandeng/PanCollection#1--the-training-and-testing-datasets-for-worldview-3)

## Prepare test dataset
Use the ``generate_data.m" to generate test data.

## Testing
### Single HSI testing
run ``python3 test_single.py -gpu '(gpu)' -dr (dataroot) -dn (dataname) -rs (resume_state)``

gpu: int

dataroot: str, e.g. /path/Chikusei.mat

dataname: str, e.g. Chikusei

resume_state: str, e.g. /path/I190000_E97

### A list of HSIs testing
run ``python3 test_list.py -gpu '(gpu)' -dr (dataroot) -dn (dataname) -rs (resume_state)``

## Connections
<a href="mailto:xyrui.aca@gmail.com">xyrui.aca@gmail.com</a>

<head> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/all.js"></script> 
    <script defer src="https://use.fontawesome.com/releases/v5.0.13/js/v4-shims.js"></script> 
</head> 
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.0.13/css/all.css">

