clear
clc

ks = 9;
ratio = 4;
sig= sqrt(ratio^2/(8*log(2)));

load('Chikusei_o.mat') 

A = double(A); 

HRMS = (A - min(A(:)))/(max(A(:)) - min(A(:))); % normalize to[0,1]
C = size(A,3);

Kernel     = fspecial('gaussian', [ks, ks], sig); % blur kernel
LRMS  = imfilter(HRMS, Kernel);  % blurring
LRMS = imresize(LRMS, 1/ratio, "bicubic"); % downsampling  % The imresize operator in the 17th line in test_single.py is the same with this operator.


% PAN = mean(HRMS(:,:,1:100), 3); % Pavia
% PAN = mean(HRMS(:,:,21:87), 3); % Houston (Houston: discard the first 20 bands)
PAN = mean(HRMS(:,:,16:81), 3); % Chikusei (Chikusei: discard the first 15 bands)

save('Chikusei.mat','HRMS','LRMS','PAN');
