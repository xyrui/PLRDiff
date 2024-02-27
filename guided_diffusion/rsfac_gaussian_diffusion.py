
"""
This code started out as a PyTorch port of the following:
https://github.com/HJ-harry/MCG_diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

The conditions are changed and coefficient matrix estimation is added.
"""

import enum
import math

import numpy as np
import torch as th
from torch.autograd import grad
import torch.nn.functional as nF
from functools import partial
import torch.nn.parameter as Para
import torch.optim as optim

from os.path import join as join
from torchvision.transforms import Compose
from tqdm.auto import tqdm

import scipy.io as sio

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            th.arange(n_timestep + 1, dtype=th.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    elif schedule == "exp":
        k = 10
        alphas_cumprod = np.exp(-k * np.arange(n_timestep+1)/n_timestep)
        alphas_cumprod = np.flip(1-alphas_cumprod)
        alphas_cumprod = (alphas_cumprod - alphas_cumprod.min())/(alphas_cumprod.max() - alphas_cumprod.min())*(1-1e-3)+1e-3
        alphas = alphas_cumprod[1:]/alphas_cumprod[:-1]
        betas = 1-alphas
    else:
        raise NotImplementedError(schedule)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class Param(th.nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.E = Para.Parameter(data=data)
    
    def forward(self,):
        return self.E

class Sgdopt(th.nn.Module): 
    def __init__(self, init):
        super(Sgdopt, self).__init__()
        self.X = th.nn.parameter.Parameter(init)

    def forward(self,):
        return self.X
    
class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    """

    def __init__(
        self,
        *,
        betas,
        accsteps
    ):

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        # accelerate
        assert self.num_timesteps%accsteps == 0
        self.alphas_cumprod = self.alphas_cumprod[::int(self.num_timesteps/accsteps)]
        assert len(self.alphas_cumprod) == accsteps
        self.num_timesteps = accsteps
        alphas = np.append(self.alphas_cumprod[0], self.alphas_cumprod[1:]/self.alphas_cumprod[:-1])
        betas = 1.0 - alphas
        
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) 
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., self.alphas_cumprod)) # 比timesteps多了一位
        self.sqrt_one_minus_alphas_cumprod_prev = np.sqrt(1.0 - self.alphas_cumprod_prev) 
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (    # DDPM原文第(7)式这个tilde(beta)
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # easy beta
        # self.posterior_variance = (
        #     betas
        # )
        
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None
    ):
        B = x.shape[0]
        noise_level = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t.item())+1]]).repeat(B, 1).to(x.device)   
   
        Ch,ms = x.shape[1],x.shape[-1] 

        model_output = []
        for b in range(Ch//3):
            model_output.append(model(x[:,b*3:b*3+3,:,:], noise_level))
        if Ch%3 != 0:
            model_output.append(model(x[:,-3:,:,:], noise_level)[:, -(Ch%3):, :,:])
        model_output = th.cat(model_output, dim=1)

        # model_output = model(x, noise_level) 

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        pred_xstart = process_xstart(   
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
        model_mean, _, posterior_log_variance = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {
            "mean": model_mean,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def sample_loop(
        self,
        model,
        shape,
        Rr, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_condition=None,
        param=None,
        save_root=None,
        sample_method= None,
        res = None,
    ):
        finalX = None
        finalE = None
        

        for (sample, E, add_res) in self.sample_loop_progressive(
            model,
            shape,
            Rr,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            param=param,
            save_root=save_root,
            sample_method=sample_method,
            res = res
        ):
            finalX = sample
            finalE = E
            add_res = add_res
             
        return finalX["sample"], finalE, add_res

    def sample_loop_progressive(self,
        model,
        shape,
        Rr, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_condition=None,
        param=None,
        save_root=None,
        sample_method = None,
        res = None
        ):
        Bb, Cc, Hh, Ww = shape
        Rr = Rr
        
        device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        Eband = param['Band']

        if noise is not None:
            img = noise.clone()
            del noise
        else:
            img = th.randn((Bb, Rr, Hh, Ww), device=device) # size [B, r, H, W]

        indices = list(range(self.num_timesteps))[::-1] 
        indices = tqdm(indices)
        
        LRHS = model_condition["LRHS"]
        PAN = model_condition["PAN"]

        blur = partial(nF.conv2d, weight=param['kernel'], padding=int((param['k_s'] - 1)/2), groups=Cc) 
        down = lambda x : x[:,:,::param['scale'], ::param['scale']]

        ## estimate coefficient matrix E (on cpu)
        bimg = th.index_select(LRHS, 1, Eband).reshape(Bb, Rr, -1).cpu() # base tensor from LRHS
        # estimate coefficient matrix E by solving least square problem
        t1 = th.matmul(bimg, bimg.transpose(1,2)) + 1e-4*th.eye(Rr).type(bimg.dtype)
        t2 = th.matmul(LRHS.reshape(Bb, Cc, -1).cpu(), bimg.transpose(1,2))
        E = th.matmul(t2, th.inverse(t1)).to(device)  

        LRHS_l = th.matmul(E, bimg.to(device)).reshape(*LRHS.shape) # Y_l: low rank component of LRHS
        res_l = LRHS - LRHS_l # D(B(R)): downsampled blured residual part
        del bimg, t1, t2

        # Set residual part R
        if res == "opt":  # optimize ||D(B(R)) - res_l||_F^2 to get R
            print('Estimate residual by optimization!')
            SGDopt = Sgdopt(nF.interpolate(res_l, [Hh, Ww], mode='bicubic')).cuda()
            optimizer = optim.SGD(SGDopt.parameters(), lr=3)
            for i in range(100):
                add_res = SGDopt()
                loss = th.sum(th.pow(down(blur(add_res)) - res_l.to(device), 2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            add_res = SGDopt().detach()
        elif res == "itp": # interpolate res_l to get R
            print('Estimate residual by interpolatation!')
            add_res = nF.interpolate(res_l, [Hh, Ww], mode='bicubic').to(device)
        else: # R=0
            print('No residual !')
            add_res = 0

        for i in indices:
            t = th.tensor(i * shape[0], device=device)

            # re-instantiate requires_grad for backpropagation
            img = img.requires_grad_()

            # Algorithm 1 line 2: sample A_{t-1} from p(A_{t-1}|A_t)
            if sample_method == 'ddpm':
                out = self.ddpm_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn
                )
        
            baseA = (out["pred_xstart"] +1)/2  # base tensor A = \hat{A}_0 (img ranges in [-1,1]. here we move out["pred_xstart"] back to range [0,1])

            xhat_1 = th.matmul(E, baseA.reshape(Bb, Rr, -1)).reshape(*shape) + add_res  # \hat{X}_0 = Ax_3 E + R

            xhat_2 = blur(xhat_1) 
            xhat_3 = down(xhat_2) # D(B(\hat{X}))
            norm1 = th.norm(LRHS - xhat_3) # ||Y - D(B(Ax_3 E + R))||

            xhat_4 = th.matmul((xhat_1).permute(0,2,3,1), param["PH"]).permute(0,3,1,2) # \hat{X}_3 r
            norm2 = th.norm(PAN - xhat_4) # ||P - \hat{X}_3 r||

            # Algorithm 1 line 4
            likelihood = norm1 + (param['eta2']/param['eta1'])*norm2
            norm_gradX = grad(outputs=likelihood, inputs=img)[0] 

            # Algorithm 1 line 5
            out["sample"] = out["sample"] - param['eta1']*norm_gradX # 
            
            del norm_gradX, baseA
            
            yield out, E, add_res
            img = out["sample"]
            # Clears out small amount of gpu memory. If not used, memory usage will accumulate and OOM will occur.
            img.detach_()

    def ddpm_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        noise = th.randn_like(x)
        nonzero_param = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_param * th.exp(0.5 * out["log_variance"]) * noise 
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
            

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
