from . import rsfac_gaussian_diffusion as gd

def create_model_and_diffusion_RS(opt):
    model = define_G(opt['model'])
    opt2 = opt['diffusion']
    diffusion = create_gaussian_diffusion(
        beta_schedule= opt2['beta_schedule'],
        beta_linear_start= opt2['beta_linear_start'],
        beta_linear_end = opt2['beta_linear_end'],
        steps=opt2['diffusion_steps'],
        accsteps=opt2['acce_steps']
    )
    return model, diffusion


def create_gaussian_diffusion(
    *,
    beta_schedule="linear",
    beta_linear_start=1e-6,
    beta_linear_end = 1e-2,
    steps=1000,
    accsteps=1000,
):
    betas = gd.make_beta_schedule(
            schedule=beta_schedule,
            n_timestep=steps,
            linear_start=beta_linear_start,
            linear_end=beta_linear_end
            )

    return gd.GaussianDiffusion(
        betas=betas,
        accsteps=accsteps
    )


#Modified from:
#https://github.com/wgcban/ddpm-cd/blob/b0213c0049bab215e470326d97499ae69416a843/model/networks.py#L82

####################
# define network
####################


# Generator
def define_G(model_opt):
    from .sr3_modules import unet
    if ('norm_groups' not in model_opt) or model_opt['norm_groups'] is None:
        model_opt['norm_groups']=32
    model = unet.UNet(
        in_channel=model_opt['in_channel'],
        out_channel=model_opt['out_channel'],
        norm_groups=model_opt['norm_groups'],
        inner_channel=model_opt['inner_channel'],
        channel_mults=model_opt['channel_multiplier'],
        attn_res=model_opt['attn_res'],
        res_blocks=model_opt['res_blocks'],
        dropout=model_opt['dropout'],
        image_size=256
    )
    return model
