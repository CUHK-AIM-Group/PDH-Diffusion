from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
# from gluonts.torch.distributions.distribution_output import DistributionOutput
from typing import Tuple
import copy


"""
diffusion models
"""


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

#beta_end 和diffusion step是list形式
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        input_size,
        # control the diffusion and sampling(reverse diffusion) procedures
        share_ratio_list,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale = None
        self.share_ratio_list = share_ratio_list  # ratio of betas are shared

        betas_list=[]
        # print('beta_end',beta_end)
        # print(len(beta_end))

        if isinstance(beta_end, list):
            for i in range(len(beta_end)):
                if beta_schedule == "linear":
                    # betas = np.linspace(1e-4, beta_end[i], diff_steps[i])
                    betas = np.linspace(1e-4, beta_end[i], diff_steps[i])
                    betas_list.append(betas)
                elif beta_schedule == "quad":
                    betas = np.linspace(1e-4 ** 0.5, beta_end[i] **
                                        0.5, diff_steps[i]) ** 2
                    betas_list.append(betas)
                elif beta_schedule == "const":
                    betas = beta_end[i] * np.ones(diff_steps[i])
                    betas_list.append(betas)
                elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                    betas = 1.0 / np.linspace(diff_steps[i], 1, diff_steps[i])
                    betas_list.append(betas)
                elif beta_schedule == "sigmoid":
                    betas = np.linspace(-6, 6, diff_steps[i])
                    betas = (beta_end[i] - 1e-4) / (np.exp(-betas) + 1)
                    betas_list.append(betas)
                elif beta_schedule == "cosine":
                    betas = cosine_beta_schedule(diff_steps[i])
                    betas_list.append(betas)
                else:
                    raise NotImplementedError(beta_schedule)
                # print(betas_list)
                    

            to_torch = partial(torch.tensor, dtype=torch.float32)
            (timesteps,) = betas.shape
            self.num_timesteps = int(timesteps)
            self.loss_type = loss_type
            self.register_buffer("betas", to_torch(betas_list[0]))

        else:

            if betas is not None:
                betas = (
                    betas.detach().cpu().numpy()
                    if isinstance(betas, torch.Tensor)
                    else betas
                )
            else:
                if beta_schedule == "linear":
                    betas = np.linspace(1e-4, beta_end, diff_steps)
                elif beta_schedule == "quad":
                    betas = np.linspace(1e-4 ** 0.5, beta_end **
                                        0.5, diff_steps) ** 2
                elif beta_schedule == "const":
                    betas = beta_end * np.ones(diff_steps)
                elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                    betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
                elif beta_schedule == "sigmoid":
                    betas = np.linspace(-6, 6, diff_steps)
                    betas = (beta_end - 1e-4) / (np.exp(-betas) + 1)
                elif beta_schedule == "cosine":
                    betas = cosine_beta_schedule(diff_steps)
                else:
                    raise NotImplementedError(beta_schedule)
            # beta_schedule.eta=0

            to_torch = partial(torch.tensor, dtype=torch.float32)
            (timesteps,) = betas.shape
            self.num_timesteps = int(timesteps)
            self.loss_type = loss_type
            self.register_buffer("betas", to_torch(betas))
        # print('len(betas_list)',len(betas_list))
        for i , cur_share_ratio in enumerate(self.share_ratio_list):
            if isinstance(beta_end, list):
                start_index = int(len(betas_list[i])*(1-cur_share_ratio)) #这好像只能控制加噪的步数
                betas_sub = copy.deepcopy(betas_list[i])
            else:
                start_index = int(len(betas)*(1-cur_share_ratio)) #这好像只能控制加噪的步数
                betas_sub = copy.deepcopy(betas)
            betas_sub[:start_index] = 0  # share the latter part of the betas
            alphas_sub = 1.0 - betas_sub
            alphas_cumprod_sub = np.cumprod(alphas_sub, axis=0)
            alphas_cumprod_prev_sub = np.append(1.0, alphas_cumprod_sub[:-1])
            suffix = int(cur_share_ratio * 100)
            self.register_buffer(
                f"alphas_cumprod_{suffix}", to_torch(alphas_cumprod_sub))
            self.register_buffer(
                f"alphas_cumprod_prev_{suffix}", to_torch(alphas_cumprod_prev_sub))

            self.register_buffer(f"sqrt_alphas_cumprod_{suffix}", to_torch(
                np.sqrt(alphas_cumprod_sub)))
            self.register_buffer(
                f"sqrt_one_minus_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 - alphas_cumprod_sub))
            )
            self.register_buffer(
                f"log_one_minus_alphas_cumprod_{suffix}", to_torch(
                    np.log(1.0 - alphas_cumprod_sub))
            )
            self.register_buffer(
                f"sqrt_recip_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub))
            )
            self.register_buffer(
                f"sqrt_recipm1_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub - 1))
            )
            self.register_buffer(

                f"posterior_mean_coef1_{suffix}",
                to_torch(betas_sub * np.sqrt(alphas_cumprod_prev_sub) / (1.0 - alphas_cumprod_sub)),)

            self.register_buffer(
                f"posterior_mean_coef2_{suffix}",

                to_torch(
                    (1.0 - alphas_cumprod_prev_sub) *
                    np.sqrt(alphas_sub) / (1.0 - alphas_cumprod_sub)

                ),)
            posterior_variance_sub = (
                betas_sub * (1.0 - alphas_cumprod_prev_sub) / (1.0 - alphas_cumprod_sub))
            self.register_buffer(
                f"posterior_variance_{suffix}", to_torch(posterior_variance_sub))

            self.register_buffer(
                f"posterior_log_variance_clipped_{suffix}",
                to_torch(np.log(np.maximum(posterior_variance_sub, 1e-20))),)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t, share_ratio: float):
        # get q(x_t|x_0) distribution foward process
        # q(x_t|x_0)=N(sqrt_alphas_cumprod*x0, (1-alphas_cumprod)I)
        suffix = int(share_ratio * 100)

        mean = extract(
            getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape) * x_start
        variance = extract(
            1.0 - getattr(self, f'alphas_cumprod_{suffix}'), t, x_start.shape)
        log_variance = extract(
            getattr(self, f'log_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape)

        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, share_ratio: float):
        # x_0=1/sqrt(alphas_cumprod)*x_t - \sqrt{1/alphas_cumprod -1 }* eps
        suffix = int(share_ratio * 100)
        return (
            extract(
                getattr(self, f'sqrt_recip_alphas_cumprod_{suffix}'), t, x_t.shape) * x_t
            - extract(getattr(self, f'sqrt_recipm1_alphas_cumprod_{suffix}'), t, x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t, share_ratio: float):
        suffix = int(share_ratio * 100)

        posterior_mean = (
            extract(
                getattr(self, f'posterior_mean_coef1_{suffix}'), t, x_t.shape) * x_start
            + extract(getattr(self, f'posterior_mean_coef2_{suffix}'), t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            getattr(self, f'posterior_variance_{suffix}'), t, x_t.shape)
        posterior_log_variance_clipped = extract(
            getattr(
                self, f'posterior_log_variance_clipped_{suffix}'), t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool, share_ratio: float):

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond), share_ratio=share_ratio,
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # changed

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t,  share_ratio=share_ratio,
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, share_ratio: float, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised, share_ratio=share_ratio,
        )

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * \
            (0.5 * model_log_variance).exp() * noise

        return sample

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, share_ratio: float):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        inter_steps = int(self.num_timesteps*(1-share_ratio))
        for i in reversed(range(inter_steps, self.num_timesteps)):
            img = self.p_sample(
                x=img, cond=cond, t=torch.full(
                    (b,), i, device=device, dtype=torch.long),
                share_ratio=share_ratio,
            )

        return img

    @torch.no_grad()
    def sample(self, share_ratio: float, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape

        x_hat = self.p_sample_loop(
            shape=shape, cond=cond, share_ratio=share_ratio)
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, share_ratio: float, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        suffix = int(share_ratio * 100)
        return (
            extract(
                getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape) * x_start
            + extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, share_ratio: float, noise=None):
        # if share betas, means only part of the betas are used.
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t = a x0 + b \eps
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, share_ratio=share_ratio)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)
        # raise ValueError("Here terminated.")
        #输入是含噪声的image，预测的目标是noise，需要告知时间步数t


        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss
    
    #原版log_prob
    # def log_prob(self, x, cond, share_ratio: float, *args, **kwargs):
    #     B, T, _ = x.shape
    #     # print('xshape',x.shape) #torch.Size([64, 40, 116])

    #     time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()

    #     # raise ValueError("Here terminated.")
    #     loss = self.p_losses(
    #         x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
    #         *args, **kwargs)

    #     return loss



    #将不同层级指导作为condition引入的log_prob
    
    # def log_prob(self, x, x_coarser_grained, cond, share_ratio: float, noise=None, *args, **kwargs):
    #     B, T, _ = x.shape
    #     # print('xshape',x.shape) #torch.Size([64, 40, 116])

    #     time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
    #     # print('time',time.shape)
    #     if x_coarser_grained!=None:
    #         noise = default(noise, lambda: torch.randn_like(x_coarser_grained.reshape(B * T, 1, -1)))
    #         time_forward=time.clone()
    #         condition = (time_forward > 10) & (time_forward < 200)
    #         time_forward[condition]=time[condition]-10
    #         time_forward[~condition]=0
    #         # print('time_forward',time_forward.shape)
    #         x_noisy_coarser_grained=self.q_sample(x_start=x_coarser_grained.reshape(B * T, 1, -1), t=time_forward, noise=noise, share_ratio=share_ratio)
    #         # print(x_noisy_coarser_grained.shape) #torch.Size([2560, 1, 116])
    #         x_noisy_coarser_grained_bt=x_noisy_coarser_grained.reshape(B, T, -1)
    #         # print(cond.shape)

    #         coarser_grained_cond=proj_diffusion_args

    #     raise ValueError("Here terminated.")
    #     cond=cond+x_noisy_coarser_grained

    #     loss = self.p_losses(
    #         x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
    #         *args, **kwargs)

    #     return loss

    def log_prob_pre(self, x, x_coarser_grained, share_ratio: float, noise=None, *args, **kwargs):
        B, T, _ = x.shape
        # print('xshape',x.shape) #torch.Size([64, 40, 116])

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        # print('time',time.shape)

        noise = default(noise, lambda: torch.randn_like(x_coarser_grained.reshape(B * T, 1, -1)))
        time_forward=time.clone()
        condition = (time_forward > 10) & (time_forward < 200)
        time_forward[condition]=time[condition]-10
        time_forward[~condition]=0
        # print('time_forward',time_forward.shape)
        x_noisy_coarser_grained=self.q_sample(x_start=x_coarser_grained.reshape(B * T, 1, -1), t=time_forward, noise=noise, share_ratio=share_ratio)
        # print(x_noisy_coarser_grained.shape) #torch.Size([2560, 1, 116])
        x_noisy_coarser_grained_bt=x_noisy_coarser_grained.reshape(B, T, -1)
        # print(cond.shape)

        # raise ValueError("Here terminated.")
        return x,x_noisy_coarser_grained_bt,time
    
    def log_prob(self, x, cond, time, share_ratio: float, noise=None, *args, **kwargs):
        B, T, _ = x.shape
        # print('xshape',x.shape) #torch.Size([64, 40, 116])
        # print('cond',cond.shape) #torch.Size([64, 40, 116])
        # print('share_ratio',share_ratio) #torch.Size([64, 40, 116])


        # raise ValueError("Here terminated.")

        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
            *args, **kwargs)

        return loss
    
    def log_prob_full(self, x, cond, share_ratio: float, *args, **kwargs):
        B, T, _ = x.shape
        # print('xshape',x.shape) #torch.Size([64, 40, 116])

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()

        # raise ValueError("Here terminated.")
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
            *args, **kwargs)

        return loss


"""
diffusion output  
"""


class DiffusionOutput(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)


# 分形分布预测

# class Fractal_distribution_MLP(nn.Module):
#     def __init__(self, input_size,output_size1=2,output_size2=40,series_num=116,hidden_size1=128, hidden_size2=64):
#         super(Fractal_distribution_MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, output_size2*2)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(output_size2*2, output_size2)
#         self.output_size1=output_size1
#         self.series_num=series_num
    
#     def forward(self, x): #x shape: [batch_size,length,num_cells(4*num_series)]
#         print('x',x.shape)
#         # x=x.view(x.size(0),-1)

#         out = self.fc1(x)

#         out = self.relu1(out)
#         out = self.fc2(out)

#         out = self.relu2(out)
#         out = self.fc3(out)

#         print('out',out.shape)

#         out = out.view(x.size(0),self.series_num, self.output_size1, -1)
#         print('out',out.shape)
#         return out
    

class Fractal_distribution_MLP(nn.Module):
    def __init__(self, input_size=82*4,output_size1=2,output_size2=40,series_num=82,length_channel=100):
        super(Fractal_distribution_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size*length_channel, input_size*length_channel)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(input_size*length_channel)
        self.fc2 = nn.Linear(input_size*length_channel, 2*series_num*output_size2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(2*series_num*output_size2)
        self.fc3 = nn.Linear(2*series_num*output_size2,2*series_num*output_size2)
        self.bn3 = nn.BatchNorm1d(2*series_num*output_size2)

        self.output_size1=output_size1
        self.series_num=series_num
        self.output_size2=output_size2
        self.dropout = nn.Dropout(0.2) #0.5 0.8 0.2
    
    def forward(self, x): #x shape: [batch_size,length,num_cells(4*num_series)]
        # print('x',x.shape)
        x=x.reshape(x.size(0),-1)

        out = self.fc1(x)
        out = self.bn1(out)

        # out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)

        # out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.bn3(out)

        # print('out',out.shape)

        out = out.view(x.size(0),self.series_num, self.output_size1, self.output_size2)
        # print('out',out.shape)
        return out