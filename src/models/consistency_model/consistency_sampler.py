from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple, Union
import numpy as np

import torch
from torch import Tensor

from tqdm import tqdm

from .consistency_model import ConsistencyModel
from .helper import improved_timesteps_schedule, karras_schedule, pad_dims_like


@dataclass
class ConsistencySamplerConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    num_steps_denoising: int = 2
    # 是否展示采样进度条
    verbose: bool = False


class ConsistencySampler:
    def __init__(self, ConsistencyModel: ConsistencyModel, cfg: ConsistencySamplerConfig) -> None:
        self.ConsistencyModel = ConsistencyModel
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, ConsistencyModel.device)
        
    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tensor:
        torch.compiler.cudagraph_mark_step_begin()  # 标记步骤边界
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        # sigma = torch.full((x.shape[0],), self.sigmas[0], dtype=x.dtype, device=x.device)
        # x = self.ConsistencyModel.denoise(x, sigma, prev_obs, prev_act)
        # trajectory.append(x)
        # Progressively denoise the sample and skip the first step as it has already
        # been run
        ts = list(reversed(self.sigmas))
        pbar = tqdm(ts, disable=(not self.cfg.verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")
            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.cfg.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)
            x = self.ConsistencyModel.denoise(x, sigma, prev_obs, prev_act)

            trajectory.append(x)
        
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, device: torch.device) -> Tensor:
    # min_inv_rho = sigma_min ** (1 / rho)
    # max_inv_rho = sigma_max ** (1 / rho)
    # l = torch.linspace(0, 1, num_steps, device=device)
    # sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    # return torch.cat((sigmas, sigmas.new_zeros(1)))
    # 强制修改，使其返回 sigma_max
    if num_steps == 1:
        sigmas = [sigma_max]
    else:
        sigmas = np.linspace(sigma_min, sigma_max, num_steps, device = device)
    print(f'采样时sigmas: {list(reversed(sigmas))}')
    return sigmas
