from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from .helper import improved_timesteps_schedule, karras_schedule, add_dims, pad_dims_like
# from utils import LossAndLogs



def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps



def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss."""
    return 1 / (sigmas[1:] - sigmas[:-1] + 1e-4)  # 避免极端值


@dataclass
class Conditioners:
    """Conditioning parameters for the Consistency Model."""
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


# @dataclass
# class SigmaDistributionConfig:
#     """Configuration for the noise distribution."""
#     sigma_min: float = 0.002,
#     sigma_max: float = 80.0,
#     rho: float = 7.0,


@dataclass
class ConsistencyModelConfig:
    """Configuration for the Consistency Model."""
    inner_model: InnerModelConfig

    # total_training_steps: int = 10_000
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    sigma_offset_noise: float = 0.3,
    rho: float = 7.0,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    lognormal_mean: float = -1.1,
    lognormal_std: float = 2.0,


@dataclass
class ConsistencyTrainingOutput:
    """Type of the output of the (Improved)ConsistencyTraining.__call__ method.

    Attributes
    ----------
    predicted : Tensor
        Predicted values.
    target : Tensor
        Target values.
    num_timesteps : int
        Number of timesteps at the current point in training from the timestep discretization schedule.
    sigmas : Tensor
        Standard deviations of the noise.存储噪声的标准差
    loss_weights : Optional[Tensor], default=None
        Weighting for the Improved Consistency Training loss.
    """

    next_outputs: Tensor
    current_outputs: Tensor
    target_outputs: Tensor
    num_timesteps: int
    sigmas: Tensor
    loss_weights: Optional[Tensor] = None

class ConsistencyModel(nn.Module):
    """Consistency Model for World Model in RL."""

    def __init__(self, cfg: ConsistencyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        """Returns the device of the inner model."""
        return self.inner_model.noise_emb.weight.device

    def setup_training(self) -> None:
        """Sets up the noise sampling function for training."""
        # assert self.sample_sigma_training is None, "Training setup already initialized."

        # def sample_sigma(n: int, device: torch.device) -> Tensor:
        #     """Samples noise levels from a log-normal distribution."""
        #     s = torch.randn(n, device=device) * cfg.scale + cfg.loc
        #     return s.exp().clip(cfg.sigma_min, cfg.sigma_max)
        print("consistency model is setting up")

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        """Applies noise to the input tensor."""
        b, c, _, _ = x.shape
        # offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        # return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)
        return x + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        """Computes the conditioning parameters for the model."""
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        sigma_data = self.cfg.sigma_data
        c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
        # 修改
        c_skip = sigma_data**2 / ((sigma - self.cfg.sigma_min)**2 + sigma_data**2)
        # (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5
        c_out = (sigma_data * (sigma - self.cfg.sigma_min )) / (sigma_data**2 + sigma**2) ** 0.5
        # c_noise = sigma.log() / 4
        c_noise = sigma
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1))))

    def compute_model_output(self, noisy_x: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        """Computes the model output for the given noisy input and conditioning."""
        rescaled_obs = obs / self.cfg.sigma_data
        # rescaled_noise = noisy_x * cs.c_in
        return self.inner_model(noisy_x, cs.c_noise, rescaled_obs, act)

    @torch.no_grad()
    def wrap_model_output(self, noisy_x: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        """Wraps the model output with skip connections and scaling."""
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        # d = d.clone().clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        
        return cs.c_skip * noisy_x + cs.c_out * model_output

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        # denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised       

    # current_training_step：当前的训练步骤，请注意使用⚠️
    def forward(self, sequence: torch.FloatTensor, actions:torch.FloatTensor, current_training_step: int, total_training_steps: int) -> ConsistencyTrainingOutput:
        """Forward pass for the Consistency Model."""
        n = self.cfg.inner_model.num_steps_conditioning
        # sequence.shape = (B, T, 64, 8, 8)
        seq_length = sequence.size(1) - n

        all_obs = sequence.clone()
        all_ations = actions.clone()

        # 用列表来收集结果
        next_outputs = []
        current_outputs = []
        target_outputs = []
        num_timesteps_list = []
        sigmas_list = []
        loss_weights_list = []

        for i in range(seq_length):
            # obs.shape is torch.Size([64, 4, 68, 8, 8])
            obs = all_obs[:, i : n + i]
            # next_obs.shape is torch.Size([64, 68, 8, 8])
            next_obs = all_obs[:, n + i]
            act = all_ations[:, i : n + i]
            # mask = batch.mask_padding[:, n + i]

            b, t, c, h, w = obs.shape
            # print(f'##############################world model ')
            # print(f'###############obs.shape is {obs.shape}')
            # print(f'###############actions.shape is {act.shape}')
            # print(f'###############next_obs.shape is {next_obs.shape}')
            # print(f'################sequence.shape is {sequence.shape}')
            # print(f'#############current_training_steps is {current_training_step} and total_trainingsteps is {total_training_steps}')
            obs = obs.reshape(b, t * c, h, w)

            # 获取 training_steps_based的离散去噪时间步
            num_timesteps = improved_timesteps_schedule(
                current_training_step,
                total_training_steps,
                self.cfg.initial_timesteps,
                self.cfg.final_timesteps,
            )

            # 生成 batch 个 sigmas
            sigmas = karras_schedule(
                num_timesteps, self.cfg.sigma_min, self.cfg.sigma_max, self.cfg.rho, self.device
            )

            # 随机选择时间步，返回 batch_size 个随机时间步
            timesteps = lognormal_timestep_distribution(
                b, sigmas, self.cfg.lognormal_mean, self.cfg.lognormal_std
            )
            # 获取当前和下一个 sigma
            current_sigmas = sigmas[timesteps]
            next_sigmas = sigmas[timesteps + 1]
            
            # Sample noise levels
            noisy_next_obs = self.apply_noise(next_obs, next_sigmas, self.cfg.sigma_offset_noise)

            # Compute conditioners
            current_cs = self.compute_conditioners(current_sigmas)
            next_cs = self.compute_conditioners(next_sigmas)

            # Forward pass
            # model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
            next_model_output = self.compute_model_output(noisy_next_obs, obs, act, next_cs) 
            next_output = self.wrap_model_output(noisy_next_obs, next_model_output, next_cs)

            # Consistency loss: compare model output at different noise levels
            with torch.no_grad():
                current_noisy_obs = self.apply_noise(next_obs, current_sigmas, self.cfg.sigma_offset_noise)
                current_model_output = self.compute_model_output(current_noisy_obs, obs, act, current_cs)
                current_output = self.wrap_model_output(current_noisy_obs, current_model_output, current_cs)

            loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_output)

            # 用预测得到的denoised_x更新序列
            denoised = next_output
            all_obs[:, n + i] = denoised

            # next_obs进行缩放得到target
            target_output = (next_obs - next_cs.c_skip * noisy_next_obs) / next_cs.c_out

            # 统计数据
            next_outputs.append(next_output)
            current_outputs.append(current_output)
            target_outputs.append(target_output)
            num_timesteps_list.append(num_timesteps)
            sigmas_list.append(sigmas)
            loss_weights_list.append(loss_weights)

            # # 改进loss
            # # consistency_loss = F.mse_loss(self.wrap_model_output(noisy_next_obs, model_output, cs)[mask], current_output[mask]))
            # # next_output = self.wrap_model_output(noisy_next_obs, model_output, cs)
            # # weight
            # loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_output)
            # consistency_loss = (pseudo_huber_loss(next_output, current_output) * loss_weights).mean()
            # prediction_loss = F.mse_loss(model_output[mask], target[mask])
            # scaling_factor = (consistency_loss.detach() / prediction_loss.detach()).clamp(min=0.1, max=10)
            # self.cfg.lambda_prediction = 1.0 * scaling_factor
            # # Total loss: weighted sum of consistency and prediction losses
            # cl_loss += self.cfg.lambda_consistency * consistency_loss
            # pre_loss += self.cfg.lambda_prediction * prediction_loss
            # loss += (self.cfg.lambda_consistency * consistency_loss + self.cfg.lambda_prediction * prediction_loss)
    

        # 堆叠列表，使其变成张量
        next_outputs = torch.stack(next_outputs, dim=1)  # shape: (B, seq_length, ...)
        current_outputs = torch.stack(current_outputs, dim=1)
        target_outputs = torch.stack(target_outputs, dim=1)
        num_timesteps_list = torch.tensor(num_timesteps_list, device=sequence.device)
        sigmas_list = torch.stack(sigmas_list, dim=1)
        loss_weights_list = torch.stack(loss_weights_list, dim=1)

        return ConsistencyTrainingOutput(
        next_outputs, current_outputs, target_outputs, num_timesteps_list, sigmas_list, loss_weights_list
        )