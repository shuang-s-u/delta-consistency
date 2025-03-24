from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .block import Conv3x3, FourierFeatures, GroupNorm, UNet

'''
cfg = InnerModelConfig(
    img_channels=68,  # 输入图像通道数
    num_steps_conditioning=4,  # 条件时间步数，根据实际情况调整
    cond_channels=256,  # 条件向量维度
    depths=[2, 2],  # 两次下采样，对应两个处理阶段
    channels=[512, 512],  # 下采样后512，最后512
    attn_depths=[False, True],  # 在第二阶段使用注意力
    num_actions=10  # 动作类别数，根据实际情况调整
)

    inner_model:
      _target_: models.diffusion.InnerModelConfig
      img_channels: 3
      num_steps_conditioning: 4
      cond_channels: 256
      depths: [2,2,2,2]
      channels: [64,64,64,64]
      attn_depths: [0,0,0,0]
'''
@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        # c_noise: (batch_size, cond_channels)
        # act: # b t e -> b (t e)
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
