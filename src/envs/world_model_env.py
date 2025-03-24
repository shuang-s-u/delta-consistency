from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision

from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_softmax_over_buckets, symexp


@dataclass
class WorldModelEnvOutput:
    # 当前模型的观测值
    frames: torch.FloatTensor
    # 
    wm_output_sequence: torch.FloatTensor


# 可能需要改动
class WorldModelEnv:
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.env = env

        self.obs = None
        self.x = None
        self.last_latent_token_emb = None

    # @torch.no_grad()
    # def reset(self) -> torch.FloatTensor:
    #     assert self.env is not None
    #     obs = rearrange(torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device), 'c h w -> 1 1 c h w')
    #     # 有 1 行，但没有列，类似于 [] 这样一个空数组，只是它的形状仍然符合 PyTorch 的张量格式。
    #     act = torch.empty(obs.size(0), 0, dtype=torch.long, device=self.device)
    #     return self.reset_from_past(obs, act)

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None

        # 获取 episode 的第一个状态
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device)
        obs = rearrange(obs, 'c h w -> 1 1 c h w')  # 形状调整为 (1, 1, C, H, W)

        # 初始化 buffer
        obs_buffer = [obs]
        act_buffer = [torch.empty(1, 0, dtype=torch.long, device=self.device)]  # 形状匹配 (1, 0)

        # 获取接下来的 3 个连续状态（总共 4 个状态）
        for _ in range(3):
            action = torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)  # 形状 (1, 1)
            next_obs, _, done, _ = self.env.step(action.item())  # 获取下一个状态
            next_obs = torchvision.transforms.functional.to_tensor(next_obs).to(self.device)
            next_obs = rearrange(next_obs, 'c h w -> 1 1 c h w')  # 形状调整

            obs_buffer.append(next_obs)  # 存储观察值
            act_buffer.append(action)  # 存储动作，形状为 (1, 1)

            # 如果 episode 结束（done = True），则重新初始化环境
            if done:
                self.env.reset()

        # 将 4 个连续状态堆叠成一个形状为 (1, 4, C, H, W) 的张量
        obs_buffer = torch.cat(obs_buffer, dim=1)  # 在时间维度（dim=1）上连接

        # 将 3 个动作堆叠成形状 (1, 3)，并确保符合 `act` 的格式 (1, 0)
        act_buffer = torch.cat(act_buffer, dim=1) if len(act_buffer) > 1 else act_buffer[0]
        # 存储 buffer
        obs_buffer = obs_buffer
        act_buffer = act_buffer
        # 返回最后一个时间步的观察值和一个空字典（预留额外信息）
        return self.reset_from_past(obs_buffer, act_buffer)

    # @torch.no_grad()
    # def reset_from_past(self, obs: torch.FloatTensor, act: torch.LongTensor) -> Tuple[WorldModelEnvOutput, WorldModelEnvOutput]:
    #     # (B, T, C, H, W)，过去 T 个时间步的观测值（图像）。 act: (B, T-1)，过去 T-1 个时间步的离散动作。
    #     # 6 个 self.obs 是第六个
    #     self.obs = obs[:, -1:]
    #     self.x = None
    #     self.last_latent_token_emb = None
    #     # self.world_model.transformer.reset_kv_cache(n=obs.size(0))

    #     delta_outputs = self.tokenizer.burn_in(obs, act)
    #     wm_output_sequence = self.world_model.burn_in(obs, act, delta_outputs, self.world_model.consistency_model)

    #     # B, T-1, C, H, W)，用于策略训练的观测序列（不包含最后一个）。
    #     obs_burn_in_policy = WorldModelEnvOutput(obs[:, :-1], wm_output_sequence[:, :-1]) 
    #     # (B, 1, C, H, W)，世界模型在 burn-in 结束后得到的最新观测。
    #     first_obs = WorldModelEnvOutput(obs[:, -1:], wm_output_sequence[:, -1:])

    #     return obs_burn_in_policy, first_obs  
    @torch.no_grad()
    def reset_from_past(self, obs: torch.FloatTensor, act: torch.LongTensor) -> Tuple[WorldModelEnvOutput, WorldModelEnvOutput]:
        # (B, T, C, H, W) -> 过去 T 个时间步的观测值（图像）
        # act: (B, T-1) -> 过去 T-1 个时间步的离散动作
        # 拼接得到 4 个动作
        b = obs.size(0)
        self.act_buffer = act[:, -4: ]

        self.obs = obs[:, -1:] # 只取最后一个时间步的观测值
        self.x = None
        self.last_latent_token_emb = None
        # self.world_model.transformer.reset_kv_cache(n=obs.size(0))

        # 计算世界模型的 burn-in 输出
        delta_outputs = self.tokenizer.burn_in(obs, act)

        # 取最后 4 个时间步的观测值
        frames_emb = self.world_model.frame_cnn(obs[:, -5:])
        self.obs_buffer = self.world_model.conv_emb(torch.cat((delta_outputs[:, -5:-1], frames_emb[:, -5:-1]), dim=2).view(-1, 68, 8, 8))
        self.obs_buffer = rearrange(self.obs_buffer, '(b t) c h w -> b t c h w', b=b)

        # self.obs_buffer = delta_outputs[:, -5: -1]  # (B, 4, C, H, W)
        # delta_outputs = self.tokenizer.burn_in(self.obs_buffer, self.act_buffer)
        wm_output_sequence = self.world_model.burn_in(obs, act, delta_outputs, self.world_model.consistency_model)
        # wm_output_sequence = self.world_model.burn_in(self.obs_buffer, self.act_buffer, delta_outputs, self.world_model.consistency_model)

        # B, T-1, C, H, W)，用于策略训练的观测序列（不包含最后一个）。
        obs_burn_in_policy = WorldModelEnvOutput(obs[:, :-1], wm_output_sequence[:, :-1]) 
        # (B, 1, C, H, W)，世界模型在 burn-in 结束后得到的最新观测。
        first_obs = WorldModelEnvOutput(obs[:, -1:], wm_output_sequence[:, -1:])

        return obs_burn_in_policy, first_obs  
    
    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor]) -> Tuple[Optional[WorldModelEnvOutput], float, float, None]:
        # print(f'############## action_buffer is {self.act_buffer.shape}')
        # print(f'############## action.shape is {action.shape}')
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
        # 在 world model中进行一步预测
        # self.act_buffer[:, -1] = action
        self.act_buffer[:, -1] = action.squeeze(-1)
        # next_obs, logit_reward, logit_end, denoising_trajectory
        next_delta_obs, logits_reward, logits_end, _ = self.world_model.sample(self.obs_buffer, self.act_buffer)

        next_delta_obs = next_delta_obs.unsqueeze(1)

        self.obs = self.tokenizer.decode(
            self.obs,
            action,
            next_delta_obs,
            should_clamp=True
        )

        # # 更新潜在连续帧
        # print(f'################ self.obs.shape is {self.obs.shape}')
        # self.x = self.world_model.frame_cnn(self.obs)
        # print(f'############### self.x.shape is {self.x.shape}')
        # # self.x = rearrange(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')
        # # print(f'################连续帧的 shape is {self.x.shape}')
        # b = self.obs.size(0)
        # input_obs = self.world_model.conv_emb(torch.cat((self.obs, self.x), dim=2).view(-1, 68, 8, 8))
        # input_obs = rearrange(input_obs, '(b t) c h w -> b t c h w', b=b)

        # 更新 buffers
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)

        next_delta_obs = next_delta_obs.squeeze(1)
        self.obs_buffer[:, -1] = next_delta_obs

        # 处理 reward done
        if self.world_model.config.two_hot_rews:
            reward = symexp(compute_softmax_over_buckets(logits_reward))
        else:
            reward = Categorical(logits=logits_reward).sample().float() - 1
        reward = reward.flatten().cpu().numpy()
        done = Categorical(logits=logits_end).sample().bool().flatten().cpu().numpy()

        obs = WorldModelEnvOutput(frames=self.obs, wm_output_sequence=next_delta_obs)

        return obs, reward, done, None

    # @torch.no_grad()
    # def step(self, action: Union[int, np.ndarray, torch.LongTensor]) -> Tuple[Optional[WorldModelEnvOutput], float, float, None]:
    #     # if self.world_model.transformer.num_blocks_left_in_kv_cache <= 1:
    #     #     self.world_model.transformer.reset_kv_cache(n=self.obs.size(0))
    #     #     self.last_latent_token_emb = None

    #     wm_output_sequence = []

    #     # if not isinstance(action, torch.Tensor):
    #     #     action = torch.tensor(action, dtype=torch.long).reshape(-1, 1).to(self.device)
    #     # # a = self.world_model.act_emb(action)

    #     if self.last_latent_token_emb is None:
    #         if self.x is None:
    #             outputs_wm = self.world_model(action , use_kv_cache=True)
    #         else:
    #             outputs_wm = self.world_model(torch.cat((self.x, action ), dim=1), use_kv_cache=True)
    #     else:
    #         outputs_wm = self.world_model(torch.cat((self.last_latent_token_emb, self.x, action), dim=1), use_kv_cache=True)

    #     wm_output_sequence.append(outputs_wm.output_sequence)

    #     if self.world_model.config.two_hot_rews:
    #         reward = symexp(compute_softmax_over_buckets(outputs_wm.logits_rewards))
    #     else:
    #         reward = Categorical(logits=outputs_wm.logits_rewards).sample().float() - 1
    #     reward = reward.flatten().cpu().numpy()
    #     done = Categorical(logits=outputs_wm.logits_ends).sample().bool().flatten().cpu().numpy()

    #     latent_tokens = []

    #     #### 处理 latent_tokens
    #     # 1024
    #     latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
    #     latent_tokens.append(latent_token)

    #     for _ in range(self.tokenizer.config.num_tokens - 1):
    #         latent_token_emb = self.world_model.latents_emb(latent_token)
    #         outputs_wm = self.world_model(latent_token_emb, use_kv_cache=True)
    #         wm_output_sequence.append(outputs_wm.output_sequence)

    #         latent_token = Categorical(logits=outputs_wm.logits_latents).sample()
    #         latent_tokens.append(latent_token)

    
    #     self.last_latent_token_emb = self.world_model.latents_emb(latent_token)

    #     q = self.tokenizer.quantizer.embed_tokens(torch.stack(latent_tokens, dim=-1))
    #     #### 处理 latent_tokens end

    #     self.obs = self.tokenizer.decode(
    #         self.obs,
    #         action,
    #         rearrange(q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokenizer.tokens_grid_res, k=self.tokenizer.token_res, l=self.tokenizer.token_res),
    #         should_clamp=True
    #     )

    #     # 更新潜在连续帧
    #     self.x = rearrange(self.world_model.frame_cnn(self.obs), 'b 1 k e -> b k e')

    #     obs = WorldModelEnvOutput(frames=self.obs, wm_output_sequence=torch.cat(wm_output_sequence, dim=1))

    #     return obs, reward, done, None

    @torch.no_grad()
    def render(self):
        return Image.fromarray(rearrange(self.obs, '1 1 c h w -> h w c').mul(255).cpu().numpy().astype(np.uint8))

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        raise NotImplementedError
