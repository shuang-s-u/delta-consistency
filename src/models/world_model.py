from dataclasses import dataclass

from torch import Tensor
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
from einops import rearrange
# from src.models import consistency
# from src.models.consistency import consistency_sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from .consistency.block import Conv3x3

from .convnet import FrameCnnConfig, FrameEncoder
from data import Batch

from models.consistency import ConsistencyModel, ConsistencyModelConfig, ConsistencySampler, ConsistencySamplerConfig, ConsistencyTrainingOutput


from .tokenizer import Tokenizer
from utils import init_weights, LossWithIntermediateLosses, symlog, two_hot, pseudo_huber_loss

# from .slicer import  Head
# from .transformer import TransformerEncoder, TransformerConfig


@dataclass
class WorldModelOutput:
    consistency_outputs: ConsistencyTrainingOutput
    # output_sequence: torch.FloatTensor
    # logits_latents: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


@dataclass
class WorldModelConfig:
    # # == codebook_size: 1024 
    # latent_vocab_size: int

    # tempo: 8
    latent_image_size: int
    # tempo: 64
    latent_image_channels: int


    num_actions: int
    image_channels: int
    image_size: int

    latents_weight: float
    rewards_weight: float
    ends_weight: float
    lambda_consistency: float 
    lambda_prediction: float 

    two_hot_rews: bool
    consistency_model_config: ConsistencyModelConfig
    consistency_sampler_config: ConsistencySamplerConfig
    # transformer_config: TransformerConfig
    frame_cnn_config: FrameCnnConfig


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.consistency_model = ConsistencyModel(config.consistency_model_config)
        self.consistency_sampler = ConsistencySampler(config.consistency_sampler_config)

        # 确保i-frames的image_size和tokenizer后的delta-frames一样（64,8,8）
        assert config.image_size // 2 ** sum(config.frame_cnn_config.down) == config.latent_image_size
        # 获取i-frames编码器
        self.frame_cnn = FrameEncoder(config.frame_cnn_config)
        # 64 + 4 -> 64
        self.conv_emb = Conv3x3(self.config.latent_image_channels + self.config.frame_cnn_config.latent_dim, self.config.latent_image_channels)

        # tempo:暂时不对 action 进行处理

        # 定义两个预测头：Rew and End  # 把consistency的输出b t e h w -> b t (e h w) consistenc_ouputs.next_outputs.shape is torch.Size([64, 22, 68, 8, 8])
        self.head_emb = config.latent_image_channels * (config.latent_image_size **2)
        self.head = nn.Sequential(
            nn.Linear(self.head_emb, self.head_emb),
            nn.SiLU(),
            nn.Linear(self.head_emb, 3+2, bias=False)
        )

        # self.transformer = TransformerEncoder(config.transformer_config)
        # # 确保图像编码后的维度（将通道* h * w之后的）与Transformer的输入维度匹配
        # assert ((config.image_size // 2 ** sum(config.frame_cnn_config.down)) ** 2) * config.frame_cnn_config.latent_dim == config.transformer_config.embed_dim
        # # 图像编码器（CNN + 维度重排 + LayerNorm）
        # # 转为 (batch, time, 1, height*width*channels)）。
        # self.frame_cnn = nn.Sequential(FrameEncoder(config.frame_cnn_config), Rearrange('b t c h w -> b t 1 (h w c)'), nn.LayerNorm(config.transformer_config.embed_dim))

        # # 动作和Δ-tokens的嵌入层
        # # self.encoder_act_emb = nn.Embedding(config.num_actions, config.image_size ** 2)
        # self.act_emb = nn.Embedding(config.num_actions, config.transformer_config.embed_dim)
        # self.latents_emb = nn.Embedding(config.latent_vocab_size, config.transformer_config.embed_dim)

        # # 定义动作和Δ-tokens在序列中的位置掩码
        # act_pattern = torch.zeros(config.transformer_config.tokens_per_block)
        # act_pattern[1] = 1
        # act_and_latents_but_last_pattern = torch.zeros(config.transformer_config.tokens_per_block) 
        # act_and_latents_but_last_pattern[1:-1] = 1

        # 定义三个预测头：Δ-tokens、奖励、终止信号
        # self.head_latents = Head(
        #     max_blocks=config.transformer_config.max_blocks,
        #     block_mask=act_and_latents_but_last_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
        #         nn.Linear(config.transformer_config.embed_dim, config.latent_vocab_size)
        #     )
        # )

        # self.head_rewards = Head(
        #     max_blocks=config.transformer_config.max_blocks,
        #     block_mask=act_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
        #         nn.Linear(config.transformer_config.embed_dim, 255 if config.two_hot_rews else 3)
        #     )
        # )

        # self.head_ends = Head(
        #     max_blocks=config.transformer_config.max_blocks,
        #     block_mask=act_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.transformer_config.embed_dim, config.transformer_config.embed_dim), nn.ReLU(),
        #         nn.Linear(config.transformer_config.embed_dim, 2)
        #     )
        # )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"
    

    def forward(self, sequence: torch.FloatTensor, actions: torch.FloatTensor, current_training_step: int, total_training_steps: int) -> WorldModelOutput:

        consistency_outputs = self.consistency_model(sequence, actions, current_training_step, total_training_steps)
        # 展平：
        outputs_for_head= consistency_outputs.next_outputs
        # consistenc_ouputs.next_outputs.shape is torch.Size([64, 22, 68, 8, 8])
        # print(f'consistenc_ouputs.next_outputs.shape is {outputs_for_head.shape}')
        outputs_for_head = rearrange(outputs_for_head, 'b t e h w -> b t (e h w)')
        logits = self.head(outputs_for_head)
        logits_rewards = logits[:, :, :-2]
        logits_ends = logits[:, :, -2:]

        return WorldModelOutput(consistency_outputs, logits_rewards, logits_ends)

    # def forward(self, sequence: torch.FloatTensor, use_kv_cache: bool = False) -> WorldModelOutput:      
    #     prev_steps = self.transformer.keys_values.size if use_kv_cache else 0
    #     num_steps = sequence.size(1)

    #     outputs = self.transformer(sequence, use_kv_cache=use_kv_cache)

    #     logits_latents = self.head_latents(outputs, num_steps, prev_steps)
    #     logits_rewards = self.head_rewards(outputs, num_steps, prev_steps)
    #     logits_ends = self.head_ends(outputs, num_steps, prev_steps)

    #     return WorldModelOutput(outputs, logits_latents, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, current_training_step: int, total_training_steps: int, tokenizer: Tokenizer, **kwargs) -> LossWithIntermediateLosses:
        # batch.ends 是一个 (B, T) 形状的张量， 约束每个 batch 内最多只有 1 个 episode 终止点，防止错误数据输入
        assert torch.all(batch.ends.sum(dim=1) <= 1)

        with torch.no_grad():
            # batch.observations = [64, 26, 3, 64, 64])  batch.actions = [64, 26]
            latent_frames = tokenizer(batch.observations[:, :-1], batch.actions[:, :-1], batch.observations[:, 1:])

        # with torch.no_grad():
        #     # batch.observations = [64, 26, 3, 64, 64])  batch.actions = [64, 26]
        #     latent_tokens = tokenizer(batch.observations[:, :-1], batch.actions[:, :-1], batch.observations[:, 1:]).token

        # [64, 25
        b, _, c, h, w = latent_frames.size()
        # # k个latent token # [64, 25, 4]
        # b, _, k = latent_tokens.size()

        # 原来是[64, 26, 1, 256]，现在 shape =  [64, 26, 4, 8, 8])
        frames_emb = self.frame_cnn(batch.observations)
        # 补全最后一个时刻的 Delta-frames
        extra_time_step = torch.zeros((b, 1, c, h, w), device=latent_frames.device, dtype=latent_frames.dtype)
        latent_frames_full = torch.cat((latent_frames, extra_time_step), dim=1)
        # 在c这个维度上进行拼接 Size([64, 26, 68, 8, 8])
        sequence = self.conv_emb(torch.cat((frames_emb, latent_frames_full), dim=2).view(-1, self.config.latent_image_channels + self.config.frame_cnn_config.latent_dim, h, w))
        sequence = rearrange(sequence, '(b t) c h w -> b t c h w', b=b)
        actions = rearrange(batch.actions, 'b t -> b t 1')
        
        # print(f'###############frames_emb.shape is {frames_emb.shape}')
        # # print(f'###############latent_tokens.shape is {latent_tokens.shape}')     
        # print(f'###############batch.observations.shape is {batch.observations.shape} and actions.shape is {actions.shape}')
        # print(f'################sequence.shape is {sequence.shape}')
        # print(f'#############current_training_steps is {current_training_step} and total_trainingsteps is {total_training_steps}')
        outputs = self(sequence, actions, current_training_step, total_training_steps)
        # 取出 有效时间步的 mask，防止计算 padding 位置的损失。并取消预条件：4
        num_steps_conditioning = self.config.consistency_model_config.inner_model.num_steps_conditioning
        mask = batch.mask_padding[:, num_steps_conditioning:]
        target_rewards = batch.rewards[:, num_steps_conditioning:]
        target_ends = batch.ends[:, num_steps_conditioning:]

        
        # 计算 consistency model 的相关 loss
        consistency_ouputs = outputs.consistency_outputs
        consistency_loss = (pseudo_huber_loss(consistency_ouputs.next_outputs[mask], consistency_ouputs.current_outputs[mask]) * consistency_ouputs.loss_weights[mask]).mean()
        prediction_loss = F.mse_loss(consistency_ouputs.next_outputs[mask], consistency_ouputs.target_outputs[mask])
        scaling_factor = (consistency_loss.detach() / prediction_loss.detach()).clamp(min=0.1, max=10)
        self.config.lambda_prediction = 1.0 * scaling_factor
        loss_consistency = self.config.lambda_consistency * consistency_loss * self.config.latents_weight
        loss_prediction = self.config.lambda_prediction * prediction_loss * self.config.latents_weight
        # loss_latents = self.config.lambda_consistency * consistency_loss + self.config.lambda_prediction * prediction_loss
        # loss_latents = loss_consistency + loss_prediction
        

        # 计算 head 相关损失
        labels_rewards = two_hot(symlog(target_rewards)) if self.config.two_hot_rews else (target_rewards.sign() + 1).long()
        loss_rewards = F.cross_entropy(outputs.logits_rewards[mask], target=labels_rewards[mask]) * self.config.rewards_weight
        loss_ends = F.cross_entropy(outputs.logits_ends[mask], target=target_ends[mask]) * self.config.ends_weight



        # # act_emb 是一个动作嵌入层，把 离散动作 变成 embedding 表示。
        # # self.act_emb = nn.Embedding(config.num_actions, config.transformer_config.embed_dim)
        # act_tokens_emb = self.act_emb(rearrange(batch.actions, 'b t -> b t 1'))
        # # 补齐 latent_tokens 维度，使得 latent_tokens_emb 的时间步长度与 frames_emb 对齐。
        # # self.latents_emb = nn.Embedding(config.latent_vocab_size, config.transformer_config.embed_dim)
        # latent_tokens_emb = self.latents_emb(torch.cat((latent_tokens, latent_tokens.new_zeros(b, 1, k)), dim=1))
        # # 在token维度上进行拼接，然后展开成一个长序列 (batch_size, sequence_length, embedding_dim)
        # sequence = rearrange(torch.cat((frames_emb, act_tokens_emb, latent_tokens_emb), dim=2), 'b t p1k e -> b (t p1k) e')
        # [64, 25, 4] -> # labels_latents.shape = sum(mask == ture)
        # labels_latents = latent_tokens[mask[:, :-1]].flatten()
        # # repeat作用： 在 K 维上扩展，使其形状变为 (B, (T-1) * K)。
        # # outputs.logits_latents.shape = (b=64, (T * K) = 26*4 )
        # # outputs.logits_latents.shape is torch.Size([64, 104, 1024])，
        # # logits_latents.shape = (sum(mask == ture), e)
        # logits_latents = outputs.logits_latents[:, :-k][repeat(mask[:, :-1], 'b t -> b (t k)', k=k)]
        # latent_acc = (logits_latents.max(dim=-1)[1] == labels_latents).float().mean()

        # print(f'###############frames_emb.shape is {frames_emb.shape}')
        # # print(f'###############latent_tokens.shape is {latent_tokens.shape}')     
        # print(f'###############batch.observations.shape is {batch.observations.shape} and batch.actions.shape is {batch.actions.shape}')
        # print(f'#############labels_latents.shape is {labels_latents.shape}')
        # print(f'#############outputs.logits_latents.shape is {outputs.logits_latents.shape}')
        # print(f'#############logits_latents.shape is {logits_latents.shape}')

        # return LossWithIntermediateLosses(loss_latents=loss_latents, loss_rewards=loss_rewards, loss_ends=loss_ends), {'latent_accuracy': latent_acc}
        return LossWithIntermediateLosses(loss_prediction = loss_prediction, loss_consistency = loss_consistency, loss_rewards=loss_rewards, loss_ends=loss_ends), {}


    # 设计采样函数
    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tensor: 
        #获取采样器

        next_delta_obs, denoising_trajectory = self.consistency_sampler.sample(self.consistency_model, prev_obs, prev_act)
        # 展平：
        outputs_for_head= next_delta_obs
        outputs_for_head = outputs_for_head.unsqueeze(1)
        outputs_for_head = rearrange(outputs_for_head, 'b t e h w -> b t (e h w)')
        logits = self.head(outputs_for_head)
        logit_reward = logits[:, :, :-2]
        logit_end = logits[:, :, -2:]
        
        return next_delta_obs, logit_reward, logit_end, denoising_trajectory
        

    @torch.no_grad()
    def burn_in(self, sequence: torch.FloatTensor, act: torch.LongTensor, delta_frames: torch.FloatTensor, consistency_model: ConsistencyModel) -> torch.FloatTensor:
        assert sequence.size(1) == act.size(1) + 1 == delta_frames.size(1) + 1
        x_emb = self.frame_cnn(sequence) 
        b = sequence.size(0)
        h = w = self.config.latent_image_size

        wm_input_sequence = self.conv_emb(torch.cat((x_emb[:, :-1], delta_frames), dim=2).view(-1,self.config.latent_image_channels + self.config.frame_cnn_config.latent_dim, h, w))
        wm_input_sequence = rearrange(wm_input_sequence, '(b t) c h w -> b t c h w', b=b)
        
        seq_length = sequence.size(1) - self.config.consistency_model_config.inner_model.num_steps_conditioning
        
        all_obs = wm_input_sequence.clone()
        all_ations = act.clone()
        predict_obs = []
        n = self.config.consistency_model_config.inner_model.num_steps_conditioning
        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            acts = all_ations[:, i : n + i]
            predict_ob, _, _, _ = self.sample(obs, acts)

            predict_obs.append(predict_ob)

        predict_obs = torch.stack(predict_obs, dim=1)

        return predict_obs

        

    # @torch.no_grad()
    # def burn_in(self, obs: torch.FloatTensor, act: torch.LongTensor, latent_tokens: torch.LongTensor, use_kv_cache: bool = False) -> torch.FloatTensor: 
    #     assert obs.size(1) == act.size(1) + 1 == latent_tokens.size(1) + 1

    #     x_emb = self.frame_cnn(obs)
    #     act_emb = rearrange(self.act_emb(act), 'b t e -> b t 1 e')
    #     q_emb = self.latents_emb(latent_tokens)
    #     x_a_q = rearrange(torch.cat((x_emb[:, :-1], act_emb, q_emb), dim=2), 'b t k2 e -> b (t k2) e')
    #     wm_input_sequence = torch.cat((x_a_q, x_emb[:, -1]), dim=1)
    #     wm_output_sequence = self(wm_input_sequence, use_kv_cache=use_kv_cache).output_sequence

    #     return wm_output_sequence
