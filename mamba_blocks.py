# -*- coding: utf-8 -*-
"""
Mamba 状态空间模型核心实现
从 coupled-mamba-main 中提取和改编
"""

import math
from typing import Optional
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

# 尝试导入优化的 Mamba 操作
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None
    print("Warning: mamba_ssm ops not available, using fallback implementation")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    """
    Mamba 状态空间模型块
    核心组件：
    - 输入投影（扩展维度）
    - 因果卷积（短期依赖）
    - 选择性扫描（状态空间建模）
    - 输出投影
    """
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 8) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 输入投影层：d_model -> d_inner * 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 因果卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM 参数投影
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 初始化 dt 投影
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D 初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" 参数
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        Args:
            hidden_states: (B, L, D)
        Returns: 
            out: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        # 简化版本：直接使用标准路径
        # 输入投影
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        xz = rearrange(xz, "b l d -> b d l")
        
        A = -torch.exp(self.A_log.float())
        
        x, z = xz.chunk(2, dim=1)
        
        # 卷积
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # SSM
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # 选择性扫描
        if selective_scan_fn is not None and self.use_fast_path:
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # Fallback: 简化的离散化实现
            y = self._selective_scan_fallback(x, dt, A, B, C, z)
        
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def _selective_scan_fallback(self, x, dt, A, B, C, z):
        """简化的选择性扫描后备实现"""
        dt = F.softplus(dt + self.dt_proj.bias.float().view(1, -1, 1))
        
        # 离散化 A 和 B
        dA = torch.exp(torch.einsum("b d l, d n -> b d l n", dt, A))
        dB = torch.einsum("b d l, b n l -> b d l n", dt, B)
        
        # 扫描
        batch, d_inner, seqlen = x.shape
        _, d_state, _ = B.shape
        
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(seqlen):
            h = h * dA[:, :, i] + x[:, :, i].unsqueeze(-1) * dB[:, :, i]
            y = torch.einsum("b d n, b n -> b d", h, C[:, :, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=2)  # (B, D, L)
        y = y + self.D.float().view(1, -1, 1) * x
        y = y * self.act(z)
        
        return y


class Block(nn.Module):
    """
    Mamba Block 包装器，包含归一化和残差连接
    """
    def __init__(
        self, 
        dim, 
        mixer_cls, 
        norm_cls=nn.LayerNorm, 
        fused_add_norm=False, 
        residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        """
        Args:
            hidden_states: 输入序列
            residual: 残差（用于残差连接）
        Returns:
            (hidden_states, residual)
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # 使用融合的 add+norm 操作（如果可用）
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if fused_add_norm_fn is not None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                # Fallback
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm(residual)
        
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """创建一个 Mamba Block"""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    
    if rms_norm and RMSNorm is not None:
        norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    else:
        norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ====== 跨模态增强模块 ======

class CrossModalEnhancer(nn.Module):
    """
    跨模态增强网络：用于融合其他两个模态的信息来增强当前模态
    采用 ResNet 风格的残差结构
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout_p=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_p),
        )
        
        # 如果维度不匹配，添加投影层
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D_in] - 通常是两个模态拼接后的特征
        Returns:
            out: [B, T, D_out]
        """
        out = self.net(x)
        
        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        
        out = self.norm(out + residual)
        return out


class FusionProjection(nn.Module):
    """
    模态特征投影层：将不同维度的模态特征投影到统一的隐藏空间
    """
    def __init__(self, input_dim, output_dim, dropout_p=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D_in]
        Returns:
            out: [B, T, D_out]
        """
        return self.projection(x)

