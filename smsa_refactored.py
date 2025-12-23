# -*- coding: utf-8 -*-
"""
SMSA多模态情感分析模型 - 重构版
集成: MDP3关键帧选择 + Coupled Mamba + MoE-FiLM调制 + M3NET超图 + GS-MCC频域分解
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    安全的L2归一化，避免NaN/Inf
    
    Args:
        x: 输入张量
        dim: 归一化的维度
        eps: 最小范数阈值
    
    Returns:
        归一化后的张量
    """
    norm = x.norm(p=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)  # 防止除零
    return x / norm

# 延迟导入torch_geometric相关库，防止环境不匹配导致崩溃
try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_scatter import scatter_add
    from torch_geometric.utils import softmax
    from torch_geometric.nn.inits import glorot, zeros
    HAS_PYG = True
except (ImportError, Exception) as e:
    print(f"Warning: Failed to import torch_geometric/torch_scatter: {e}")
    HAS_PYG = False
    # 定义伪类以防报错
    class MessagePassing(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()

from torch.nn import Parameter

# 导入Mamba实现
from mamba_blocks import Mamba, Block, create_block, CrossModalEnhancer


# ==================== 位置编码模块 ====================

class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦余弦位置编码 (Vaswani et al., "Attention Is All You Need")
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 特征维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # 计算分母项 (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices
        
        # 注册为buffer（不参与梯度更新，但会随模型保存/加载）
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, T, D]
        Returns:
            带位置编码的张量 [B, T, D]
        """
        seq_len = x.size(1)
        # 截取对应长度的位置编码并加到输入上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TextGuidedGating(nn.Module):
    """
    文本引导的软门控模块
    
    用全局文本特征来"预筛选"视听帧，替代硬性的关键帧抽取。
    
    逻辑:
    1. 取文本序列的均值或 [CLS] 得到 text_global (shape: [B, D])
    2. 将 text_global 通过 Linear + Sigmoid 生成门控向量
    3. 执行逐元素相乘进行软门控
    """
    def __init__(self, text_dim: int, target_dim: int, dropout: float = 0.1):
        """
        Args:
            text_dim: 文本特征维度 (text_global的维度)
            target_dim: 目标模态特征维度 (audio/video投影后的维度)
            dropout: dropout比率
        """
        super().__init__()
        self.text_dim = text_dim
        self.target_dim = target_dim
        
        # 门控生成器: text_global -> gate vector
        # 使用两层MLP增强表达能力
        self.gate_generator = nn.Sequential(
            nn.Linear(text_dim, target_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(target_dim, target_dim),
            nn.Sigmoid(),  # 门控值在 [0, 1] 范围内
        )
    
    def forward(
        self,
        text_global: torch.Tensor,  # [B, text_dim]
        target_seq: torch.Tensor,   # [B, T, target_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_global: 全局文本特征 [B, text_dim]
            target_seq: 目标模态序列 (audio/video) [B, T, target_dim]
        
        Returns:
            gated_seq: 门控后的序列 [B, T, target_dim]
            gate: 门控向量 [B, target_dim] (用于分析)
        """
        # 生成门控向量 [B, target_dim]
        gate = self.gate_generator(text_global)
        
        # 广播并应用门控: [B, target_dim] -> [B, 1, target_dim] * [B, T, target_dim]
        gated_seq = target_seq * gate.unsqueeze(1)
        
        return gated_seq, gate


# ==================== 配置类 ====================
class ModelConfig:
    """模型组件开关配置"""
    def __init__(
        self,
        # 关键帧选择 (MDP3) - 新的百分比自适应模式
        use_key_frame_selector=True,
        n_segments=4,              # 分段数（默认：4段）
        frame_ratio=60,            # 每段选择百分比1-100（默认：60%）
        key_frame_lambda=0.2,
        
        # Coupled Mamba
        use_coupled_mamba=True,
        
        # MoE-FiLM调制 (MoFME)
        use_moe_film=True,
        num_film_experts=8,
        film_top_k=4,
        
        # 超图建模 (M3NET)
        use_hypergraph=True,
        num_hypergraph_layers=3,
        
        # 频域分解 (GS-MCC)
        use_frequency_decomp=True,
        num_fourier_layers=4,
        
        # 超球体正则
        use_sphere_regularization=True,
        sphere_radius=1.0,
        
        # ⭐ 先验模态（social/context）是否直接参与融合
        direct_fusion_priors=True,  # True=参与融合，False=只用于调制
        
        # ⭐ MLP架构选择
        use_improved_mlp=False,  # True=改进版（4层深层），False=原始版（2层简单）
    ):
        self.use_key_frame_selector = use_key_frame_selector
        self.n_segments = n_segments
        self.frame_ratio = frame_ratio
        self.key_frame_lambda = key_frame_lambda
        
        # 验证参数
        if not (1 <= frame_ratio <= 100):
            raise ValueError(f"frame_ratio必须在1-100之间，当前值：{frame_ratio}")
        
        self.use_coupled_mamba = use_coupled_mamba
        
        self.use_moe_film = use_moe_film
        self.num_film_experts = num_film_experts
        self.film_top_k = film_top_k
        
        self.use_hypergraph = use_hypergraph
        self.num_hypergraph_layers = num_hypergraph_layers
        
        self.use_frequency_decomp = use_frequency_decomp
        self.num_fourier_layers = num_fourier_layers
        
        self.use_sphere_regularization = use_sphere_regularization
        self.sphere_radius = sphere_radius
        
        self.direct_fusion_priors = direct_fusion_priors
        self.use_improved_mlp = use_improved_mlp


# ==================== 1. 关键帧选择模块 (参考MDP3) ====================

class MultiGaussianKernel(nn.Module):
    """多高斯核函数"""
    def __init__(self, alphas=[2**k for k in range(-3, 2)]):
        super().__init__()
        self.alphas = alphas
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        Y = X.unsqueeze(0) if Y is None else Y.unsqueeze(0)
        X = X.unsqueeze(1)
        l2_distance_square = ((X - Y) ** 2).sum(2)
        return sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in self.alphas])


class KeyFrameSelector(nn.Module):
    """
    关键帧选择器 (参考MDP3) - 新的百分比自适应模式
    使用语句级文本查询帧级音画模态，选择关键帧
    
    新模式说明：
    - n_segments: 将视频分成多少段
    - frame_ratio: 每段选择的帧百分比（1-100）
    - 实际选择帧数会根据视频长度自动计算
    """
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        n_segments: int = 4,          # 分段数（默认：4段）
        frame_ratio: int = 60,        # 每段选择百分比（默认：60%）
        lamda: float = 0.2,
        condition_size: int = 1,
    ):
        super().__init__()
        self.n_segments = n_segments
        self.frame_ratio = frame_ratio
        self.lamda = lamda
        self.condition_size = condition_size
        self.hidden_dim = hidden_dim
        
        # 验证参数
        if not (1 <= frame_ratio <= 100):
            raise ValueError(f"frame_ratio必须在1-100之间，当前值：{frame_ratio}")
        
        # 投影层：将音频和视频投影到统一空间
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_query_proj = nn.Linear(text_dim, hidden_dim)
        
        # 多高斯核
        self.kernel = MultiGaussianKernel(alphas=[2**k for k in range(-3, 2)])
        
        # ====== 帧数统计 ======
        self.frame_stats = {
            'total_utterances': 0,
            'frame_counts': [],
            'min_frames': float('inf'),
            'max_frames': 0,
        }
        self.enable_logging = False  # 默认关闭，由训练器控制
        self.log_every = 32  # 每N个utterance打印一次
        self.logger = None  # 由训练器设置logger
        
    def _compute_n_selection(self, seq_len: int) -> int:
        """
        根据视频长度和配置计算应该选择多少帧（新的百分比模式）
        
        Args:
            seq_len: 视频的帧数
        
        Returns:
            n_selection: 应该选择的关键帧数量
        """
        # 计算每段的帧数
        frames_per_segment = seq_len / self.n_segments
        
        # 计算每段应该选择多少帧（向下取整，但保证至少1帧）
        select_per_segment = max(1, int(frames_per_segment * self.frame_ratio / 100))
        
        # 总选择数
        total_selection = select_per_segment * self.n_segments
        
        # 确保不超过视频总帧数
        total_selection = min(total_selection, seq_len)
        
        return total_selection
    
    def forward(
        self,
        audio_sequence: torch.Tensor,  # [B, T, D_audio]
        video_sequence: torch.Tensor,  # [B, T, D_video]
        text_global: torch.Tensor,     # [B, D_text]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择关键帧（新的百分比自适应模式 + 动态padding）
        
        Returns:
            selected_audio: [B, max_frames, D_audio] - 动态padding到batch内最大长度
            selected_video: [B, max_frames, D_video] - 动态padding到batch内最大长度
            selected_indices: [B, max_frames] - 索引（padding部分为-1）
            mask: [B, max_frames] - 1表示有效帧，0表示padding帧 ⭐ 关键！
        """
        batch_size, seq_len, _ = audio_sequence.shape
        device = audio_sequence.device
        
        # ====== 计算每个样本应该选择多少帧 ======
        actual_lengths = []
        for b in range(batch_size):
            n_selection_b = self._compute_n_selection(seq_len)
            actual_lengths.append(n_selection_b)
        
        # ====== 帧数统计和打印 ======
        if self.enable_logging and self.training:
            self.frame_stats['total_utterances'] += batch_size
            for _ in range(batch_size):
                self.frame_stats['frame_counts'].append(seq_len)
            self.frame_stats['min_frames'] = min(self.frame_stats['min_frames'], seq_len)
            self.frame_stats['max_frames'] = max(self.frame_stats['max_frames'], seq_len)
            
            # 定期打印（前5次）
            should_print = (
                self.frame_stats['total_utterances'] % self.log_every == 0 and
                self.frame_stats['total_utterances'] <= self.log_every * 5
            )
            
            if should_print:
                recent_frames = self.frame_stats['frame_counts'][-min(self.log_every, len(self.frame_stats['frame_counts'])):]
                avg_frames = sum(recent_frames) / len(recent_frames)
                avg_selected = sum(actual_lengths) / len(actual_lengths)
                
                # 使用logger输出（如果可用），否则使用print
                if self.logger is not None:
                    self.logger.info(f"\n{'='*70}")
                    self.logger.info(f"[KeyFrameSelector] 帧数统计 (已处理 {self.frame_stats['total_utterances']} 个utterances)")
                    self.logger.info(f"{'='*70}")
                    self.logger.info(f"  当前batch: {batch_size}样本 × {seq_len}帧/样本")
                    self.logger.info(f"  最近{len(recent_frames)}个utterance平均帧数: {avg_frames:.1f}")
                    self.logger.info(f"  历史帧数范围: [{self.frame_stats['min_frames']}, {self.frame_stats['max_frames']}]")
                    self.logger.info(f"  MDP3配置: n_segments={self.n_segments}, frame_ratio={self.frame_ratio}%")
                    self.logger.info(f"  当前batch选择帧数: {actual_lengths}, 平均: {avg_selected:.1f}")
                    self.logger.info(f"{'='*70}\n")
                else:
                    # 如果没有logger，使用print并刷新
                    import sys
                    sys.stdout.flush()
                    print(f"\n{'='*70}", flush=True)
                    print(f"[KeyFrameSelector] 帧数统计 (已处理 {self.frame_stats['total_utterances']} 个utterances)", flush=True)
                    print(f"{'='*70}", flush=True)
                    print(f"  当前batch: {batch_size}样本 × {seq_len}帧/样本", flush=True)
                    print(f"  最近{len(recent_frames)}个utterance平均帧数: {avg_frames:.1f}", flush=True)
                    print(f"  历史帧数范围: [{self.frame_stats['min_frames']}, {self.frame_stats['max_frames']}]", flush=True)
                    print(f"  MDP3配置: n_segments={self.n_segments}, frame_ratio={self.frame_ratio}%", flush=True)
                    print(f"  当前batch选择帧数: {actual_lengths}, 平均: {avg_selected:.1f}", flush=True)
                    print(f"{'='*70}\n", flush=True)
        # ==========================================
        
        # 投影到统一空间
        audio_features = self.audio_proj(audio_sequence)  # [B, T, H]
        video_features = self.video_proj(video_sequence)  # [B, T, H]
        
        # 音画特征拼接
        av_features = (audio_features + video_features) / 2  # 简单平均，也可用concat
        
        # 文本查询
        text_query = self.text_query_proj(text_global)  # [B, H]
        
        # 逐样本选择关键帧
        selected_audio_list = []
        selected_video_list = []
        selected_indices_list = []
        
        for b in range(batch_size):
            av_b = av_features[b]  # [T, H]
            text_b = text_query[b:b+1]  # [1, H]
            n_selection_b = actual_lengths[b]
            
            # 使用DPP选择关键帧（传入动态的n_selection）
            selected_idx = self._select_frames_fast(av_b, text_b, n_selection_b)
            
            # ⭐ 确保选择的帧数正确
            if len(selected_idx) == 0:
                # 如果选择失败，使用均匀采样
                selected_idx = list(range(0, seq_len, max(1, seq_len // n_selection_b)))[:n_selection_b]
            
            if len(selected_idx) < n_selection_b:
                # 如果选择不够，填充最后一帧
                if len(selected_idx) > 0:
                    selected_idx = selected_idx + [selected_idx[-1]] * (n_selection_b - len(selected_idx))
                else:
                    # 极端情况：完全失败，使用第0帧
                    selected_idx = [0] * n_selection_b
            
            selected_idx = selected_idx[:n_selection_b]
            
            # ⭐ 安全检查：确保所有索引在有效范围内
            selected_idx = [min(max(0, idx), seq_len - 1) for idx in selected_idx]
            
            selected_indices_list.append(selected_idx)
            selected_audio_list.append(audio_sequence[b, selected_idx])
            selected_video_list.append(video_sequence[b, selected_idx])
        
        # ====== 动态Padding到batch内最大长度 ======
        max_frames = max(actual_lengths)
        
        padded_audio_list = []
        padded_video_list = []
        padded_indices_list = []
        mask_list = []
        
        for b in range(batch_size):
            actual_len = actual_lengths[b]
            audio_b = selected_audio_list[b]  # [actual_len, D]
            video_b = selected_video_list[b]  # [actual_len, D]
            indices_b = selected_indices_list[b]  # list of length actual_len
            
            if actual_len < max_frames:
                # 需要padding
                pad_len = max_frames - actual_len
                
                # Padding策略：使用零向量（更清晰的"无信息"信号）
                pad_audio = torch.zeros(pad_len, audio_b.shape[-1], device=device, dtype=audio_b.dtype)
                pad_video = torch.zeros(pad_len, video_b.shape[-1], device=device, dtype=video_b.dtype)
                
                audio_padded = torch.cat([audio_b, pad_audio], dim=0)
                video_padded = torch.cat([video_b, pad_video], dim=0)
                
                # 索引padding为-1（标记为无效）
                indices_padded = indices_b + [-1] * pad_len
                
                # 创建mask：前actual_len个为1，后面为0
                mask_b = torch.cat([
                    torch.ones(actual_len, device=device),
                    torch.zeros(pad_len, device=device)
                ], dim=0)
            else:
                # 不需要padding
                audio_padded = audio_b
                video_padded = video_b
                indices_padded = indices_b
                mask_b = torch.ones(actual_len, device=device)
            
            padded_audio_list.append(audio_padded)
            padded_video_list.append(video_padded)
            padded_indices_list.append(torch.tensor(indices_padded, device=device))
            mask_list.append(mask_b)
        
        # Stack成batch
        selected_audio = torch.stack(padded_audio_list)  # [B, max_frames, D_audio]
        selected_video = torch.stack(padded_video_list)  # [B, max_frames, D_video]
        selected_indices = torch.stack(padded_indices_list)  # [B, max_frames]
        mask = torch.stack(mask_list)  # [B, max_frames] ⭐ 关键的mask！
        
        # ====== Mask验证 ======
        assert mask.shape == (batch_size, max_frames), f"Mask shape错误: {mask.shape}"
        assert (mask.sum(dim=1) == torch.tensor(actual_lengths, device=device).float()).all(), \
            "Mask的有效帧数与actual_lengths不匹配！"
        
        return selected_audio, selected_video, selected_indices, mask
    
    def _select_frames_fast(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor, n_selection: int) -> List[int]:
        """
        快速帧选择算法（参考MDP3，支持动态n_selection）
        
        Args:
            image_embeds: 图像嵌入 [T, H]
            text_embeds: 文本嵌入 [1, H]
            n_selection: 要选择的帧数（动态）
        
        Returns:
            selected_indices: 选中的帧索引列表
        """
        N_image = len(image_embeds)
        
        # ⭐ 边界检查1: n_selection不能超过实际帧数
        n_selection = min(n_selection, N_image)
        
        # ⭐ 边界检查2: n_selection至少为1
        n_selection = max(1, n_selection)
        
        # 计算分段大小（基于n_segments）
        segment_size = max(1, N_image // self.n_segments)
        segment_num = (N_image + segment_size - 1) // segment_size
        
        # 动态规划初始化
        INF = 1e9
        dp = [[0.] + [-INF] * n_selection for _ in range(segment_num + 1)]
        trace = [[[] for _ in range(n_selection + 1)] for _ in range(segment_num + 1)]
        
        for seg_idx in range(1, segment_num + 1):
            candidate_index = range(
                (seg_idx - 1) * segment_size,
                min(seg_idx * segment_size, N_image)
            )
            candidate_embeds = [image_embeds[i] for i in candidate_index]
            
            if len(candidate_embeds) == 0:
                continue
            
            sim_matrix = self.kernel(torch.stack(candidate_embeds))
            
            for start_selected_num in range(0, min(n_selection, (seg_idx - 1) * segment_size) + 1):
                conditional_index = trace[seg_idx - 1][start_selected_num][
                    -min(self.condition_size, len(trace[seg_idx - 1][start_selected_num])):]
                offset = len(conditional_index)
                
                additional_embeds = [text_embeds[0].reshape(-1)]
                if conditional_index:
                    additional_embeds += [image_embeds[i] for i in conditional_index]
                
                additional = self.kernel(
                    torch.stack(additional_embeds),
                    torch.stack(additional_embeds + candidate_embeds)
                )
                
                total_matrix = torch.cat([
                    additional,
                    torch.cat([
                        additional[:, -len(sim_matrix):].T,
                        sim_matrix
                    ], dim=1)
                ], dim=0)
                
                max_selection = min(n_selection - start_selected_num, len(candidate_index))
                cur_scores, cur_traces = self._seqdpp_select_super_fast(
                    total_matrix, offset, max_selection
                )
                
                for to_select_num, (cur_score, cur_trace) in enumerate(zip(cur_scores, cur_traces)):
                    cur_trace = [i + int((seg_idx - 1) * segment_size) for i in cur_trace]
                    cur_score = dp[seg_idx - 1][start_selected_num] + cur_score
                    cur_trace = trace[seg_idx - 1][start_selected_num] + cur_trace
                    
                    # ⭐ 边界检查：防止数组越界
                    target_idx = start_selected_num + to_select_num
                    if target_idx <= n_selection:
                        if cur_score > dp[seg_idx][target_idx]:
                            dp[seg_idx][target_idx] = cur_score
                            trace[seg_idx][target_idx] = cur_trace
        
        # ⭐ 安全访问：如果n_selection超出范围，找最接近的
        actual_selection = min(n_selection, len(trace[segment_num]) - 1)
        if actual_selection < 0:
            return []  # 完全失败，返回空列表
        
        return trace[segment_num][actual_selection]
    
    def _seqdpp_select_super_fast(self, total_matrix, offset, to_select_num):
        """超快速序列DPP选择（带数值稳定性保护）"""
        if to_select_num == 0:
            return [0.0], [[]]
        
        # ⭐ 添加数值稳定性检查
        if torch.isnan(total_matrix).any() or torch.isinf(total_matrix).any():
            # 数值异常，返回均匀采样
            return [0.0], [[]]
        
        cur_trace = []
        ret_scores = [0.0]
        r, S_matrix = total_matrix[0:1, 1:], total_matrix[1:, 1:]
        candidate_index = list(range(len(S_matrix) - offset))
        
        # ⭐ 边界检查
        if len(candidate_index) == 0:
            return [0.0], [[]]
        
        conditional_idx = list(range(offset))
        L = None
        if len(conditional_idx) > 0:
            try:
                # ⭐ 添加正则化项提高数值稳定性
                matrix_to_decompose = S_matrix[conditional_idx][:, conditional_idx]
                # 添加小的正则化项确保正定
                matrix_to_decompose = matrix_to_decompose + torch.eye(
                    len(conditional_idx), 
                    dtype=matrix_to_decompose.dtype, 
                    device=matrix_to_decompose.device
                ) * 1e-6
                L = torch.linalg.cholesky(matrix_to_decompose)
            except RuntimeError as e:
                # Cholesky分解失败，矩阵不正定
                return [0.0], [[]]
        
        while len(cur_trace) < to_select_num:
            max_obj = -1e9
            cur_selected_idx = -1
            better_L = None
            
            for i in candidate_index:
                if i in cur_trace:
                    continue
                
                try:
                    cur_idx = i + offset
                    selected_idx = conditional_idx + [j + offset for j in cur_trace] + [cur_idx]
                    
                    if L is None:
                        cur_sim_v = S_matrix[selected_idx][:, selected_idx]
                        # ⭐ 数值保护：确保非负
                        cur_sim_v = torch.clamp(cur_sim_v, min=1e-10)
                        cur_L = torch.sqrt(cur_sim_v).reshape(1, 1)
                        logdet = torch.log(torch.clamp(cur_sim_v, min=1e-10))
                    else:
                        cur_sim_v = S_matrix[cur_idx:cur_idx + 1][:, selected_idx]
                        cur_L, logdet = self._cholesky_update_determinant(L, cur_sim_v)
                    
                    # ⭐ 数值保护：r可能有0或负数
                    r_selected = torch.clamp(r[:, selected_idx], min=1e-10)
                    cur_obj = 1. / self.lamda * 2 * torch.log(r_selected).sum() + logdet
                    
                    # ⭐ 检查结果有效性
                    if torch.isnan(cur_obj) or torch.isinf(cur_obj):
                        continue
                    
                    if cur_obj > max_obj or cur_selected_idx == -1:
                        max_obj = cur_obj
                        cur_selected_idx = i
                        better_L = cur_L
                        
                except (RuntimeError, IndexError) as e:
                    # 计算失败，跳过这个候选
                    continue
            
            # ⭐ 如果没有找到有效的候选，退出
            if cur_selected_idx == -1:
                break
            
            ret_scores.append(max_obj.clone() if isinstance(max_obj, torch.Tensor) else max_obj)
            cur_trace.append(cur_selected_idx)
            L = better_L
        
        ret_traces = [sorted(cur_trace[:j]) for j in range(len(cur_trace) + 1)]
        return ret_scores, ret_traces
    
    def _cholesky_update_determinant(self, L, v):
        """Cholesky更新和行列式计算（带数值保护）"""
        try:
            n = L.shape[0]
            v = v.view(-1, 1)
            
            # ⭐ 数值保护：solve_triangular
            v_projected = torch.linalg.solve_triangular(L, v[:n], upper=False)
            
            # ⭐ 数值保护：确保开方项非负
            diag_value = v[-1] - v_projected.T @ v_projected
            diag_value = torch.clamp(diag_value, min=1e-10)
            new_diag_element = torch.sqrt(diag_value)
            
            new_row = torch.cat((v_projected.flatten(), new_diag_element.view(1)))
            new_L = torch.zeros((n + 1, n + 1), dtype=L.dtype, device=L.device)
            new_L[:n, :n] = L
            new_L[n, :n] = new_row[:-1]
            new_L[n, n] = new_diag_element
            
            # ⭐ 数值保护：log计算
            new_diag = torch.diag(new_L)
            new_diag = torch.clamp(new_diag, min=1e-10)
            new_logdet = 2 * torch.log(new_diag).sum()
            
            # ⭐ 检查结果有效性
            if torch.isnan(new_logdet) or torch.isinf(new_logdet):
                raise RuntimeError("Invalid logdet")
            
            return new_L, new_logdet
            
        except (RuntimeError, Exception) as e:
            # 计算失败，返回原始L和一个安全的logdet
            return L, torch.tensor(0.0, device=L.device, dtype=L.dtype)


# ==================== 2. MoE-FiLM调制模块 (参考MoFME) ====================

class KeepTopK(nn.Module):
    """保留Top-K并置零其他"""
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, x):
        if self.top_k >= x.size(-1):
            return x
        
        # 保留top-k，其余置零
        values, indices = torch.topk(x, self.top_k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        return x * mask


class FiLMExpert(nn.Module):
    """
    单个FiLM专家
    实现: modulation_product * x + modulation_add
    """
    def __init__(self, dim: int):
        super().__init__()
        self.expert_product = nn.Linear(dim, dim)
        self.expert_add = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] 或 [B, D]
        gamma = self.expert_product(x)  # 缩放因子
        beta = self.expert_add(x)       # 偏移因子
        return gamma * x + beta


class MoE_FiLM_Modulation(nn.Module):
    """
    MoE-FiLM调制模块 (参考MoFME)
    使用多个FiLM专家对模态特征进行调制
    """
    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,  # 社会关系或情境的维度
        num_experts: int = 8,
        top_k: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        if hidden_dim is None:
            hidden_dim = feature_dim
        
        # 共享的Router
        self.router = nn.Sequential(
            nn.Linear(condition_dim, num_experts),
            KeepTopK(top_k=top_k),
            nn.Softmax(dim=-1)
        )
        
        # 多个FiLM专家
        self.experts = nn.ModuleList([
            FiLMExpert(feature_dim) for _ in range(num_experts)
        ])
        
        # 共享的FFN (Adaptor)
        self.adaptor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Base network
        self.base_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
    
    def forward(
        self,
        features: torch.Tensor,      # [B, T, D] 或 [B, D]
        condition: torch.Tensor,      # [B, D_cond]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 待调制的特征
            condition: 条件向量 (社会关系或情境)
        Returns:
            modulated: 调制后的特征
            weights: 专家权重 [B, num_experts] (用于分析)
        """
        # 计算专家权重
        weights = self.router(condition)  # [B, num_experts]
        
        # Base transformation
        x = self.base_net(features)
        
        # 加权组合所有专家的输出
        y = 0
        if features.dim() == 3:  # [B, T, D]
            for i in range(self.num_experts):
                weight_i = weights[:, i].view(-1, 1, 1)  # [B, 1, 1]
                y += weight_i * self.experts[i](x)
        else:  # [B, D]
            for i in range(self.num_experts):
                weight_i = weights[:, i].view(-1, 1)  # [B, 1]
                y += weight_i * self.experts[i](x)
        
        # Adaptor (共享FFN)
        y = self.adaptor(y)
        
        # 残差连接
        output = y + features
        
        return output, weights


# ==================== 3. Mamba相关模块 ====================

class MambaBlock(nn.Module):
    """Mamba块（用于非耦合情况）"""
    def __init__(
        self,
        hidden_dim: int,
        dropout_p: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 输入序列
            mask: [B, T] 1=有效帧，0=padding帧
        """
        x = self.mamba(x)
        x = self.dropout(x)
        
        # ⭐ 严格应用mask：将padding位置的输出清零
        if mask is not None:
            # mask: [B, T] -> [B, T, 1]
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask_expanded  # padding位置的特征被清零
        
        return x


class CoupledMambaBlock(nn.Module):
    """
    多模态耦合Mamba块 (保留原实现)
    """
    def __init__(
        self,
        modality_names: List[str],
        hidden_dim: int,
        prior_dim: int,
        dropout_p: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.modality_names = modality_names
        self.hidden_dim = hidden_dim
        self.num_modalities = len(modality_names)
        
        # 每个模态独立的Mamba块
        self.modality_mambas = nn.ModuleDict({
            name: Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for name in modality_names
        })
        
        # 跨模态增强模块
        self.cross_modal_enhancers = nn.ModuleDict({
            name: CrossModalEnhancer(
                input_dim=hidden_dim * (self.num_modalities - 1),
                output_dim=hidden_dim,
                dropout_p=dropout_p,
            ) for name in modality_names
        })
        
        # 先验调制
        self.prior_to_gate = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, self.num_modalities),
        )
        
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(
        self,
        modality_sequences: Dict[str, torch.Tensor],
        prior_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_sequences: 各模态序列
            prior_embedding: 先验嵌入
            mask: [B, T] 1=有效帧，0=padding帧 ⭐
        """
        batch_size = prior_embedding.size(0)
        
        # 1. 每个模态通过自己的Mamba
        mamba_outputs = {}
        for name in self.modality_names:
            x = modality_sequences[name]
            mamba_out = self.modality_mambas[name](x)
            
            # ⭐ 严格应用mask
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
                mamba_out = mamba_out * mask_expanded  # padding位置清零
            
            mamba_outputs[name] = mamba_out
        
        # 2. 根据先验生成模态权重
        prior_embedding = safe_normalize(prior_embedding, dim=-1)  # ⭐ 修复：使用安全归一化
        prior_gates = torch.sigmoid(self.prior_to_gate(prior_embedding))  # [B, M]
        
        # 3. 跨模态增强
        updated = {}
        for i, name in enumerate(self.modality_names):
            other_modalities = [
                mamba_outputs[other_name]
                for other_name in self.modality_names
                if other_name != name
            ]
            
            other_concat = torch.cat(other_modalities, dim=-1)
            cross_modal_info = self.cross_modal_enhancers[name](other_concat)
            
            gate_i = prior_gates[:, i].view(batch_size, 1, 1)
            enhanced = gate_i * cross_modal_info + mamba_outputs[name]
            enhanced = enhanced + modality_sequences[name]
            enhanced = self.dropout(enhanced)
            
            # ⭐ 再次应用mask（防止残差连接引入padding信息）
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
                enhanced = enhanced * mask_expanded  # padding位置清零
            
            updated[name] = enhanced
        
        return updated


class CoupledMambaStack(nn.Module):
    """Coupled Mamba堆叠"""
    def __init__(
        self,
        modality_names: List[str],
        hidden_dim: int,
        prior_dim: int,
        num_layers: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.modality_names = modality_names
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            CoupledMambaBlock(
                modality_names=modality_names,
                hidden_dim=hidden_dim,
                prior_dim=prior_dim,
                dropout_p=dropout_p,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(hidden_dim) for name in modality_names
        })
        
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(
        self,
        modality_sequences: Dict[str, torch.Tensor],
        prior_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_sequences: 各模态序列
            prior_embedding: 先验嵌入
            mask: [B, T] 1=有效帧，0=padding帧 ⭐
        """
        x = modality_sequences
        
        for layer in self.layers:
            updated = layer(modality_sequences=x, prior_embedding=prior_embedding, mask=mask)
            for name in self.modality_names:
                res = x[name]
                upd = self.dropout(updated[name])
                x[name] = self.layer_norms[name](res + upd)
                
                # ⭐ 每层之后都应用mask（防止LayerNorm引入padding信息）
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
                    x[name] = x[name] * mask_expanded  # padding位置清零
        
        return x


# ==================== 4. 注意力池化 ====================

class AttentionPooling(nn.Module):
    """注意力池化：将序列压缩为句级向量"""
    def __init__(
        self,
        input_dim: int,
        cond_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        if cond_dim is not None and cond_dim > 0:
            self.query_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            self.query_vector = nn.Parameter(torch.randn(input_dim))
        
        self.attn_dropout = nn.Dropout(dropout_p)
        self.input_norm = nn.LayerNorm(input_dim)
    
    def forward(
        self,
        sequence: torch.Tensor,
        cond_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = sequence.shape
        
        sequence = self.input_norm(sequence)
        
        if hasattr(self, "query_mlp") and cond_embedding is not None:
            query = self.query_mlp(cond_embedding)  # [B, D]
        else:
            query = self.query_vector.unsqueeze(0).expand(batch_size, -1)
        
        scores = torch.einsum("btd,bd->bt", sequence, query)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=1)
        attn_weights = self.attn_dropout(attn_weights)
        pooled = torch.einsum("bt,btd->bd", attn_weights, sequence)
        return pooled


# ==================== 5. 超图卷积模块 (参考M3NET) ====================

class HypergraphConv(MessagePassing):
    """
    超图卷积层 (参考M3NET)
    支持两种超边：
    1. 上下文超边：连接同模态的不同utterance
    2. 跨模态超边：连接同utterance的不同模态
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs
    ):
        if not HAS_PYG:
            raise ImportError("torch_geometric/torch_scatter not installed or failed to import!")
            
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if not HAS_PYG: return
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
        hyperedge_attr: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [N, D] 节点特征
            hyperedge_index: [2, E] 超边索引 (node_idx, hyperedge_idx)
            hyperedge_weight: [num_edges] 超边权重
            hyperedge_attr: [num_edges, D] 超边属性
        """
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        
        # 计算度数
        D = scatter_add(
            hyperedge_weight[hyperedge_index[1]],
            hyperedge_index[0],
            dim=0,
            dim_size=num_nodes
        )
        D = 1.0 / D
        D[D == float("inf")] = 0
        
        B = scatter_add(
            x.new_ones(hyperedge_index.size(1)),
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges
        )
        B = 1.0 / B
        B[B == float("inf")] = 0
        
        # 两阶段传播
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, size=(num_nodes, num_edges))
        
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, size=(num_edges, num_nodes))
        
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return F.leaky_relu(out)
    
    def message(self, x_j, norm_i):
        H, F = self.heads, self.out_channels
        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j
        return out


class HypergraphModule(nn.Module):
    """
    超图建模模块 (参考M3NET)
    构建两种超边并进行超图卷积
    """
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 超图卷积层
        self.hyperconv_layers = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.ReLU()
    
    def forward(
        self,
        utterance_features: Dict[str, torch.Tensor],  # dict(modality -> [B, D])
        batch_dia_len: List[int],  # 每个对话的utterance数量
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            utterance_features: 语句级特征 {text: [B,D], audio: [B,D], video: [B,D]}
            batch_dia_len: 每个对话中的utterance数量 (例如 [5, 3, 7] 表示batch中3个对话)
        Returns:
            updated_features: 更新后的特征
        """
        modality_names = list(utterance_features.keys())
        num_modalities = len(modality_names)
        
        # 构建超图索引
        hyperedge_index, node_features = self._build_hypergraph(
            utterance_features, batch_dia_len, modality_names
        )
        
        # 超图卷积
        x = node_features
        for layer in self.hyperconv_layers:
            x = layer(x, hyperedge_index)
            x = self.dropout(x)
        
        # 分解回各模态
        updated_features = self._split_features(x, batch_dia_len, modality_names)
        
        return updated_features
    
    def _build_hypergraph(
        self,
        utterance_features: Dict[str, torch.Tensor],
        batch_dia_len: List[int],
        modality_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建超图结构
        返回:
            hyperedge_index: [2, E] 超边索引
            node_features: [N, D] 节点特征 (所有模态拼接)
        """
        device = utterance_features[modality_names[0]].device
        num_modalities = len(modality_names)
        
        # 拼接所有模态特征为节点
        node_features_list = []
        for name in modality_names:
            node_features_list.append(utterance_features[name])
        node_features = torch.cat(node_features_list, dim=0)  # [B*M, D]
        
        # 构建超边
        node_idx_list = []
        hyperedge_idx_list = []
        edge_count = 0
        node_offset = 0
        
        for dia_len in batch_dia_len:
            # 为这个对话构建超边
            
            # 1. 上下文超边：每个模态连接该对话的所有utterance
            for m_idx in range(num_modalities):
                nodes_in_edge = [node_offset + m_idx * dia_len + i for i in range(dia_len)]
                for node in nodes_in_edge:
                    node_idx_list.append(node)
                    hyperedge_idx_list.append(edge_count)
                edge_count += 1
            
            # 2. 跨模态超边：每个utterance连接所有模态
            for utt_idx in range(dia_len):
                nodes_in_edge = [
                    node_offset + m_idx * dia_len + utt_idx
                    for m_idx in range(num_modalities)
                ]
                for node in nodes_in_edge:
                    node_idx_list.append(node)
                    hyperedge_idx_list.append(edge_count)
                edge_count += 1
            
            node_offset += dia_len * num_modalities
        
        hyperedge_index = torch.tensor(
            [node_idx_list, hyperedge_idx_list],
            dtype=torch.long,
            device=device
        )
        
        return hyperedge_index, node_features
    
    def _split_features(
        self,
        node_features: torch.Tensor,
        batch_dia_len: List[int],
        modality_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """将节点特征分解回各模态"""
        num_modalities = len(modality_names)
        total_utterances = sum(batch_dia_len)
        
        # node_features: [total_utterances * num_modalities, D]
        # 按模态分组
        features_per_modality = torch.chunk(node_features, num_modalities, dim=0)
        
        updated_features = {}
        for i, name in enumerate(modality_names):
            updated_features[name] = features_per_modality[i]  # [total_utterances, D]
        
        return updated_features


# ==================== 6. 傅里叶图分解模块 (参考GS-MCC) ====================

class FourierGraphDecomposition(nn.Module):
    """
    傅里叶图分解模块 (参考GS-MCC)
    将特征分解为高频和低频分支
    """
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 4,
        sparsity_threshold: float = 0.01,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        
        # 傅里叶卷积权重
        # rfft会将D维变为D//2+1维
        self.frequency_size = embed_size // 2 + 1
        self.w_layers = nn.ParameterList([
            nn.Parameter(
                self.scale * torch.randn(2, self.frequency_size, self.frequency_size * hidden_size_factor)
            )
            for _ in range(num_layers)
        ])
        self.b_layers = nn.ParameterList([
            nn.Parameter(self.scale * torch.randn(2, self.frequency_size * hidden_size_factor))
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            low_freq: [B, D] 低频分量
            high_freq: [B, D] 高频分量
        """
        B, D = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, D//2+1]
        
        # 傅里叶图卷积
        x_complex = torch.stack([x_fft.real, x_fft.imag], dim=-1)  # [B, D//2+1, 2]
        
        for i in range(self.num_layers):
            x_real = x_complex[..., 0]
            x_imag = x_complex[..., 1]
            
            # 复数乘法
            o_real = F.relu(
                torch.einsum('bd,dd->bd', x_real, self.w_layers[i][0]) -
                torch.einsum('bd,dd->bd', x_imag, self.w_layers[i][1]) +
                self.b_layers[i][0]
            )
            o_imag = F.relu(
                torch.einsum('bd,dd->bd', x_imag, self.w_layers[i][0]) +
                torch.einsum('bd,dd->bd', x_real, self.w_layers[i][1]) +
                self.b_layers[i][1]
            )
            
            x_complex = torch.stack([o_real, o_imag], dim=-1)
            x_complex = F.softshrink(x_complex, lambd=self.sparsity_threshold)
        
        # 分离高低频
        freq_mag = torch.sqrt(x_complex[..., 0]**2 + x_complex[..., 1]**2)
        threshold = freq_mag.median(dim=-1, keepdim=True)[0]
        
        low_mask = (freq_mag <= threshold).float().unsqueeze(-1)
        high_mask = (freq_mag > threshold).float().unsqueeze(-1)
        
        low_freq_complex = x_complex * low_mask
        high_freq_complex = x_complex * high_mask
        
        # IFFT
        low_freq_fft = torch.complex(low_freq_complex[..., 0], low_freq_complex[..., 1])
        high_freq_fft = torch.complex(high_freq_complex[..., 0], high_freq_complex[..., 1])
        
        low_freq = torch.fft.irfft(low_freq_fft, n=D, dim=-1, norm='ortho')
        high_freq = torch.fft.irfft(high_freq_fft, n=D, dim=-1, norm='ortho')
        
        return low_freq, high_freq


# ==================== 7. 超球体正则化 ====================

class SphereRegularization(nn.Module):
    """超球体正则化层"""
    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            x_normalized: [B, D] 归一化到球面的特征
            sphere_loss: 标量，用于正则化
        """
        norms = x.norm(p=2, dim=-1)  # [B]
        sphere_loss = ((norms - self.radius) ** 2).mean()
        x_normalized = x / (norms.unsqueeze(-1) + 1e-6)
        return x_normalized, sphere_loss


# ==================== 8. 主模型 ====================

class MultimodalEmotionModel_Refactored(nn.Module):
    """
    多模态情感分析主模型 - 重构版
    集成所有新组件
    """
    def __init__(
        self,
        # 输入维度
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        text_global_dim: int,
        social_dim: int,  # 社会关系维度
        context_dim: int,  # 情境维度
        
        # 隐藏维度
        hidden_dim: int,
        
        # 模型配置
        model_config: ModelConfig,
        
        # 其他参数
        num_ism_layers: int = 2,
        num_coupled_layers: int = 2,
        num_labels: int = 7,
        fusion_hidden_dim: Optional[int] = None,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        
        self.config = model_config
        self.hidden_dim = hidden_dim
        self.modality_names = ["text", "audio", "video"]
        
        # 1. 关键帧选择 (MDP3) - 新的百分比模式
        if self.config.use_key_frame_selector:
            self.key_frame_selector = KeyFrameSelector(
                audio_dim=audio_input_dim,
                video_dim=video_input_dim,
                text_dim=text_global_dim,
                hidden_dim=hidden_dim,
                n_segments=self.config.n_segments,
                frame_ratio=self.config.frame_ratio,
                lamda=self.config.key_frame_lambda,
            )
        
        # 2. 输入投影层
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_input_dim, hidden_dim)
        self.video_proj = nn.Linear(video_input_dim, hidden_dim)
        
        # 先验特征投影层（将social和context也投影到hidden_dim）
        self.social_proj = nn.Linear(social_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # 2.5. 位置编码 (Sinusoidal Positional Encoding)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_dim,
            max_len=5000,  # 支持最长5000帧
            dropout=dropout_p,
        )
        
        # 2.6. 文本引导的软门控 (Text-Guided Gating)
        # 用全局文本特征来预筛选音频和视频帧
        self.text_guided_gate_audio = TextGuidedGating(
            text_dim=text_global_dim,  # 使用原始text_global维度
            target_dim=hidden_dim,
            dropout=dropout_p,
        )
        self.text_guided_gate_video = TextGuidedGating(
            text_dim=text_global_dim,
            target_dim=hidden_dim,
            dropout=dropout_p,
        )
        
        # 3. MoE-FiLM调制 (MoFME) - 社会关系和情境各自的
        if self.config.use_moe_film:
            # 为3个基本模态各创建2个MoE-FiLM（社会关系 + 情境）
            self.social_film_text = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=social_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
            self.social_film_audio = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=social_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
            self.social_film_video = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=social_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
            
            self.context_film_text = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=context_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
            self.context_film_audio = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=context_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
            self.context_film_video = MoE_FiLM_Modulation(
                feature_dim=hidden_dim,
                condition_dim=context_dim,
                num_experts=self.config.num_film_experts,
                top_k=self.config.film_top_k,
                dropout=dropout_p,
            )
        
        # 4. 时序建模 (Mamba / Coupled Mamba)
        if self.config.use_coupled_mamba:
            # Coupled Mamba (使用拼接的social+context作为prior)
            self.coupled_stack = CoupledMambaStack(
                modality_names=self.modality_names,
                hidden_dim=hidden_dim,
                prior_dim=social_dim + context_dim,
                num_layers=num_coupled_layers,
                dropout_p=dropout_p,
            )
        else:
            # 独立Mamba
            self.text_mamba = nn.ModuleList([
                MambaBlock(hidden_dim, dropout_p) for _ in range(num_ism_layers)
            ])
            self.audio_mamba = nn.ModuleList([
                MambaBlock(hidden_dim, dropout_p) for _ in range(num_ism_layers)
            ])
            self.video_mamba = nn.ModuleList([
                MambaBlock(hidden_dim, dropout_p) for _ in range(num_ism_layers)
            ])
        
        # 5. 注意力池化 (帧 → 语句)
        cond_dim = text_global_dim + social_dim + context_dim
        self.attn_pooling = nn.ModuleDict({
            name: AttentionPooling(
                input_dim=hidden_dim,
                cond_dim=cond_dim,
                dropout_p=dropout_p,
            ) for name in self.modality_names
        })
        
        # 6. 超图建模 (M3NET)
        if self.config.use_hypergraph:
            self.hypergraph = HypergraphModule(
                hidden_dim=hidden_dim,
                num_layers=self.config.num_hypergraph_layers,
                dropout=dropout_p,
            )
        
        # 7. 分类头
        if fusion_hidden_dim is None:
            fusion_hidden_dim = hidden_dim
        
        # ⭐ 根据配置决定融合几个模态
        if self.config.direct_fusion_priors:
            # 融合5个模态 (text, audio, video, social, context)
            num_modalities_for_fusion = 5
        else:
            # 只融合3个基础模态 (text, audio, video)，social/context只用于调制
            num_modalities_for_fusion = 3
        
        # ⭐ 根据配置选择MLP架构
        if self.config.use_improved_mlp:
            # 改进版：GELU激活
            self.modality_fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities_for_fusion, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
            )
        else:
            # 原始版：ReLU激活
            self.modality_fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities_for_fusion, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )
        
        # 8. 频域分解 (GS-MCC) - 在融合之后
        if self.config.use_frequency_decomp:
            self.freq_decomp = FourierGraphDecomposition(
                embed_size=fusion_hidden_dim,  # 使用融合后的维度
                hidden_size=fusion_hidden_dim,
                num_layers=self.config.num_fourier_layers,
            )
            # 频域融合（根据MLP类型选择激活函数）
            if self.config.use_improved_mlp:
                self.freq_fusion = nn.Sequential(
                    nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                    nn.GELU(),
                )
            else:
                self.freq_fusion = nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim)
        
        # 9. 超球体正则化
        if self.config.use_sphere_regularization:
            self.sphere_reg = SphereRegularization(radius=self.config.sphere_radius)
        
        # 10. 分类头（根据配置选择架构）
        if self.config.use_improved_mlp:
            # 改进版：4层深层MLP + GELU + 残差 + LayerNorm
            self.pre_classifier_mlp = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            )
            self.pre_classifier_norm = nn.LayerNorm(fusion_hidden_dim)
            
            self.classifier = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_p * 0.7),
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p * 0.5),
                nn.Linear(fusion_hidden_dim, num_labels),
            )
        else:
            # 原始版：2层简单MLP + ReLU
            self.pre_classifier_mlp = None
            self.pre_classifier_norm = None
            
            self.classifier = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(fusion_hidden_dim, num_labels),
            )
    
    def forward(
        self,
        text_sequence: torch.Tensor,   # [B, T, D_text]
        audio_sequence: torch.Tensor,  # [B, T, D_audio]
        video_sequence: torch.Tensor,  # [B, T, D_video]
        text_global: torch.Tensor,     # [B, D_text_global]
        social_embedding: torch.Tensor,  # [B, D_social]
        context_embedding: torch.Tensor,  # [B, D_context]
        mask: Optional[torch.Tensor] = None,
        batch_dia_len: Optional[List[int]] = None,  # 每个对话的utterance数
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        """
        batch_size = text_sequence.size(0)
        aux_outputs = {}
        
        # ====== 输入归一化 ⭐ 修复：使用安全归一化避免NaN ======
        text_global = safe_normalize(text_global, dim=-1)
        social_embedding = safe_normalize(social_embedding, dim=-1)
        context_embedding = safe_normalize(context_embedding, dim=-1)
        # =========================================================
        
        # ====== 阶段1: 关键帧选择（新：返回mask）======
        keyframe_mask = None  # 初始化mask
        if self.config.use_key_frame_selector:
            audio_sequence, video_sequence, selected_indices, keyframe_mask = self.key_frame_selector(
                audio_sequence, video_sequence, text_global
            )
            aux_outputs['selected_frame_indices'] = selected_indices
            aux_outputs['keyframe_mask'] = keyframe_mask  # ⭐ 保存mask用于调试
            
            # 同时对text_sequence进行相同的帧选择（包括padding）
            # selected_indices: [B, max_frames]（padding位置为-1）
            # keyframe_mask: [B, max_frames]（1=有效帧，0=padding）
            batch_size = text_sequence.size(0)
            max_frames = selected_indices.size(1)
            text_sequence_selected = []
            
            for b in range(batch_size):
                indices_b = selected_indices[b]  # [max_frames]
                # 只选择有效的索引（>= 0）
                valid_mask_b = indices_b >= 0
                valid_indices = indices_b[valid_mask_b]
                
                # 选择text帧
                text_b = text_sequence[b, valid_indices]  # [actual_len, D_text]
                
                # Padding到max_frames
                if text_b.size(0) < max_frames:
                    pad_len = max_frames - text_b.size(0)
                    pad_text = torch.zeros(pad_len, text_b.shape[-1], 
                                          device=text_b.device, dtype=text_b.dtype)
                    text_b = torch.cat([text_b, pad_text], dim=0)
                
                text_sequence_selected.append(text_b)
            
            text_sequence = torch.stack(text_sequence_selected)  # [B, max_frames, D_text]
            
            # ⭐ 关键：验证mask的正确性
            assert keyframe_mask is not None, "KeyFrameSelector必须返回mask!"
            assert keyframe_mask.shape[0] == batch_size, "Mask的batch维度错误"
            assert keyframe_mask.shape[1] == max_frames, "Mask的序列长度错误"
        
        # ====== 阶段2: 投影到统一空间 ======
        text_seq = self.text_proj(text_sequence)  # [B, T, H]
        audio_seq = self.audio_proj(audio_sequence)
        video_seq = self.video_proj(video_sequence)
        
        # ====== 阶段2.5: 位置编码 (Sinusoidal Positional Encoding) ======
        # 为三个模态添加位置信息
        text_seq = self.positional_encoding(text_seq)   # [B, T, H]
        audio_seq = self.positional_encoding(audio_seq) # [B, T, H]
        video_seq = self.positional_encoding(video_seq) # [B, T, H]
        
        # ====== 阶段2.6: 文本引导的软门控 (Text-Guided Gating) ======
        # 用全局文本特征来预筛选音频和视频帧
        # text_global 已经在前面做过归一化，直接使用
        audio_seq, gate_audio = self.text_guided_gate_audio(text_global, audio_seq)
        video_seq, gate_video = self.text_guided_gate_video(text_global, video_seq)
        
        # 保存门控信息用于分析
        aux_outputs['text_guided_gates'] = {
            'audio': gate_audio,  # [B, H]
            'video': gate_video,  # [B, H]
        }
        
        # ====== 阶段3: MoE-FiLM调制 ======
        if self.config.use_moe_film:
            # 社会关系调制
            text_seq, social_weights_text = self.social_film_text(text_seq, social_embedding)
            audio_seq, social_weights_audio = self.social_film_audio(audio_seq, social_embedding)
            video_seq, social_weights_video = self.social_film_video(video_seq, social_embedding)
            
            # 情境调制
            text_seq, context_weights_text = self.context_film_text(text_seq, context_embedding)
            audio_seq, context_weights_audio = self.context_film_audio(audio_seq, context_embedding)
            video_seq, context_weights_video = self.context_film_video(video_seq, context_embedding)
            
            aux_outputs['social_film_weights'] = {
                'text': social_weights_text,
                'audio': social_weights_audio,
                'video': social_weights_video,
            }
            aux_outputs['context_film_weights'] = {
                'text': context_weights_text,
                'audio': context_weights_audio,
                'video': context_weights_video,
            }
        
        # ====== 阶段4: 时序建模 (Coupled Mamba) ======
        prior_embedding = torch.cat([social_embedding, context_embedding], dim=-1)
        
        # ⭐ 使用关键帧的mask（如果有），否则使用传入的mask
        effective_mask = keyframe_mask if keyframe_mask is not None else mask
        
        if self.config.use_coupled_mamba:
            modality_sequences = {
                "text": text_seq,
                "audio": audio_seq,
                "video": video_seq,
            }
            coupled_sequences = self.coupled_stack(
                modality_sequences=modality_sequences,
                prior_embedding=prior_embedding,
                mask=effective_mask,  # ⭐ 使用有效的mask
            )
            text_seq = coupled_sequences["text"]
            audio_seq = coupled_sequences["audio"]
            video_seq = coupled_sequences["video"]
        else:
            # 独立Mamba
            for mamba_layer in self.text_mamba:
                text_seq = mamba_layer(text_seq, effective_mask)  # ⭐ 使用mask
            for mamba_layer in self.audio_mamba:
                audio_seq = mamba_layer(audio_seq, effective_mask)  # ⭐ 使用mask
            for mamba_layer in self.video_mamba:
                video_seq = mamba_layer(video_seq, effective_mask)  # ⭐ 使用mask
        
        # ====== 阶段5: 注意力池化 (帧 → 语句) ======
        cond_for_pool = torch.cat([text_global, social_embedding, context_embedding], dim=-1)
        
        # ⭐ 严格使用有效的mask（关键帧的mask，或原始mask）
        text_utt = self.attn_pooling["text"](text_seq, cond_for_pool, effective_mask)  # [B, H]
        audio_utt = self.attn_pooling["audio"](audio_seq, cond_for_pool, effective_mask)
        video_utt = self.attn_pooling["video"](video_seq, cond_for_pool, effective_mask)
        
        # ====== 阶段6: 超图建模 (M3NET) ======
        if self.config.use_hypergraph and batch_dia_len is not None:
            utterance_features = {
                "text": text_utt,
                "audio": audio_utt,
                "video": video_utt,
            }
            updated_features = self.hypergraph(utterance_features, batch_dia_len)
            text_utt = updated_features["text"]
            audio_utt = updated_features["audio"]
            video_utt = updated_features["video"]
        
        # ====== 阶段7: 融合模态 ======
        # 将social和context投影到hidden_dim（即使不用于融合，也保留投影用于分析）
        social_proj = self.social_proj(social_embedding)  # [B, H]
        context_proj = self.context_proj(context_embedding)  # [B, H]
        
        # ⭐ 根据配置决定融合哪些模态
        if self.config.direct_fusion_priors:
            # 融合5个模态: [text, audio, video, social, context]
            all_modalities = torch.cat([
                text_utt, audio_utt, video_utt, social_proj, context_proj
            ], dim=-1)  # [B, H*5]
        else:
            # 只融合3个基础模态: [text, audio, video]
            # social和context只通过FiLM调制影响特征，不直接参与融合
            all_modalities = torch.cat([
                text_utt, audio_utt, video_utt
            ], dim=-1)  # [B, H*3]
        
        fused_embedding = self.modality_fusion(all_modalities)  # [B, fusion_H]
        
        # ====== 阶段8: 频域分解 (GS-MCC) ======
        if self.config.use_frequency_decomp:
            low_freq, high_freq = self.freq_decomp(fused_embedding)
            freq_combined = torch.cat([low_freq, high_freq], dim=-1)
            fused_embedding = self.freq_fusion(freq_combined)
            aux_outputs['low_freq'] = low_freq
            aux_outputs['high_freq'] = high_freq
        
        # ====== 阶段9: 超球体正则化 ======
        sphere_loss = torch.tensor(0.0, device=fused_embedding.device)
        if self.config.use_sphere_regularization:
            fused_embedding, sphere_loss = self.sphere_reg(fused_embedding)
        
        # ====== 阶段10: Pre-Classifier（仅改进版MLP）======
        if self.config.use_improved_mlp:
            # 改进版：残差MLP + LayerNorm
            residual = fused_embedding
            fused_embedding = self.pre_classifier_mlp(fused_embedding)
            fused_embedding = self.pre_classifier_norm(residual + fused_embedding)
        
        # ====== 阶段11: 分类 ======
        logits = self.classifier(fused_embedding)  # [B, num_labels]
        
        aux_outputs.update({
            'fused_embedding': fused_embedding,
            'sphere_loss': sphere_loss,
            'text_embedding': text_utt,
            'audio_embedding': audio_utt,
            'video_embedding': video_utt,
            'social_embedding': social_proj,
            'context_embedding': context_proj,
        })
        
        return logits, aux_outputs


