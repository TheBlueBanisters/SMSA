# -*- coding: utf-8 -*-
"""
MLP组件备份 - 改进版本
备份时间: 2025-11-24
用途: 保存改进版MLP设计，方便后续恢复
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ==================== 改进版MLP组件 ====================

class ImprovedModalityFusion(nn.Module):
    """改进的模态融合层（GELU激活）"""
    def __init__(self, hidden_dim: int, fusion_hidden_dim: int, dropout_p: float = 0.5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, fusion_hidden_dim),
            nn.GELU(),  # ReLU→GELU
            nn.Dropout(dropout_p),
        )
    
    def forward(self, all_modalities):
        return self.fusion(all_modalities)


class ImprovedPreClassifier(nn.Module):
    """改进的Pre-Classifier（残差MLP + LayerNorm）"""
    def __init__(self, fusion_hidden_dim: int, dropout_p: float = 0.5):
        super().__init__()
        self.pre_classifier_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
        )
        self.pre_classifier_norm = nn.LayerNorm(fusion_hidden_dim)
    
    def forward(self, fused_embedding):
        residual = fused_embedding
        fused_embedding = self.pre_classifier_mlp(fused_embedding)
        fused_embedding = self.pre_classifier_norm(residual + fused_embedding)
        return fused_embedding


class ImprovedClassifier(nn.Module):
    """改进的分类器（4层深层MLP，GELU激活）"""
    def __init__(self, fusion_hidden_dim: int, num_labels: int, dropout_p: float = 0.5):
        super().__init__()
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
    
    def forward(self, fused_embedding):
        return self.classifier(fused_embedding)


# ==================== 改进版Forward逻辑片段 ====================

def improved_forward_snippet_init(fusion_hidden_dim, dropout_p):
    """
    在__init__中添加的组件（改进版）
    """
    # Pre-Classifier残差MLP
    pre_classifier_mlp = nn.Sequential(
        nn.Linear(fusion_hidden_dim, fusion_hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
    )
    # Pre-Classifier LayerNorm
    pre_classifier_norm = nn.LayerNorm(fusion_hidden_dim)
    
    # 改进版Classifier（4层深层MLP）
    classifier = nn.Sequential(
        nn.Linear(fusion_hidden_dim, fusion_hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout_p * 0.7),
        nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_p * 0.5),
        nn.Linear(fusion_hidden_dim, 1),
    )
    
    return pre_classifier_mlp, pre_classifier_norm, classifier


def improved_forward_snippet(fused_embedding, pre_classifier_mlp, pre_classifier_norm, classifier):
    """
    在forward中的处理逻辑（改进版）
    """
    # 残差MLP + LayerNorm
    residual = fused_embedding
    fused_embedding = pre_classifier_mlp(fused_embedding)
    fused_embedding = pre_classifier_norm(residual + fused_embedding)
    
    # 分类
    logits = classifier(fused_embedding)
    return logits


# ==================== 原始MLP组件（当前使用）====================

class OriginalModalityFusion(nn.Module):
    """原始的模态融合层（当前使用）"""
    def __init__(self, hidden_dim: int, fusion_hidden_dim: int, dropout_p: float = 0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
    
    def forward(self, all_modalities):
        return self.fusion(all_modalities)


class OriginalClassifier(nn.Module):
    """原始的分类器（当前使用）"""
    def __init__(self, fusion_hidden_dim: int, num_labels: int, dropout_p: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_hidden_dim, num_labels),
        )
    
    def forward(self, fused_embedding):
        return self.classifier(fused_embedding)


# ==================== 原始Forward逻辑片段 ====================

def original_forward_snippet(
    text_utt, audio_utt, video_utt, social_proj, context_proj,
    modality_fusion, classifier, freq_decomp=None, freq_fusion=None,
    sphere_reg=None, use_frequency_decomp=False, use_sphere_regularization=False
):
    """
    原始的forward逻辑（阶段7-10）
    
    用于对比和回退参考
    """
    # ====== 阶段7: 融合5个模态 ======
    all_modalities = torch.cat([
        text_utt, audio_utt, video_utt, social_proj, context_proj
    ], dim=-1)  # [B, H*5]
    
    fused_embedding = modality_fusion(all_modalities)  # [B, fusion_H]
    
    # ====== 阶段8: 频域分解 (GS-MCC) ======
    aux_outputs = {}
    if use_frequency_decomp:
        low_freq, high_freq = freq_decomp(fused_embedding)
        freq_combined = torch.cat([low_freq, high_freq], dim=-1)
        fused_embedding = freq_fusion(freq_combined)
        aux_outputs['low_freq'] = low_freq
        aux_outputs['high_freq'] = high_freq
    
    # ====== 阶段9: 超球体正则化 ======
    sphere_loss = torch.tensor(0.0, device=fused_embedding.device)
    if use_sphere_regularization:
        fused_embedding, sphere_loss = sphere_reg(fused_embedding)
    
    # ====== 阶段10: 分类 ======
    logits = classifier(fused_embedding)  # [B, num_labels]
    
    aux_outputs['sphere_loss'] = sphere_loss
    aux_outputs['fused_embedding'] = fused_embedding
    
    return logits, aux_outputs


# ==================== 恢复方法 ====================

def restore_original_mlp_to_model(model, hidden_dim, fusion_hidden_dim, num_labels, dropout_p):
    """
    将模型的MLP组件恢复到原始版本
    
    使用方法:
        from mlp_components_backup import restore_original_mlp_to_model
        restore_original_mlp_to_model(model, 256, 256, 1, 0.3)
    """
    model.modality_fusion = nn.Sequential(
        nn.Linear(hidden_dim * 5, fusion_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
    )
    
    model.classifier = nn.Sequential(
        nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(fusion_hidden_dim, num_labels),
    )
    
    print("✓ MLP组件已恢复到原始版本")


# ==================== 配置记录 ====================

ORIGINAL_CONFIG = {
    'name': '原始版本（当前使用）',
    'modality_fusion': {
        'structure': 'Linear(H*5 → fusion_H) → ReLU → Dropout',
        'activation': 'ReLU',
        'dropout': 0.3,
    },
    'classifier': {
        'structure': 'Linear(fusion_H → fusion_H) → ReLU → Dropout → Linear(fusion_H → 1)',
        'layers': 2,
        'hidden_dims': [256, 256],
        'activation': 'ReLU',
        'dropout': 0.3,
    },
    'pre_classifier': {
        'layernorm': False,
        'activation': None,
        'residual': False,
    },
    'freq_fusion': {
        'structure': 'Linear(fusion_H*2 → fusion_H)',
        'activation': None,
    },
    'regularization': {
        'dropout': 0.3,
        'weight_decay': 1e-4,
        'noise_scale': 0.05,
        'early_stop_patience': 10,
    },
}

IMPROVED_CONFIG = {
    'name': '改进版本（已备份）',
    'modality_fusion': {
        'structure': 'Linear(H*5 → fusion_H) → GELU → Dropout',
        'activation': 'GELU',
        'dropout': 0.5,
    },
    'pre_classifier': {
        'structure': 'ResidualMLP(256→512→256) + LayerNorm',
        'layernorm': True,
        'activation': 'GELU',
        'residual': True,
    },
    'classifier': {
        'structure': '256→512→512→256→1, 4层深层MLP',
        'layers': 4,
        'hidden_dims': [256, 512, 512, 256, 1],
        'activation': 'GELU',
        'dropout': [0.5, 0.35, 0.25],
    },
    'freq_fusion': {
        'structure': 'Linear(fusion_H*2 → fusion_H) → GELU',
        'activation': 'GELU',
    },
    'regularization': {
        'dropout': 0.5,
        'weight_decay': 5e-4,
        'noise_scale': 0.1,
        'early_stop_patience': 8,
    },
    'keyframe': {
        'n_segments': 4,
        'frame_ratio': 70,
    },
}

# ==================== 性能基线 ====================

ORIGINAL_PERFORMANCE = {
    'dataset': 'CH-SIMS',
    'training_date': '2025-11-24',
    'metrics': {
        'test_mae': 0.505,
        'test_corr': 0.598,
        'test_acc_2': 0.741,
        'test_f1_2': 0.740,
    },
    'overfitting': {
        'train_mae': 0.135,
        'valid_mae': 0.496,
        'gap': 0.361,  # 严重过拟合
    },
}

"""
备份说明：

1. 原始MLP特点：
   - 简单2层结构
   - ReLU激活
   - 无LayerNorm
   - 无残差连接
   - 中间层维度无扩展

2. 性能表现：
   - 测试集 MAE: 0.505
   - 测试集 Acc_2: 74.1%
   - 严重过拟合（训练/验证gap=0.36）

3. 恢复方法：
   - 使用 restore_original_mlp_to_model(model, 256, 256, 1, 0.3)
   - 或直接复制上面的代码到 smsa_refactored.py

4. 主要问题：
   - MLP容量不足（2层，无扩展）
   - 激活函数不够好（ReLU）
   - 缺少归一化（无LayerNorm）
   - 严重过拟合
"""



