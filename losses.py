# -*- coding: utf-8 -*-
"""
自定义损失函数模块
用于解决回归任务中的训练问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for Regression
    
    对难样本（误差大的样本）给予更高权重，有助于模型关注难以预测的样本。
    
    Args:
        gamma: 聚焦参数，越大越关注难样本。推荐范围：[1.0, 3.0]
    
    Reference:
        Inspired by Focal Loss (Lin et al., ICCV 2017)
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] or [B] 模型预测值
            target: [B, 1] or [B] 真实标签
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        if target.dim() == 2:
            target = target.squeeze(-1)
        
        # 计算基础MSE
        mse = (pred - target) ** 2
        
        # 计算focal权重：误差越大，权重越高
        error = torch.abs(pred - target)
        focal_weight = torch.pow(error, self.gamma)
        
        # 加权loss
        loss = (focal_weight * mse).mean()
        
        return loss


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    
    对异常值更鲁棒的损失函数。在误差小时使用L2，误差大时使用L1。
    
    Args:
        delta: 切换阈值，推荐范围：[0.5, 2.0]
    
    Reference:
        Huber, "Robust Estimation of a Location Parameter" (1964)
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] or [B] 模型预测值
            target: [B, 1] or [B] 真实标签
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        if target.dim() == 2:
            target = target.squeeze(-1)
        
        error = torch.abs(pred - target)
        
        # 小误差：使用L2
        is_small_error = error <= self.delta
        squared_loss = 0.5 * error ** 2
        
        # 大误差：使用L1
        linear_loss = self.delta * error - 0.5 * self.delta ** 2
        
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        
        return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss
    
    对不同情感强度区间的样本给予不同权重。
    特别关注边界附近（中性区域）和极端情感的样本。
    
    Args:
        neutral_weight: 中性区域[-0.15, 0.15]的权重
        extreme_weight: 极端情感(<-0.7 或 >0.7)的权重
        normal_weight: 其他区域的权重
    """
    def __init__(self, neutral_weight=2.0, extreme_weight=2.0, normal_weight=1.0):
        super().__init__()
        self.neutral_weight = neutral_weight
        self.extreme_weight = extreme_weight
        self.normal_weight = normal_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] or [B] 模型预测值
            target: [B, 1] or [B] 真实标签
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        if target.dim() == 2:
            target = target.squeeze(-1)
        
        # 计算MSE
        mse = (pred - target) ** 2
        
        # 计算权重
        weights = torch.ones_like(target) * self.normal_weight
        
        # 中性边界区域
        neutral_mask = (target >= -0.15) & (target <= 0.15)
        weights[neutral_mask] = self.neutral_weight
        
        # 极端情感区域
        extreme_mask = (target < -0.7) | (target > 0.7)
        weights[extreme_mask] = self.extreme_weight
        
        # 加权loss
        loss = (weights * mse).mean()
        
        return loss


class BalancedMSELoss(nn.Module):
    """
    Balanced MSE Loss
    
    根据正负样本的分布自动平衡权重，缓解类别不平衡问题。
    
    Args:
        smooth: 平滑因子，避免除零
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] or [B] 模型预测值
            target: [B, 1] or [B] 真实标签
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        if target.dim() == 2:
            target = target.squeeze(-1)
        
        # 分为正负两类
        pos_mask = target >= 0
        neg_mask = target < 0
        
        pos_count = pos_mask.sum().float() + self.smooth
        neg_count = neg_mask.sum().float() + self.smooth
        total = target.size(0)
        
        # 计算平衡权重
        pos_weight = total / (2 * pos_count)
        neg_weight = total / (2 * neg_count)
        
        # 应用权重
        mse = (pred - target) ** 2
        weights = torch.where(pos_mask, pos_weight, neg_weight)
        
        loss = (weights * mse).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合多个损失函数，平衡不同目标。
    例如：MSE + Huber + Range Penalty
    
    Args:
        mse_weight: MSE损失权重
        huber_weight: Huber损失权重
        range_weight: 范围惩罚权重（鼓励预测到整个[-1, 1]范围）
        huber_delta: Huber loss的delta参数
    """
    def __init__(self, mse_weight=1.0, huber_weight=0.5, range_weight=0.1, huber_delta=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.range_weight = range_weight
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = HuberLoss(delta=huber_delta)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] or [B] 模型预测值
            target: [B, 1] or [B] 真实标签
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(-1)
        if target.dim() == 2:
            target = target.squeeze(-1)
        
        # 1. MSE Loss
        mse = self.mse_loss(pred, target)
        
        # 2. Huber Loss
        huber = self.huber_loss(pred, target)
        
        # 3. Range Penalty: 鼓励预测值覆盖更大范围
        # 如果预测值都集中在很小的范围，给予惩罚
        pred_std = pred.std()
        target_std = target.std()
        range_penalty = F.relu(target_std - pred_std)  # 如果预测方差小于真实方差，惩罚
        
        # 组合loss
        total_loss = (
            self.mse_weight * mse +
            self.huber_weight * huber +
            self.range_weight * range_penalty
        )
        
        return total_loss


# 便捷的损失函数选择器
def get_loss_function(loss_type='mse', **kwargs):
    """
    根据名称获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - 'mse': 标准MSE Loss
            - 'l1' / 'mae': L1 Loss (MAE)
            - 'focal_mse': Focal MSE Loss
            - 'huber': Huber Loss
            - 'weighted_mse': Weighted MSE Loss
            - 'balanced_mse': Balanced MSE Loss
            - 'combined': Combined Loss
        **kwargs: 损失函数的参数
    
    Returns:
        损失函数实例
    
    Examples:
        >>> criterion = get_loss_function('l1')
        >>> criterion = get_loss_function('focal_mse', gamma=2.0)
        >>> criterion = get_loss_function('huber', delta=1.0)
        >>> criterion = get_loss_function('combined', mse_weight=1.0, huber_weight=0.5)
    """
    loss_dict = {
        'mse': lambda: nn.MSELoss(),
        'l1': lambda: nn.L1Loss(),
        'mae': lambda: nn.L1Loss(),  # 别名
        'focal_mse': lambda: FocalMSELoss(**kwargs),
        'huber': lambda: HuberLoss(**kwargs),
        'weighted_mse': lambda: WeightedMSELoss(**kwargs),
        'balanced_mse': lambda: BalancedMSELoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs),
    }
    
    if loss_type not in loss_dict:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_dict.keys())}")
    
    return loss_dict[loss_type]()


if __name__ == '__main__':
    # 测试损失函数
    print("=" * 60)
    print("测试自定义损失函数")
    print("=" * 60)
    
    # 模拟数据
    pred = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2])
    target = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    print(f"\n预测值: {pred}")
    print(f"真实值: {target}")
    
    # 测试各种损失
    losses = {
        'MSE': nn.MSELoss(),
        'Focal MSE (gamma=2.0)': FocalMSELoss(gamma=2.0),
        'Huber (delta=1.0)': HuberLoss(delta=1.0),
        'Weighted MSE': WeightedMSELoss(),
        'Balanced MSE': BalancedMSELoss(),
        'Combined': CombinedLoss(),
    }
    
    print("\n损失函数对比:")
    print("-" * 60)
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{name:30s}: {loss_value.item():.6f}")
    
    print("\n" + "=" * 60)

