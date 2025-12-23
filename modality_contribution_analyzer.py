# -*- coding: utf-8 -*-
"""
模态贡献度分析器
用于验证social和context模态的有效性，并量化它们的贡献
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict


class ModalityContributionAnalyzer:
    """
    模态贡献度分析器
    
    支持多种分析方法：
    1. 即时消融分析 - 对比有/无特定模态的预测差异
    2. 梯度重要性分析 - 计算loss对各模态的梯度范数
    3. 预测方差分析 - 添加扰动后的预测稳定性
    4. 特征激活分析 - 统计模态特征的激活分布
    """
    
    def __init__(self, modalities=['social', 'context']):
        """
        Args:
            modalities: 要分析的模态列表
        """
        self.modalities = modalities
        self.stats = defaultdict(lambda: defaultdict(list))
        
    def analyze_ablation(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        消融分析：对比有/无特定模态的性能差异
        
        Args:
            model: 模型
            batch: 输入batch
            criterion: 损失函数
            
        Returns:
            contribution_scores: 各模态的贡献分数
        """
        model.eval()
        device = next(model.parameters()).device
        
        # 准备输入
        text_seq = batch['text'].to(device)
        audio_seq = batch['audio'].to(device)
        video_seq = batch['vision'].to(device)
        text_global = batch['text_global'].to(device)
        social = batch['social'].to(device)
        context = batch['context'].to(device)
        labels = batch['label'].to(device)
        
        contribution_scores = {}
        
        with torch.no_grad():
            # 1. 完整模型预测
            logits_full, _ = model(
                text_sequence=text_seq,
                audio_sequence=audio_seq,
                video_sequence=video_seq,
                text_global=text_global,
                social_embedding=social,
                context_embedding=context,
            )
            loss_full = criterion(logits_full.squeeze(-1), labels.squeeze(-1))
            
            # 2. 分析social模态
            if 'social' in self.modalities:
                # 将social置零
                social_zero = torch.zeros_like(social)
                logits_no_social, _ = model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social_zero,
                    context_embedding=context,
                )
                loss_no_social = criterion(logits_no_social.squeeze(-1), labels.squeeze(-1))
                
                # 计算贡献：loss增加量和预测差异
                loss_increase_social = (loss_no_social - loss_full).item()
                pred_diff_social = (logits_full - logits_no_social).abs().mean().item()
                
                contribution_scores['social_loss_impact'] = loss_increase_social
                contribution_scores['social_pred_diff'] = pred_diff_social
            
            # 3. 分析context模态
            if 'context' in self.modalities:
                # 将context置零
                context_zero = torch.zeros_like(context)
                logits_no_context, _ = model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social,
                    context_embedding=context_zero,
                )
                loss_no_context = criterion(logits_no_context.squeeze(-1), labels.squeeze(-1))
                
                loss_increase_context = (loss_no_context - loss_full).item()
                pred_diff_context = (logits_full - logits_no_context).abs().mean().item()
                
                contribution_scores['context_loss_impact'] = loss_increase_context
                contribution_scores['context_pred_diff'] = pred_diff_context
            
            # 4. 分析两个模态同时移除
            if 'social' in self.modalities and 'context' in self.modalities:
                social_zero = torch.zeros_like(social)
                context_zero = torch.zeros_like(context)
                logits_no_both, _ = model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social_zero,
                    context_embedding=context_zero,
                )
                loss_no_both = criterion(logits_no_both.squeeze(-1), labels.squeeze(-1))
                
                loss_increase_both = (loss_no_both - loss_full).item()
                pred_diff_both = (logits_full - logits_no_both).abs().mean().item()
                
                contribution_scores['both_loss_impact'] = loss_increase_both
                contribution_scores['both_pred_diff'] = pred_diff_both
        
        return contribution_scores
    
    def analyze_gradient_importance(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        梯度重要性分析：计算loss对各模态的梯度范数
        
        Args:
            model: 模型
            batch: 输入batch
            criterion: 损失函数
            
        Returns:
            gradient_norms: 各模态的梯度范数
        """
        model.train()
        device = next(model.parameters()).device
        
        # 准备输入
        text_seq = batch['text'].to(device)
        audio_seq = batch['audio'].to(device)
        video_seq = batch['vision'].to(device)
        text_global = batch['text_global'].to(device)
        social = batch['social'].to(device).requires_grad_(True)
        context = batch['context'].to(device).requires_grad_(True)
        labels = batch['label'].to(device)
        
        # 前向传播
        logits, _ = model(
            text_sequence=text_seq,
            audio_sequence=audio_seq,
            video_sequence=video_seq,
            text_global=text_global,
            social_embedding=social,
            context_embedding=context,
        )
        
        # 计算loss
        loss = criterion(logits.squeeze(-1), labels.squeeze(-1))
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        gradient_norms = {}
        
        if social.grad is not None:
            social_grad_norm = social.grad.norm().item()
            social_grad_mean = social.grad.abs().mean().item()
            social_grad_max = social.grad.abs().max().item()
            gradient_norms['social_grad_norm'] = social_grad_norm
            gradient_norms['social_grad_mean'] = social_grad_mean
            gradient_norms['social_grad_max'] = social_grad_max
        
        if context.grad is not None:
            context_grad_norm = context.grad.norm().item()
            context_grad_mean = context.grad.abs().mean().item()
            context_grad_max = context.grad.abs().max().item()
            gradient_norms['context_grad_norm'] = context_grad_norm
            gradient_norms['context_grad_mean'] = context_grad_mean
            gradient_norms['context_grad_max'] = context_grad_max
        
        return gradient_norms
    
    def analyze_feature_statistics(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        特征统计分析：分析模态特征的分布特性
        
        Args:
            batch: 输入batch
            
        Returns:
            feature_stats: 特征统计信息
        """
        device = batch['social'].device
        social = batch['social']
        context = batch['context']
        
        feature_stats = {}
        
        # Social模态统计
        feature_stats['social_mean'] = social.mean().item()
        feature_stats['social_std'] = social.std().item()
        feature_stats['social_min'] = social.min().item()
        feature_stats['social_max'] = social.max().item()
        feature_stats['social_norm'] = social.norm(dim=-1).mean().item()
        
        # Context模态统计
        feature_stats['context_mean'] = context.mean().item()
        feature_stats['context_std'] = context.std().item()
        feature_stats['context_min'] = context.min().item()
        feature_stats['context_max'] = context.max().item()
        feature_stats['context_norm'] = context.norm(dim=-1).mean().item()
        
        # 与text特征对比（如果有）
        if 'text_global' in batch:
            text_global = batch['text_global']
            text_norm = text_global.norm(dim=-1).mean().item()
            
            feature_stats['social_vs_text_norm_ratio'] = feature_stats['social_norm'] / (text_norm + 1e-8)
            feature_stats['context_vs_text_norm_ratio'] = feature_stats['context_norm'] / (text_norm + 1e-8)
        
        return feature_stats
    
    def analyze_prediction_variance(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        noise_scale: float = 0.1,
        n_samples: int = 5,
    ) -> Dict[str, float]:
        """
        预测方差分析：添加噪声后的预测稳定性
        
        Args:
            model: 模型
            batch: 输入batch
            noise_scale: 噪声scale
            n_samples: 采样次数
            
        Returns:
            variance_scores: 方差分数
        """
        model.eval()
        device = next(model.parameters()).device
        
        # 准备输入
        text_seq = batch['text'].to(device)
        audio_seq = batch['audio'].to(device)
        video_seq = batch['vision'].to(device)
        text_global = batch['text_global'].to(device)
        social = batch['social'].to(device)
        context = batch['context'].to(device)
        
        variance_scores = {}
        
        with torch.no_grad():
            # 原始预测
            logits_orig, _ = model(
                text_sequence=text_seq,
                audio_sequence=audio_seq,
                video_sequence=video_seq,
                text_global=text_global,
                social_embedding=social,
                context_embedding=context,
            )
            
            # 添加噪声到social并采样
            social_predictions = []
            for _ in range(n_samples):
                social_noisy = social + torch.randn_like(social) * noise_scale
                logits, _ = model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social_noisy,
                    context_embedding=context,
                )
                social_predictions.append(logits)
            
            social_predictions = torch.stack(social_predictions)
            social_variance = social_predictions.var(dim=0).mean().item()
            social_stability = 1.0 / (social_variance + 1e-8)
            
            variance_scores['social_prediction_variance'] = social_variance
            variance_scores['social_stability_score'] = social_stability
            
            # 添加噪声到context并采样
            context_predictions = []
            for _ in range(n_samples):
                context_noisy = context + torch.randn_like(context) * noise_scale
                logits, _ = model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social,
                    context_embedding=context_noisy,
                )
                context_predictions.append(logits)
            
            context_predictions = torch.stack(context_predictions)
            context_variance = context_predictions.var(dim=0).mean().item()
            context_stability = 1.0 / (context_variance + 1e-8)
            
            variance_scores['context_prediction_variance'] = context_variance
            variance_scores['context_stability_score'] = context_stability
        
        return variance_scores
    
    def comprehensive_analysis(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        analyze_gradient: bool = True,
        analyze_variance: bool = False,  # 默认关闭，计算量较大
    ) -> Dict[str, float]:
        """
        综合分析：整合所有分析方法
        
        Args:
            model: 模型
            batch: 输入batch
            criterion: 损失函数
            analyze_gradient: 是否分析梯度
            analyze_variance: 是否分析预测方差
            
        Returns:
            all_scores: 所有分析结果
        """
        all_scores = {}
        
        # 1. 消融分析 (最重要)
        ablation_scores = self.analyze_ablation(model, batch, criterion)
        all_scores.update(ablation_scores)
        
        # 2. 特征统计 (开销小)
        feature_stats = self.analyze_feature_statistics(batch)
        all_scores.update(feature_stats)
        
        # 3. 梯度分析 (可选，需要backward)
        if analyze_gradient:
            try:
                gradient_norms = self.analyze_gradient_importance(model, batch, criterion)
                all_scores.update(gradient_norms)
            except Exception as e:
                print(f"Warning: Gradient analysis failed: {e}")
        
        # 4. 方差分析 (可选，计算量大)
        if analyze_variance:
            try:
                variance_scores = self.analyze_prediction_variance(model, batch)
                all_scores.update(variance_scores)
            except Exception as e:
                print(f"Warning: Variance analysis failed: {e}")
        
        return all_scores
    
    def format_analysis_results(self, scores: Dict[str, float]) -> str:
        """
        格式化分析结果为可读字符串
        
        Args:
            scores: 分析分数
            
        Returns:
            formatted_str: 格式化的字符串
        """
        lines = ["\n" + "=" * 60]
        lines.append("模态贡献度分析报告")
        lines.append("=" * 60)
        
        # 1. 消融分析结果
        if 'social_loss_impact' in scores or 'context_loss_impact' in scores:
            lines.append("\n【消融分析】移除模态后的性能变化:")
            lines.append("-" * 60)
            
            if 'social_loss_impact' in scores:
                loss_impact = scores['social_loss_impact']
                pred_diff = scores.get('social_pred_diff', 0)
                impact_str = "提升" if loss_impact > 0 else "下降"
                lines.append(f"  Social模态:")
                lines.append(f"    - Loss变化: {abs(loss_impact):.6f} ({impact_str})")
                lines.append(f"    - 预测差异: {pred_diff:.6f}")
                
                # 评估贡献度
                if loss_impact > 0.01:
                    contribution = "⭐️⭐️⭐️ 高贡献"
                elif loss_impact > 0.001:
                    contribution = "⭐️⭐️ 中等贡献"
                elif loss_impact > 0:
                    contribution = "⭐️ 低贡献"
                else:
                    contribution = "❌ 负面影响"
                lines.append(f"    - 贡献评级: {contribution}")
            
            if 'context_loss_impact' in scores:
                loss_impact = scores['context_loss_impact']
                pred_diff = scores.get('context_pred_diff', 0)
                impact_str = "提升" if loss_impact > 0 else "下降"
                lines.append(f"  Context模态:")
                lines.append(f"    - Loss变化: {abs(loss_impact):.6f} ({impact_str})")
                lines.append(f"    - 预测差异: {pred_diff:.6f}")
                
                if loss_impact > 0.01:
                    contribution = "⭐️⭐️⭐️ 高贡献"
                elif loss_impact > 0.001:
                    contribution = "⭐️⭐️ 中等贡献"
                elif loss_impact > 0:
                    contribution = "⭐️ 低贡献"
                else:
                    contribution = "❌ 负面影响"
                lines.append(f"    - 贡献评级: {contribution}")
            
            if 'both_loss_impact' in scores:
                loss_impact = scores['both_loss_impact']
                lines.append(f"  同时移除两个模态:")
                lines.append(f"    - Loss变化: {abs(loss_impact):.6f}")
                lines.append(f"    - 联合效应: {loss_impact:.6f}")
        
        # 2. 特征统计
        if 'social_norm' in scores or 'context_norm' in scores:
            lines.append("\n【特征统计】模态特征的分布特性:")
            lines.append("-" * 60)
            
            if 'social_norm' in scores:
                lines.append(f"  Social模态:")
                lines.append(f"    - 均值: {scores['social_mean']:.4f}")
                lines.append(f"    - 标准差: {scores['social_std']:.4f}")
                lines.append(f"    - 范围: [{scores['social_min']:.4f}, {scores['social_max']:.4f}]")
                lines.append(f"    - L2范数: {scores['social_norm']:.4f}")
            
            if 'context_norm' in scores:
                lines.append(f"  Context模态:")
                lines.append(f"    - 均值: {scores['context_mean']:.4f}")
                lines.append(f"    - 标准差: {scores['context_std']:.4f}")
                lines.append(f"    - 范围: [{scores['context_min']:.4f}, {scores['context_max']:.4f}]")
                lines.append(f"    - L2范数: {scores['context_norm']:.4f}")
        
        # 3. 梯度分析
        if 'social_grad_norm' in scores or 'context_grad_norm' in scores:
            lines.append("\n【梯度分析】模态对loss的敏感度:")
            lines.append("-" * 60)
            
            if 'social_grad_norm' in scores:
                lines.append(f"  Social模态梯度:")
                lines.append(f"    - 梯度范数: {scores['social_grad_norm']:.6f}")
                lines.append(f"    - 平均梯度: {scores['social_grad_mean']:.6f}")
            
            if 'context_grad_norm' in scores:
                lines.append(f"  Context模态梯度:")
                lines.append(f"    - 梯度范数: {scores['context_grad_norm']:.6f}")
                lines.append(f"    - 平均梯度: {scores['context_grad_mean']:.6f}")
        
        lines.append("=" * 60 + "\n")
        
        return "\n".join(lines)


# 便捷函数
def quick_analyze_modality_contribution(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
) -> str:
    """
    快速分析模态贡献度并返回格式化结果
    
    Args:
        model: 模型
        batch: 输入batch
        criterion: 损失函数
        
    Returns:
        formatted_report: 格式化的分析报告
    """
    analyzer = ModalityContributionAnalyzer(modalities=['social', 'context'])
    scores = analyzer.comprehensive_analysis(model, batch, criterion, analyze_gradient=False)
    report = analyzer.format_analysis_results(scores)
    return report


if __name__ == '__main__':
    print("模态贡献度分析器 - 使用示例")
    print("=" * 60)
    print("""
    # 在训练循环中使用:
    
    from modality_contribution_analyzer import ModalityContributionAnalyzer
    
    analyzer = ModalityContributionAnalyzer(modalities=['social', 'context'])
    
    # 每隔N个batch分析一次
    if batch_idx % 50 == 0:
        scores = analyzer.comprehensive_analysis(model, batch, criterion)
        report = analyzer.format_analysis_results(scores)
        logger.info(report)
    """)

