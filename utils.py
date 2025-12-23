# -*- coding: utf-8 -*-
"""
工具函数模块
包含日志、指标计算、模型保存加载等功能
"""

import os
import json
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, ConstantInputWarning, NearConstantInputWarning


def setup_seed(seed: int = 42):
    """
    设置随机种子以保证可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, log_name: str = 'train') -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        log_name: 日志名称
    Returns:
        logger: 日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
    
    # 创建 logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 创建文件 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def dict_to_str(d: Dict) -> str:
    """将字典转换为字符串用于打印"""
    return ', '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                      for k, v in d.items()])


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calc_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        计算回归任务的指标
        
        Args:
            y_pred: 预测值 [N]
            y_true: 真实值 [N]
        Returns:
            metrics: 包含各种指标的字典
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Pearson 相关系数
        corr, _ = pearsonr(y_pred, y_true)
        
        # 二分类指标（情感极性）
        y_pred_binary = (y_pred >= 0).astype(int)
        y_true_binary = (y_true >= 0).astype(int)
        acc_2 = accuracy_score(y_true_binary, y_pred_binary)
        f1_2 = f1_score(y_true_binary, y_pred_binary, average='weighted')
        
        # 七分类指标（细粒度情感）
        def to_7class(scores):
            """将连续得分转换为7类"""
            bins = [-3, -2, -1, 0, 1, 2, 3]
            classes = np.digitize(scores, bins) - 1
            return np.clip(classes, 0, 6)
        
        y_pred_7 = to_7class(y_pred)
        y_true_7 = to_7class(y_true)
        acc_7 = accuracy_score(y_true_7, y_pred_7)
        f1_7 = f1_score(y_true_7, y_pred_7, average='weighted')
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Corr': corr,
            'Acc_2': acc_2,
            'F1_2': f1_2,
            'Acc_7': acc_7,
            'F1_7': f1_7,
        }
    
    @staticmethod
    def calc_chsims_metrics(y_pred: np.ndarray, y_true: np.ndarray, debug=False) -> Dict[str, float]:
        """
        计算CH-SIMS特定的指标
        
        CH-SIMS标签区间：
        - Negative: {-1.0, -0.8}
        - Weakly negative: {-0.6, -0.4, -0.2}
        - Neutral: {0.0}
        - Weakly positive: {0.2, 0.4, 0.6}
        - Positive: {0.8, 1.0}
        
        Args:
            y_pred: 预测值 [N]
            y_true: 真实值 [N]
            debug: 是否打印调试信息
        Returns:
            metrics: 包含 MAE, Corr, Acc_2, Acc_3, Acc_5, F1 等指标
        """
        mae = mean_absolute_error(y_true, y_pred)
        
        # 调试：检查中性区间的样本数量
        if debug:
            neutral_true = np.sum((y_true >= -0.1) & (y_true <= 0.1))
            neutral_pred = np.sum((y_pred >= -0.1) & (y_pred <= 0.1))
            print(f"\n[DEBUG] 中性区间 [-0.1, 0.1] 样本统计:")
            print(f"  真实标签中的中性样本: {neutral_true}/{len(y_true)} ({neutral_true/len(y_true)*100:.1f}%)")
            print(f"  预测值中的中性样本: {neutral_pred}/{len(y_pred)} ({neutral_pred/len(y_pred)*100:.1f}%)")
            print(f"  预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
            print(f"  真实标签范围: [{y_true.min():.3f}, {y_true.max():.3f}]")
        
        # 计算相关系数，处理常量输入的情况
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                warnings.filterwarnings('ignore', category=NearConstantInputWarning)
                corr, _ = pearsonr(y_pred, y_true)
                # 如果结果是nan，设为0
                if np.isnan(corr):
                    corr = 0.0
        except Exception:
            corr = 0.0
        
        # 二分类（正/负）
        def to_2class(scores):
            return (scores >= 0).astype(int)
        
        y_pred_2 = to_2class(y_pred)
        y_true_2 = to_2class(y_true)
        acc_2 = accuracy_score(y_true_2, y_pred_2)
        f1_2 = f1_score(y_true_2, y_pred_2, average='weighted')
        
        # 三分类（负/中/正）
        def to_3class(scores):
            classes = np.zeros_like(scores, dtype=int)
            classes[scores < -0.1] = 0   # Negative
            classes[scores > 0.1] = 2    # Positive
            classes[(scores >= -0.1) & (scores <= 0.1)] = 1  # Neutral
            return classes
        
        y_pred_3 = to_3class(y_pred)
        y_true_3 = to_3class(y_true)
        acc_3 = accuracy_score(y_true_3, y_pred_3)
        f1_3 = f1_score(y_true_3, y_pred_3, average='weighted')
        
        # 五分类（按照CH-SIMS定义）
        def to_5class(scores):
            """
            Negative: {-1.0, -0.8} -> 0
            Weakly negative: {-0.6, -0.4, -0.2} -> 1
            Neutral: {0.0} -> 2
            Weakly positive: {0.2, 0.4, 0.6} -> 3
            Positive: {0.8, 1.0} -> 4
            """
            classes = np.zeros_like(scores, dtype=int)
            classes[scores <= -0.7] = 0  # Negative
            classes[(scores > -0.7) & (scores < -0.1)] = 1  # Weakly negative
            classes[(scores >= -0.1) & (scores <= 0.1)] = 2  # Neutral
            classes[(scores > 0.1) & (scores < 0.7)] = 3  # Weakly positive
            classes[scores >= 0.7] = 4  # Positive
            return classes
        
        y_pred_5 = to_5class(y_pred)
        y_true_5 = to_5class(y_true)
        acc_5 = accuracy_score(y_true_5, y_pred_5)
        f1_5 = f1_score(y_true_5, y_pred_5, average='weighted')
        
        return {
            'MAE': mae,
            'Corr': corr,
            'Acc_2': acc_2,
            'F1_2': f1_2,
            'Acc_3': acc_3,
            'F1_3': f1_3,
            'Acc_5': acc_5,
            'F1_5': f1_5,
        }
    
    @staticmethod
    def calc_classification_metrics(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> Dict[str, float]:
        """
        计算分类任务的指标
        
        Args:
            y_pred: 预测类别 [N]
            y_true: 真实类别 [N]
            num_classes: 类别数
        Returns:
            metrics: 包含各种指标的字典
        """
        acc = accuracy_score(y_true, y_pred)
        
        # 微平均和宏平均 F1
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'Acc': acc,
            'F1_micro': f1_micro,
            'F1_macro': f1_macro,
            'F1_weighted': f1_weighted,
        }
    
    @staticmethod
    def calc_meld_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        计算MELD数据集的7分类指标（包含每个类别的ACC和F1）
        
        MELD情感类别：
        - 0: neutral（中性）
        - 1: joy（喜悦）
        - 2: sadness（悲伤）
        - 3: anger（愤怒）
        - 4: surprise（惊讶）
        - 5: fear（恐惧）
        - 6: disgust（厌恶）
        
        Args:
            y_pred: 预测类别 [N]
            y_true: 真实类别 [N]
        Returns:
            metrics: 包含总体指标和每个类别指标的字典
        """
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        
        # 情感类别名称
        emotion_names = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
        num_classes = 7
        
        # 总体指标
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Acc': acc,
            'F1_micro': f1_micro,
            'F1_macro': f1_macro,
            'F1_weighted': f1_weighted,
        }
        
        # 计算每个类别的指标
        # 获取每个类别的精确率、召回率、F1
        precision_per_class = precision_score(y_true, y_pred, labels=range(num_classes), 
                                               average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=range(num_classes), 
                                         average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=range(num_classes), 
                                 average=None, zero_division=0)
        
        # 计算每个类别的准确率（需要用混淆矩阵）
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        # 每个类别的准确率 = 正确预测数 / 该类别的样本总数
        class_totals = cm.sum(axis=1)  # 每个类别的真实样本数
        class_correct = cm.diagonal()  # 每个类别正确预测的数量
        
        for i, name in enumerate(emotion_names):
            # 每个类别的准确率（如果该类别没有样本，设为0）
            if class_totals[i] > 0:
                acc_i = class_correct[i] / class_totals[i]
            else:
                acc_i = 0.0
            
            metrics[f'Acc_{name}'] = acc_i
            metrics[f'P_{name}'] = precision_per_class[i]
            metrics[f'R_{name}'] = recall_per_class[i]
            metrics[f'F1_{name}'] = f1_per_class[i]
            metrics[f'Support_{name}'] = int(class_totals[i])
        
        return metrics


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self, 
        patience: int = 10, 
        mode: str = 'min', 
        delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Args:
            patience: 容忍的 epoch 数
            mode: 'min' 表示指标越小越好，'max' 表示越大越好
            delta: 最小改进量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.is_better = lambda a, b: a < b - delta
        else:
            self.is_better = lambda a, b: a > b + delta
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            epoch: 当前 epoch
        Returns:
            should_stop: 是否应该停止训练
        """
        # 如果patience=0，禁用早停
        if self.patience <= 0:
            return False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.is_better(score, self.best_score):
            # 改进了
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'[EarlyStopping] Metric improved to {score:.4f}')
        else:
            # 没有改进
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] No improvement for {self.counter} epochs')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'[EarlyStopping] Early stopping triggered! Best epoch: {self.best_epoch}')
                return True
        
        return False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False,
):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前 epoch
        metrics: 指标字典
        save_path: 保存路径
        is_best: 是否是最佳模型
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config, save_path: str):
    """保存配置到 JSON 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)


def load_config_from_json(json_path: str, config_class):
    """从 JSON 文件加载配置"""
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = config_class()
    config.update(config_dict)
    return config


class AverageMeter:
    """用于跟踪和计算平均值的辅助类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

