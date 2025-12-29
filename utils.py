# -*- coding: utf-8 -*-
"""
工具函数模块
包含日志、指标计算、模型保存加载、绘图等功能
"""

import os
import json
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score
from scipy.stats import pearsonr, ConstantInputWarning, NearConstantInputWarning

# 尝试导入绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，支持无头服务器
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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
        # 注意：此函数用于通用回归任务（如MOSI/MOSEI），默认剔除中性样本
        # 对于CH-SIMS数据集，请使用 calc_chsims_metrics 函数
        non_zeros = np.abs(y_true) > 1e-6  # 剔除标签为0的样本（MOSI/MOSEI标准做法）
        if non_zeros.sum() > 0:
            y_pred_binary = (y_pred[non_zeros] > 0).astype(int)
            y_true_binary = (y_true[non_zeros] > 0).astype(int)
            acc_2 = accuracy_score(y_true_binary, y_pred_binary)
            f1_2 = f1_score(y_true_binary, y_pred_binary, average='weighted')
        else:
            acc_2 = 0.0
            f1_2 = 0.0
        
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
    def calc_chsims_metrics(y_pred: np.ndarray, y_true: np.ndarray, debug=False, 
                            exclude_zero_for_binary=False) -> Dict[str, float]:
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
            exclude_zero_for_binary: 是否在计算Acc-2时剔除中性样本（label=0）
                - False（默认）：CH-SIMS官方做法，使用 >=0 作为阈值，0归为正类
                - True：MOSI/MOSEI风格，剔除中性样本
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
            print(f"  Acc-2评估模式: {'MOSI风格(剔除中性)' if exclude_zero_for_binary else 'CH-SIMS风格(包含中性)'}")
        
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
        if exclude_zero_for_binary:
            # MOSI/MOSEI风格：剔除中性样本（标签为0的样本）
            # 使用 > 0 而不是 >= 0
            non_zeros = np.abs(y_true) > 1e-6
            if non_zeros.sum() > 0:
                y_pred_2 = (y_pred[non_zeros] > 0).astype(int)
                y_true_2 = (y_true[non_zeros] > 0).astype(int)
                acc_2 = accuracy_score(y_true_2, y_pred_2)
                f1_2 = f1_score(y_true_2, y_pred_2, average='weighted')
            else:
                acc_2 = 0.0
                f1_2 = 0.0
        else:
            # CH-SIMS官方风格：不剔除中性样本，>=0 归为正类
            y_pred_2 = (y_pred >= 0).astype(int)
            y_true_2 = (y_true >= 0).astype(int)
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
        # Weighted Accuracy: 按样本数量加权的准确率
        # = Σ(每个类别的准确率 × 该类别的样本占比)
        # 样本多的类别权重高，样本少的类别权重低
        # 数学上等价于 recall_score(average='weighted')
        weighted_acc = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        # F1_weighted: 按样本数量加权的F1（样本越多权重越高）
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Acc': weighted_acc,  # Weighted Accuracy: 按样本数量加权（样本多的类别权重高）
            'F1_micro': f1_micro,
            'F1_macro': f1_macro,
            'F1_weighted': f1_weighted,  # weighted F1
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
    
    @staticmethod
    def calc_iemocap_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        计算IEMOCAP数据集的4分类指标（包含每个类别的ACC和F1）
        
        IEMOCAP情感类别：
        - 0: neutral（中性）
        - 1: happy（快乐，包含excited）
        - 2: sad（悲伤）
        - 3: angry（愤怒）
        
        Args:
            y_pred: 预测类别 [N]
            y_true: 真实类别 [N]
        Returns:
            metrics: 包含总体指标和每个类别指标的字典
        """
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        
        # 情感类别名称
        emotion_names = ['neutral', 'happy', 'sad', 'angry']
        num_classes = 4
        
        # 总体指标
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ⭐ Balanced Accuracy: 各类别 Recall 的简单平均
        # 对于类别不平衡数据更公平，不受多数类主导
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        metrics = {
            'Acc': acc,
            'Balanced_Acc': balanced_acc,  # ⭐ 平衡准确率
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


class MetricsHistory:
    """用于记录训练过程中的指标历史"""
    
    def __init__(self):
        self.history = {
            'train': {},
            'valid': {},
            'test': {}
        }
    
    def update(self, split: str, epoch: int, metrics: Dict[str, float]):
        """
        更新指标历史
        
        Args:
            split: 数据集划分 ('train', 'valid', 'test')
            epoch: 当前epoch
            metrics: 指标字典
        """
        for key, value in metrics.items():
            if key not in self.history[split]:
                self.history[split][key] = []
            self.history[split][key].append((epoch, value))
    
    def get_metric_values(self, split: str, metric_name: str) -> tuple:
        """
        获取指定指标的历史值
        
        Returns:
            (epochs, values): epoch列表和对应的值列表
        """
        if metric_name not in self.history[split]:
            return [], []
        
        data = self.history[split][metric_name]
        epochs = [d[0] for d in data]
        values = [d[1] for d in data]
        return epochs, values
    
    def save_to_json(self, save_path: str):
        """保存历史记录到JSON文件"""
        # 转换为可序列化的格式
        serializable_history = {}
        for split, metrics in self.history.items():
            serializable_history[split] = {}
            for metric_name, data in metrics.items():
                serializable_history[split][metric_name] = {
                    'epochs': [d[0] for d in data],
                    'values': [float(d[1]) if not np.isnan(d[1]) else None for d in data]
                }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)


class TrainingPlotter:
    """训练过程绘图器"""
    
    # 指标配置：(显示名称, 是否越大越好)
    METRIC_CONFIG = {
        # 回归任务指标
        'MAE': ('MAE', False),
        'mae': ('MAE', False),
        'Acc_2': ('Acc-2 (Binary)', True),
        'acc_2': ('Acc-2 (Binary)', True),
        'Acc_3': ('Acc-3 (Ternary)', True),
        'acc_3': ('Acc-3 (Ternary)', True),
        'Acc_5': ('Acc-5 (5-Class)', True),
        'acc_5': ('Acc-5 (5-Class)', True),
        'loss': ('Loss', False),
        'Corr': ('Correlation', True),
        'corr': ('Correlation', True),
        'F1_2': ('F1-2 (Binary)', True),
        'F1_3': ('F1-3 (Ternary)', True),
        'F1_5': ('F1-5 (5-Class)', True),
        # 分类任务指标 (MELD/IEMOCAP)
        'Acc': ('Accuracy', True),
        'F1_weighted': ('F1 (Weighted)', True),
        'F1_macro': ('F1 (Macro)', True),
        'F1_micro': ('F1 (Micro)', True),
        'Balanced_Acc': ('Balanced Accuracy', True),
    }
    
    # 绘图样式配置
    SPLIT_STYLES = {
        'train': {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o', 'label': 'Train'},
        'valid': {'color': '#A23B72', 'linestyle': '--', 'marker': 's', 'label': 'Valid'},
        'test': {'color': '#F18F01', 'linestyle': '-.', 'marker': '^', 'label': 'Test'},
    }
    
    def __init__(self, save_dir: str, logger=None):
        """
        初始化绘图器
        
        Args:
            save_dir: 图片保存目录
            logger: 日志记录器（可选）
        """
        self.save_dir = save_dir
        self.logger = logger
        self.figures_dir = os.path.join(save_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def plot_metric(self, 
                    history: MetricsHistory, 
                    metric_name: str,
                    include_test: bool = True) -> Optional[str]:
        """
        绘制单个指标的训练曲线
        
        Args:
            history: 指标历史记录
            metric_name: 指标名称（如 'MAE', 'Acc_2', 'loss'）
            include_test: 是否包含测试集曲线
            
        Returns:
            保存的图片路径，如果绘制失败返回None
        """
        if not MATPLOTLIB_AVAILABLE:
            self._log("⚠️  matplotlib未安装，跳过绘图")
            return None
        
        # 获取指标配置
        display_name, higher_better = self.METRIC_CONFIG.get(
            metric_name, (metric_name, False)
        )
        
        # 收集数据
        splits_to_plot = ['train', 'valid']
        if include_test:
            splits_to_plot.append('test')
        
        has_data = False
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for split in splits_to_plot:
            epochs, values = history.get_metric_values(split, metric_name)
            if epochs and values:
                has_data = True
                style = self.SPLIT_STYLES[split]
                ax.plot(epochs, values, 
                       color=style['color'],
                       linestyle=style['linestyle'],
                       marker=style['marker'],
                       markersize=4,
                       linewidth=1.5,
                       label=style['label'],
                       alpha=0.8)
        
        if not has_data:
            plt.close(fig)
            return None
        
        # 设置图表样式
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(display_name, fontsize=12)
        ax.set_title(f'{display_name} vs Epoch', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景色
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # 添加最佳值标注
        for split in splits_to_plot:
            epochs, values = history.get_metric_values(split, metric_name)
            if epochs and values:
                valid_values = [(e, v) for e, v in zip(epochs, values) if not np.isnan(v)]
                if valid_values:
                    if higher_better:
                        best_idx = np.argmax([v for _, v in valid_values])
                    else:
                        best_idx = np.argmin([v for _, v in valid_values])
                    best_epoch, best_value = valid_values[best_idx]
                    style = self.SPLIT_STYLES[split]
                    ax.annotate(f'{best_value:.4f}',
                               xy=(best_epoch, best_value),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=style['color'],
                               fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.figures_dir, f'{metric_name.lower()}_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self._log(f"✓ 已保存 {display_name} 曲线图: {save_path}")
        return save_path
    
    def plot_all_metrics(self, 
                         history: MetricsHistory,
                         plot_config: Dict[str, bool],
                         include_test: bool = True) -> Dict[str, str]:
        """
        根据配置绘制所有指标
        
        Args:
            history: 指标历史记录
            plot_config: 绘图配置，如 {'mae': True, 'acc_2': True, ...}
            include_test: 是否包含测试集曲线
            
        Returns:
            保存的图片路径字典
        """
        if not MATPLOTLIB_AVAILABLE:
            self._log("⚠️  matplotlib未安装，跳过所有绘图")
            return {}
        
        saved_paths = {}
        
        # 指标名称映射（配置中的key -> 实际的指标名称）
        metric_mapping = {
            'mae': 'MAE',
            'acc_2': 'Acc_2',
            'acc_3': 'Acc_3',
            'acc_5': 'Acc_5',
            'loss': 'loss',
            'corr': 'Corr',
        }
        
        for config_key, should_plot in plot_config.items():
            if should_plot:
                metric_name = metric_mapping.get(config_key, config_key)
                save_path = self.plot_metric(history, metric_name, include_test)
                if save_path:
                    saved_paths[config_key] = save_path
        
        return saved_paths
    
    def plot_combined_figure(self,
                             history: MetricsHistory,
                             plot_config: Dict[str, bool],
                             include_test: bool = True) -> Optional[str]:
        """
        绘制组合图（所有指标在一个图中，使用子图）
        
        Args:
            history: 指标历史记录
            plot_config: 绘图配置
            include_test: 是否包含测试集曲线
            
        Returns:
            保存的图片路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # 筛选需要绘制的指标
        metrics_to_plot = [k for k, v in plot_config.items() if v]
        if not metrics_to_plot:
            return None
        
        # 指标名称映射
        metric_mapping = {
            'mae': 'MAE',
            'acc_2': 'Acc_2',
            'acc_3': 'Acc_3',
            'acc_5': 'Acc_5',
            'loss': 'loss',
            'corr': 'Corr',
        }
        
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        splits_to_plot = ['train', 'valid']
        if include_test:
            splits_to_plot.append('test')
        
        for idx, config_key in enumerate(metrics_to_plot):
            ax = axes[idx]
            metric_name = metric_mapping.get(config_key, config_key)
            display_name, _ = self.METRIC_CONFIG.get(metric_name, (metric_name, False))
            
            for split in splits_to_plot:
                epochs, values = history.get_metric_values(split, metric_name)
                if epochs and values:
                    style = self.SPLIT_STYLES[split]
                    ax.plot(epochs, values,
                           color=style['color'],
                           linestyle=style['linestyle'],
                           marker=style['marker'],
                           markersize=3,
                           linewidth=1.2,
                           label=style['label'],
                           alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(display_name, fontsize=10)
            ax.set_title(display_name, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, 'training_curves_combined.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self._log(f"✓ 已保存组合曲线图: {save_path}")
        return save_path

