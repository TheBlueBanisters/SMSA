# -*- coding: utf-8 -*-
"""
统一训练脚本 - 重构版
支持 chsims/chsimsv2/meld 数据集
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Dict

from config_refactored import get_config
from data_loader_refactored import create_dataloaders_refactored, create_dialogue_dataloaders
from smsa_refactored import MultimodalEmotionModel_Refactored
from utils import (
    setup_seed, setup_logger, dict_to_str,
    MetricsCalculator, EarlyStopping,
    save_checkpoint, count_parameters,
    save_config, AverageMeter,
    MetricsHistory, TrainingPlotter
)

# 尝试导入模态分析器（如果存在）
try:
    from modality_contribution_analyzer import ModalityContributionAnalyzer
    MODALITY_ANALYZER_AVAILABLE = True
except ImportError:
    MODALITY_ANALYZER_AVAILABLE = False


class Trainer:
    """统一训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 设置随机种子
        setup_seed(config.seed)
        
        # 设置日志
        self.logger = setup_logger(config.log_dir, 'train')
        self.logger.info(str(config))
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 设置指标记录文件
        self.metrics_file = getattr(config, 'metrics_file', None)
        
        # 初始化数据加载器
        self.setup_data()
        
        # 初始化模型
        self.setup_model()
        
        # 初始化优化器和调度器
        self.setup_optimizer()
        
        # 初始化损失函数
        self.setup_criterion()
        
        # 初始化指标计算器和早停
        self.metrics_calc = MetricsCalculator()
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience,
            mode=config.metric_mode,
            verbose=True,
        )
        
        self.best_metric = float('inf') if config.metric_mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # ====== 模态贡献度分析器 ======
        enable_analysis = getattr(config, 'enable_modality_analysis', False)
        if MODALITY_ANALYZER_AVAILABLE and enable_analysis:
            self.modality_analyzer = ModalityContributionAnalyzer(modalities=['social', 'context'])
            self.analyze_modality_every = getattr(config, 'analyze_modality_every', 10)
            self.modality_analysis_enabled = True
            self.modality_analysis_epochs = getattr(config, 'modality_analysis_epochs', 3)
            self.logger.info(f"✓ 模态分析器已启用 (每{self.analyze_modality_every}个batch分析)")
            self.logger.info(f"  ⚠️  注意：即时消融分析会影响训练速度")
        else:
            self.modality_analyzer = None
            self.modality_analysis_enabled = False
            if not enable_analysis:
                self.logger.info("ℹ️  模态分析已禁用（可通过 --enable_modality_analysis 启用）")
            elif not MODALITY_ANALYZER_AVAILABLE:
                self.logger.info("⚠️  模态分析器未安装（modality_contribution_analyzer.py）")
        # ==========================================
        
        # ====== 训练曲线绘图 ======
        self.metrics_history = MetricsHistory()
        
        # 检查是否启用绘图
        self.plotting_enabled = getattr(config, 'plotting_enabled', False)
        
        if self.plotting_enabled:
            # 根据任务类型构建绘图配置
            if config.task_type == 'regression':
                # 回归任务指标
                self.plot_config = {
                    'mae': getattr(config, 'plot_mae', True),
                    'loss': getattr(config, 'plot_loss', True),
                    'corr': getattr(config, 'plot_corr', True),
                    'acc_2': getattr(config, 'plot_acc2', True),
                    'acc_3': getattr(config, 'plot_acc3', True),
                    'acc_5': getattr(config, 'plot_acc5', True),
                }
            else:
                # 分类任务指标
                self.plot_config = {
                    'loss': getattr(config, 'plot_loss', True),
                    'Acc': getattr(config, 'plot_acc', True),
                    'F1_weighted': getattr(config, 'plot_f1_weighted', True),
                    'F1_macro': getattr(config, 'plot_f1_macro', True),
                }
            
            self.plotter = TrainingPlotter(config.save_dir, self.logger)
            enabled_plots = [k for k, v in self.plot_config.items() if v]
            self.logger.info(f"✓ 训练曲线绘图已启用: {', '.join(enabled_plots)}")
        else:
            self.plot_config = {}
            self.plotter = None
            self.logger.info("ℹ️  训练曲线绘图已禁用（可通过 --enable_plotting 启用）")
        # ==========================================
        
        # ====== 课程学习配置 ======
        self.curriculum_mode = getattr(config, 'curriculum_mode', 'none')
        self.curriculum_epochs = getattr(config, 'curriculum_epochs', 5)
        if self.curriculum_mode != 'none':
            self.logger.info(f"✓ 课程学习已启用: mode={self.curriculum_mode}, epochs={self.curriculum_epochs}")
        else:
            self.logger.info("ℹ️  课程学习已禁用（可通过 --curriculum_mode 启用）")
        # ==========================================
        
        # ====== 混合回放池 (Experience Replay) ======
        self.use_replay_buffer = getattr(config, 'use_replay_buffer', False)
        self.replay_buffer_threshold = getattr(config, 'replay_buffer_threshold', 1.5)  # loss阈值
        self.replay_buffer_ratio = getattr(config, 'replay_buffer_ratio', 0.2)  # 回放比例
        self.replay_buffer_max_size = getattr(config, 'replay_buffer_max_size', 500)  # 最大容量
        
        if self.use_replay_buffer:
            self.replay_buffer = []  # 存储高loss对话
            self.logger.info(f"✓ 混合回放池已启用:")
            self.logger.info(f"  - Loss阈值: {self.replay_buffer_threshold}")
            self.logger.info(f"  - 回放比例: {self.replay_buffer_ratio*100:.0f}%")
            self.logger.info(f"  - 最大容量: {self.replay_buffer_max_size}")
        else:
            self.replay_buffer = None
            self.logger.info("ℹ️  混合回放池已禁用（可通过 --use_replay_buffer 启用）")
        # ==========================================
    
    def setup_data(self):
        """设置数据加载器"""
        self.logger.info("Loading data...")
        self.logger.info(f"Dataset: {self.config.dataset_name}")
        
        # 检查是否使用对话级 batching（超图建模需要）
        use_dialogue_batching = getattr(self.config, 'use_dialogue_batching', False)
        
        if use_dialogue_batching:
            self.logger.info("⚠️  使用对话级 Batching（超图建模模式）")
            dialogue_batch_size = getattr(self.config, 'dialogue_batch_size', 8)
            max_dialogue_len = getattr(self.config.model_config, 'max_dialogue_len', 50)
            
            # 获取每批最大 utterance 数（控制显存）
            max_utterances_per_batch = getattr(self.config, 'max_utterances_per_batch', 128)
            
            self.train_loader, self.valid_loader, self.test_loader = create_dialogue_dataloaders(
                data_dir=self.config.data_dir,
                num_workers=self.config.num_workers,
                seq_length=self.config.seq_length,
                augment_train=self.config.augment_train,
                noise_scale=self.config.noise_scale,
                max_dialogue_len=max_dialogue_len,
                max_utterances_per_batch=max_utterances_per_batch,
            )
            
            self.logger.info(f"Train dialogues: {len(self.train_loader.dataset)}")
            self.logger.info(f"Max utterances per batch: {max_utterances_per_batch}")
            self.logger.info(f"Valid dialogues: {len(self.valid_loader.dataset)}")
            self.logger.info(f"Test dialogues: {len(self.test_loader.dataset)}")
            self.logger.info(f"Dialogue batch size: {dialogue_batch_size}")
            self.logger.info(f"Max dialogue len: {max_dialogue_len}")
        else:
            # 原始 utterance 级 batching
            self.train_loader, self.valid_loader, self.test_loader = create_dataloaders_refactored(
                data_dir=self.config.data_dir,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                seq_length=self.config.seq_length,
                augment_train=self.config.augment_train,
                noise_scale=self.config.noise_scale,
                cache_size=self.config.cache_size,
                use_weighted_sampler=getattr(self.config, 'use_weighted_sampler', False),
            )
            
            self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"Valid samples: {len(self.valid_loader.dataset)}")
            self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def setup_model(self):
        """设置模型"""
        self.logger.info("Initializing model...")
        
        self.model = MultimodalEmotionModel_Refactored(
            text_input_dim=self.config.text_input_dim,
            audio_input_dim=self.config.audio_input_dim,
            video_input_dim=self.config.video_input_dim,
            text_global_dim=self.config.text_global_dim,
            social_dim=self.config.social_dim,
            context_dim=self.config.context_dim,
            hidden_dim=self.config.hidden_dim,
            model_config=self.config.model_config,
            num_ism_layers=self.config.num_ism_layers,
            num_coupled_layers=self.config.num_coupled_layers,
            num_labels=self.config.num_labels,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            dropout_p=self.config.dropout_p,
        ).to(self.device)
        
        num_params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {num_params:,}")
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # 计算 warmup 步数
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.0)
        warmup_epochs = int(self.config.num_epochs * warmup_ratio)
        
        if self.config.scheduler_type == 'step':
            base_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type == 'cosine':
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - warmup_epochs,  # 余弦退火阶段的总长度
            )
        elif self.config.scheduler_type == 'reduce_on_plateau':
            # ReduceLROnPlateau 不支持 SequentialLR，单独处理
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.metric_mode,
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
            )
            if warmup_epochs > 0:
                self.logger.warning(f"⚠ ReduceLROnPlateau 不支持 Warmup，已忽略 warmup_ratio={warmup_ratio}")
            return
        else:
            self.scheduler = None
            return
        
        # 如果启用了 warmup，使用 SequentialLR 组合 warmup + 主调度器
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,  # 从 10% 学习率开始
                end_factor=1.0,    # 预热到 100% 学习率
                total_iters=warmup_epochs,
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs],
            )
            self.logger.info(f"✓ 启用学习率预热: {warmup_epochs} epochs (warmup_ratio={warmup_ratio})")
        else:
            self.scheduler = base_scheduler
    
    def setup_criterion(self):
        """设置损失函数"""
        if self.config.task_type == 'regression':
            # 尝试使用改进的损失函数
            loss_type = getattr(self.config, 'loss_function', 'mse')
            
            if loss_type == 'l1' or loss_type == 'mae':
                self.criterion = nn.L1Loss()
                self.logger.info("✓ 使用 L1 Loss (MAE)")
            elif loss_type == 'focal_mse':
                try:
                    from losses import FocalMSELoss
                    self.criterion = FocalMSELoss(gamma=2.0)
                    self.logger.info("✓ 使用 Focal MSE Loss (gamma=2.0)")
                except ImportError:
                    self.logger.warning("⚠️  losses.py不可用，使用标准MSE Loss")
                    self.criterion = nn.MSELoss()
            elif loss_type == 'huber':
                try:
                    from losses import HuberLoss
                    self.criterion = HuberLoss(delta=1.0)
                    self.logger.info("✓ 使用 Huber Loss (delta=1.0)")
                except ImportError:
                    self.logger.warning("⚠️  losses.py不可用，使用标准MSE Loss")
                    self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.MSELoss()
                self.logger.info("使用标准 MSE Loss")
        else:
            # ⭐ 分类任务：支持多种损失函数和类别权重
            loss_type = getattr(self.config, 'loss_function', 'ce')
            use_class_weights = getattr(self.config, 'use_class_weights', False)
            
            # 计算类别权重（如果需要）
            class_weights = None
            if use_class_weights and hasattr(self, 'train_loader'):
                class_counts = self._compute_class_distribution()
                if class_counts is not None:
                    total_samples = sum(class_counts)
                    num_classes = len(class_counts)
                    # 基础逆频率权重 (使用sqrt平滑，防止权重过大导致过拟合)
                    import math
                    weights = [math.sqrt(total_samples / (num_classes * max(c, 1))) for c in class_counts]
                    # 如果不希望平滑，可以使用原始公式：
                    # weights = [total_samples / (num_classes * max(c, 1)) for c in class_counts]
                    
                    class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                    self.logger.info(f"✓ 已计算类别权重(sqrt平滑): {[f'{w:.2f}' for w in weights]}")
                else:
                    self.logger.warning("⚠️  无法计算类别分布，将不使用自动类别权重")

            # 根据配置选择损失函数
            if loss_type == 'focal':
                try:
                    from losses import FocalLoss
                    # FocalLoss 可以接收 alpha (类别权重) 和 gamma
                    gamma = getattr(self.config, 'focal_gamma', 2.0)
                    label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
                    
                    # ⭐ 动态gamma参数
                    dynamic_gamma = getattr(self.config, 'focal_dynamic_gamma', False)
                    gamma_min = getattr(self.config, 'focal_gamma_min', 0.5)
                    gamma_decay_mode = getattr(self.config, 'focal_gamma_decay_mode', 'cosine')
                    
                    self.criterion = FocalLoss(
                        alpha=class_weights, 
                        gamma=gamma, 
                        label_smoothing=label_smoothing,
                        dynamic_gamma=dynamic_gamma,
                        gamma_min=gamma_min,
                        gamma_decay_mode=gamma_decay_mode
                    )
                    
                    if dynamic_gamma:
                        self.logger.info(f"✓ 使用 Dynamic Focal Loss (gamma: {gamma:.1f}→{gamma_min:.1f}, "
                                       f"mode={gamma_decay_mode}, weighted={class_weights is not None}, smooth={label_smoothing})")
                    else:
                        self.logger.info(f"✓ 使用 Focal Loss (gamma={gamma}, weighted={class_weights is not None}, smooth={label_smoothing})")
                except ImportError:
                    self.logger.warning("⚠️  losses.py不可用，回退到 CrossEntropyLoss")
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                # 默认为 CrossEntropyLoss
                label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
                if class_weights is not None:
                    self.logger.info(f"✓ 使用带权重的 CrossEntropyLoss (smooth={label_smoothing})")
                else:
                    self.logger.info(f"使用标准 CrossEntropyLoss (smooth={label_smoothing})")
        
        # ⭐ KL散度多任务学习：初始化KL损失函数
        use_kl_mtl = getattr(self.config, 'use_kl_mtl', False)
        if use_kl_mtl:
            try:
                from losses import MaskedKLDivLoss
                self.kl_criterion = MaskedKLDivLoss(reduction='batchmean')
                self.logger.info("✓ 启用 KL散度多任务学习 (GS-MCC风格)")
            except ImportError:
                self.logger.warning("⚠️  losses.py不可用，KL MTL将被禁用")
                self.config.use_kl_mtl = False
        else:
            self.kl_criterion = None
    
    def _compute_class_distribution(self):
        """计算训练集的类别分布"""
        try:
            num_classes = self.config.num_labels
            class_counts = [0] * num_classes
            
            # 遍历训练集统计各类别样本数
            for batch in self.train_loader:
                labels = batch['label']
                for label in labels:
                    label_idx = int(label.item())
                    if 0 <= label_idx < num_classes:
                        class_counts[label_idx] += 1
            
            return class_counts
        except Exception as e:
            self.logger.warning(f"计算类别分布时出错: {e}")
            return None
    
    def _apply_curriculum(self, epoch: int) -> None:
        """
        应用课程学习策略
        
        策略 A: freeze_backbone
            - 在课程学习期间冻结 Mamba Backbone，只训练 MoE/FiLM/Head
            - 课程结束后解冻所有参数
        
        策略 B: alpha_blending
            - 渐进式增加 MoE 的影响力
            - alpha = min(1.0, (epoch + 1) / curriculum_epochs)
            - 所有参数始终可训练
        
        ⭐ 新增：动态gamma衰减
            - 如果criterion是FocalLoss且启用dynamic_gamma，会自动更新gamma值
            - gamma从初始值逐渐衰减到gamma_min，平衡少数类和多数类
        
        Args:
            epoch: 当前 epoch 编号（从1开始）
        """
        # ⭐ 更新 Focal Loss 的 gamma（如果启用了动态gamma）
        if hasattr(self.criterion, 'dynamic_gamma') and self.criterion.dynamic_gamma:
            # epoch从1开始，但update_gamma期望从0开始的索引
            self.criterion.update_gamma(epoch - 1, self.config.num_epochs)
            self.logger.info(f"📉 Dynamic Focal Loss: gamma={self.criterion.gamma:.3f} "
                           f"(init={self.criterion.gamma_init:.1f}, min={self.criterion.gamma_min:.1f})")
        
        if self.curriculum_mode == 'none':
            return
        
        curriculum_epochs = self.curriculum_epochs
        
        if self.curriculum_mode == 'freeze_backbone':
            # ====== 策略 A: 冻结骨干网络 ======
            if epoch <= curriculum_epochs:
                # 冻结阶段：只训练调制模块和分类头
                frozen_count = 0
                trainable_count = 0
                
                # ⭐ 根据配置决定可训练的调制模块
                trainable_keys = [
                    'classifier', 'pre_classifier',  # 分类头
                    'modality_fusion',  # 融合层
                    'freq_fusion', 'freq_decomp',  # 频域分解
                ]
                
                # 如果启用了 MoE-FiLM，添加 FiLM 模块
                if self.config.model_config.use_moe_film:
                    trainable_keys.extend(['social_film', 'context_film'])
                
                # ⭐ 如果启用了 DSPS，添加 DSPS 投影层
                if self.config.model_config.use_dsps:
                    trainable_keys.append('dsps_proj')
                
                for name, param in self.model.named_parameters():
                    # 判断是否是可训练模块的参数
                    is_trainable_module = any(key in name for key in trainable_keys)
                    
                    if is_trainable_module:
                        param.requires_grad = True
                        trainable_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1
                
                self.logger.info(f"📚 Curriculum: Freezing Backbone (Epoch {epoch}/{curriculum_epochs})")
                self.logger.info(f"   Frozen params: {frozen_count}, Trainable params: {trainable_count}")
                self.logger.info(f"   Trainable modules: {trainable_keys}")
            else:
                # 解冻阶段：所有参数可训练
                for param in self.model.parameters():
                    param.requires_grad = True
                
                if epoch == curriculum_epochs + 1:
                    self.logger.info(f"📚 Curriculum: Unfreezing All Parameters (Epoch {epoch})")
                    self.logger.info(f"   All {sum(1 for _ in self.model.parameters())} parameters are now trainable")
        
        elif self.curriculum_mode == 'alpha_blending':
            # ====== 策略 B: 渐进式 Alpha 混合 ======
            # ⭐ 注意：alpha_blending 仅对 MoE-FiLM 有效
            # 当 MoE-FiLM 关闭时，此策略无效果（但不报错，只是警告）
            
            if not self.config.model_config.use_moe_film:
                if epoch == 1:
                    self.logger.warning(
                        f"⚠️  Curriculum alpha_blending 仅对 MoE-FiLM 有效，"
                        f"当前 MoE-FiLM 已关闭，此策略将不起作用"
                    )
            else:
                # 计算当前 alpha 值：从 1/curriculum_epochs 渐进到 1.0
                alpha = min(1.0, epoch / curriculum_epochs)
                
                # 调用模型的 set_moe_alpha 方法
                if hasattr(self.model, 'set_moe_alpha'):
                    self.model.set_moe_alpha(alpha)
                
                self.logger.info(f"📚 Curriculum: Setting MoE Alpha to {alpha:.4f} (Epoch {epoch}/{curriculum_epochs})")
            
            # 确保所有参数可训练
            for param in self.model.parameters():
                param.requires_grad = True
    
    def log_metrics_to_file(self, epoch: int, train_metrics: Dict[str, float], 
                           valid_metrics: Dict[str, float], test_metrics: Dict[str, float] = None):
        """记录每个epoch的指标到txt文件"""
        if self.metrics_file is None:
            return
        
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(f"{'='*60}\n")
                f.write(f"Epoch {epoch}/{self.config.num_epochs}\n")
                f.write(f"{'='*60}\n\n")
                
                # 记录训练集指标
                f.write("训练集 (Train):\n")
                for key, value in train_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # 记录验证集指标
                f.write("验证集 (Valid):\n")
                for key, value in valid_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # 记录测试集指标（如果有）
                if test_metrics is not None:
                    f.write("测试集 (Test):\n")
                    for key, value in test_metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                f.write("\n")
        except Exception as e:
            self.logger.warning(f"无法写入指标文件: {e}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        # ⭐ 应用课程学习策略（在每个 epoch 开始时）
        self._apply_curriculum(epoch)
        
        self.model.train()
        
        loss_meter = AverageMeter()
        sphere_loss_meter = AverageMeter()
        moe_loss_meter = AverageMeter()  # ⭐ MoE负载均衡损失计量器
        mtl_loss_meter = AverageMeter()  # ⭐ CH-SIMSv2 MTL损失计量器
        
        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch}/{self.config.num_epochs} [Train]',
                    position=0,
                    leave=True,
                    ncols=120,  # 固定宽度，避免显示问题
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')  # 优化格式
        
        all_preds = []
        all_labels = []
        
        # ====== 控制关键帧日志 ======
        if self.config.model_config.use_key_frame_selector:
            if hasattr(self.model, 'key_frame_selector') and self.model.key_frame_selector is not None:
                enable_kf_log = getattr(self.config, 'enable_keyframe_logging', False)
                kfs_log_freq = getattr(self.config, 'keyframe_log_every', 32)
                
                if epoch == 1:
                    # 第1个epoch显示配置信息
                    self.logger.info(f"\n{'='*70}")
                    self.logger.info(f"📊 关键帧选择器配置:")
                    self.logger.info(f"  use_key_frame_selector: {self.config.model_config.use_key_frame_selector}")
                    self.logger.info(f"  enable_keyframe_logging: {enable_kf_log}")
                    self.logger.info(f"  n_segments: {self.model.key_frame_selector.n_segments}")
                    self.logger.info(f"  frame_ratio: {self.model.key_frame_selector.frame_ratio}%")
                    
                    if enable_kf_log:
                        self.model.key_frame_selector.enable_logging = True
                        self.model.key_frame_selector.log_every = kfs_log_freq
                        self.model.key_frame_selector.logger = self.logger  # 传递logger
                        self.logger.info(f"  ✓ 关键帧统计已启用 (每{kfs_log_freq}个utterance打印)")
                    else:
                        self.model.key_frame_selector.enable_logging = False
                        self.logger.info(f"  ℹ️  关键帧统计已禁用")
                        self.logger.info(f"     启用方法: 修改train_unified.sh中的ENABLE_KEYFRAME_LOGGING=true")
                    self.logger.info(f"{'='*70}\n")
                else:
                    # 第2个epoch开始禁用日志
                    self.model.key_frame_selector.enable_logging = False
        else:
            if epoch == 1:
                self.logger.info("ℹ️  关键帧选择器已禁用")
        # ==========================================
        
        for batch_idx, batch in enumerate(pbar):
            
            # ====== 模态贡献度分析 ======
            if (self.modality_analysis_enabled and 
                self.modality_analyzer is not None and
                epoch <= self.modality_analysis_epochs and
                batch_idx % self.analyze_modality_every == 0 and
                batch_idx > 0):
                
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} - 模态贡献度分析")
                self.logger.info(f"{'='*70}")
                
                try:
                    analysis_batch = {
                        'text': batch['text'].to(self.device),
                        'audio': batch['audio'].to(self.device),
                        'vision': batch['vision'].to(self.device),
                        'text_global': batch['text_global'].to(self.device),
                        'social': batch['social'].to(self.device),
                        'context': batch['context'].to(self.device),
                        'label': batch['label'].to(self.device),
                    }
                    
                    scores = self.modality_analyzer.comprehensive_analysis(
                        self.model, analysis_batch, self.criterion, 
                        analyze_gradient=False, analyze_variance=False
                    )
                    report = self.modality_analyzer.format_analysis_results(scores)
                    self.logger.info(report)
                except Exception as e:
                    self.logger.warning(f"模态分析失败: {e}")
            # ==========================================
            
            # 将数据移到设备上
            text_seq = batch['text'].to(self.device)
            audio_seq = batch['audio'].to(self.device)
            video_seq = batch['vision'].to(self.device)
            text_global = batch['text_global'].to(self.device)
            social = batch['social'].to(self.device)
            context = batch['context'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # CH-SIMSv2 MTL: 读取单模态标签（如果启用）
            mtl_lambda = getattr(self.config, 'chsimsv2_mtl_lambda', 0.0)
            if mtl_lambda > 0 and 'label_T' in batch:
                labels_T = batch['label_T'].to(self.device)
                labels_A = batch['label_A'].to(self.device)
                labels_V = batch['label_V'].to(self.device)
            else:
                labels_T = labels_A = labels_V = None
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # ⭐ 获取 batch_dia_len：
            # - 对话级 batching 时：从 batch 中获取真实的对话长度列表
            # - utterance 级 batching 时：每个样本独立，使用 [1, 1, ...] 作为 fallback
            if 'batch_dia_len' in batch:
                # 对话级 batching：使用真实的对话长度
                batch_dia_len_for_hypergraph = batch['batch_dia_len']
            elif self.config.model_config.use_hypergraph:
                # utterance 级 batching + 超图：fallback 到 [1] * batch_size
                batch_dia_len_for_hypergraph = [1] * text_seq.size(0)
            else:
                batch_dia_len_for_hypergraph = None
            
            logits, aux_outputs = self.model(
                text_sequence=text_seq,
                audio_sequence=audio_seq,
                video_sequence=video_seq,
                text_global=text_global,
                social_embedding=social,
                context_embedding=context,
                batch_dia_len=batch_dia_len_for_hypergraph,
            )
            
            # 计算损失
            if self.config.task_type == 'regression':
                loss = self.criterion(logits.squeeze(-1), labels.squeeze(-1))
            else:
                loss = self.criterion(logits, labels.long().squeeze())
            
            # 添加超球面正则化损失（可能已弃用）
            sphere_loss = aux_outputs['sphere_loss']
            
            # ⭐ 添加MoE负载均衡损失（独立于sphere_loss）
            moe_loss = aux_outputs.get('moe_loss', torch.tensor(0.0, device=loss.device))
            moe_loss_weight = getattr(self.config, 'moe_loss_weight', 0.0)
            
            # ⭐ CH-SIMSv2 MTL: 计算单模态辅助损失
            mtl_loss = torch.tensor(0.0, device=loss.device)
            if mtl_lambda > 0 and labels_T is not None:
                # 从 aux_outputs 获取单模态表示（如果模型支持）
                # 目前先使用主预测作为所有模态的预测（简化版）
                # TODO: 未来可在模型中添加专门的单模态预测头
                loss_T = self.criterion(logits.squeeze(-1), labels_T.squeeze(-1))
                loss_A = self.criterion(logits.squeeze(-1), labels_A.squeeze(-1))
                loss_V = self.criterion(logits.squeeze(-1), labels_V.squeeze(-1))
                mtl_loss = (loss_T + loss_A + loss_V) / 3.0
            
            # ⭐ KL散度多任务学习 (GS-MCC): 计算单模态分类损失 + KL一致性损失
            kl_mtl_loss = torch.tensor(0.0, device=loss.device)
            use_kl_mtl = getattr(self.config, 'use_kl_mtl', False)
            if use_kl_mtl and self.kl_criterion is not None:
                # 获取单模态logits
                text_logits = aux_outputs.get('text_logits', None)
                audio_logits = aux_outputs.get('audio_logits', None)
                video_logits = aux_outputs.get('video_logits', None)
                
                if text_logits is not None and audio_logits is not None and video_logits is not None:
                    # 获取权重
                    kl_weight = getattr(self.config, 'kl_mtl_weight', 1.0)
                    unimodal_weight = getattr(self.config, 'unimodal_loss_weight', 1.0)
                    
                    # 1. 单模态分类损失（各模态独立预测真实标签）
                    labels_for_unimodal = labels.long().squeeze()
                    unimodal_loss = (
                        self.criterion(text_logits, labels_for_unimodal) +
                        self.criterion(audio_logits, labels_for_unimodal) +
                        self.criterion(video_logits, labels_for_unimodal)
                    )
                    
                    # 2. KL一致性损失（让单模态预测接近融合后的软标签）
                    # 融合后的softmax作为软标签（target）
                    soft_target = F.softmax(logits.detach(), dim=-1)  # [B, C]
                    
                    # 各模态的log_softmax
                    text_log_prob = F.log_softmax(text_logits, dim=-1)
                    audio_log_prob = F.log_softmax(audio_logits, dim=-1)
                    video_log_prob = F.log_softmax(video_logits, dim=-1)
                    
                    # KL散度: D_KL(P || Q) where P=soft_target, Q=unimodal_prob
                    kl_loss = (
                        self.kl_criterion(text_log_prob, soft_target) +
                        self.kl_criterion(audio_log_prob, soft_target) +
                        self.kl_criterion(video_log_prob, soft_target)
                    )
                    
                    kl_mtl_loss = unimodal_weight * unimodal_loss + kl_weight * kl_loss
            
            # 计算总损失
            total_loss = loss + self.config.sphere_loss_weight * sphere_loss + moe_loss_weight * moe_loss + mtl_lambda * mtl_loss + kl_mtl_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # 更新统计
            loss_meter.update(loss.item(), text_seq.size(0))
            sphere_loss_meter.update(sphere_loss.item(), text_seq.size(0))
            moe_loss_meter.update(moe_loss.item(), text_seq.size(0))  # ⭐ 更新MoE损失
            mtl_loss_meter.update(mtl_loss.item(), text_seq.size(0))  # ⭐ 更新MTL损失
            
            # 收集预测和标签
            all_preds.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # ====== 混合回放池：收集高loss对话 ======
            if self.use_replay_buffer and self.replay_buffer is not None:
                # 计算当前batch的平均loss（用于判断是否加入回放池）
                batch_loss = loss.item()
                
                # 动态阈值：使用当前epoch的平均loss的倍数
                dynamic_threshold = loss_meter.avg * self.replay_buffer_threshold if loss_meter.avg > 0 else self.replay_buffer_threshold
                
                if batch_loss > dynamic_threshold:
                    # 将当前batch加入回放池（存储必要信息）
                    replay_item = {
                        'text': text_seq.detach().cpu(),
                        'audio': audio_seq.detach().cpu(),
                        'vision': video_seq.detach().cpu(),
                        'text_global': text_global.detach().cpu(),
                        'social': social.detach().cpu(),
                        'context': context.detach().cpu(),
                        'label': labels.detach().cpu(),
                        'loss': batch_loss,
                    }
                    if 'batch_dia_len' in batch:
                        replay_item['batch_dia_len'] = batch['batch_dia_len']
                    
                    self.replay_buffer.append(replay_item)
                    
                    # 控制回放池大小（移除最老的）
                    if len(self.replay_buffer) > self.replay_buffer_max_size:
                        self.replay_buffer = self.replay_buffer[-self.replay_buffer_max_size:]
            # ==========================================
            
            # 更新进度条（包含MoE损失和MTL损失）
            postfix_dict = {
                'loss': f'{loss_meter.avg:.4f}',
                'moe': f'{moe_loss_meter.avg:.4f}',  # ⭐ 显示MoE损失
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            }
            if mtl_lambda > 0:
                postfix_dict['mtl'] = f'{mtl_loss_meter.avg:.4f}'  # ⭐ 显示MTL损失
            if self.use_replay_buffer and self.replay_buffer:
                postfix_dict['buf'] = f'{len(self.replay_buffer)}'  # ⭐ 显示回放池大小
            pbar.set_postfix(postfix_dict)
        
        # ====== 混合回放池：从回放池中采样训练 ======
        if self.use_replay_buffer and self.replay_buffer and len(self.replay_buffer) > 0:
            import random
            
            # 计算需要回放的batch数量
            num_replay_batches = max(1, int(len(self.train_loader) * self.replay_buffer_ratio))
            num_replay_batches = min(num_replay_batches, len(self.replay_buffer))
            
            # 按loss排序，优先选择高loss的样本
            sorted_buffer = sorted(self.replay_buffer, key=lambda x: x['loss'], reverse=True)
            replay_samples = sorted_buffer[:num_replay_batches]
            
            replay_loss_sum = 0.0
            replay_count = 0
            
            for replay_item in replay_samples:
                # 将数据移到设备
                text_seq = replay_item['text'].to(self.device)
                audio_seq = replay_item['audio'].to(self.device)
                video_seq = replay_item['vision'].to(self.device)
                text_global = replay_item['text_global'].to(self.device)
                social = replay_item['social'].to(self.device)
                context = replay_item['context'].to(self.device)
                labels = replay_item['label'].to(self.device)
                
                batch_dia_len = replay_item.get('batch_dia_len', None)
                
                # 前向传播
                self.optimizer.zero_grad()
                
                logits, aux_outputs = self.model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social,
                    context_embedding=context,
                    batch_dia_len=batch_dia_len,
                )
                
                # 计算损失
                if self.config.task_type == 'regression':
                    loss = self.criterion(logits.squeeze(-1), labels.squeeze(-1))
                else:
                    loss = self.criterion(logits, labels.long().squeeze())
                
                # 添加辅助损失
                sphere_loss = aux_outputs['sphere_loss']
                moe_loss = aux_outputs.get('moe_loss', torch.tensor(0.0, device=loss.device))
                moe_loss_weight = getattr(self.config, 'moe_loss_weight', 0.0)
                
                total_loss = loss + self.config.sphere_loss_weight * sphere_loss + moe_loss_weight * moe_loss
                
                # 反向传播
                total_loss.backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                
                replay_loss_sum += loss.item()
                replay_count += 1
                
                # 收集预测
                all_preds.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
            
            if replay_count > 0:
                self.logger.info(f"  📦 Replay: {replay_count} batches, avg_loss={replay_loss_sum/replay_count:.4f}")
            
            # 清理回放池中已改善的样本（loss降低到阈值以下的）
            current_avg_loss = loss_meter.avg
            self.replay_buffer = [
                item for item in self.replay_buffer 
                if item['loss'] > current_avg_loss * 0.8  # 保留loss仍然较高的样本
            ]
        # ==========================================
        
        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        if self.config.task_type == 'regression':
            if self.config.metrics_type == 'chsims':
                # 在第一个epoch启用调试输出
                debug_mode = (epoch == 1)
                metrics = self.metrics_calc.calc_chsims_metrics(all_preds.squeeze(), all_labels.squeeze(), debug=debug_mode)
            else:
                metrics = self.metrics_calc.calc_regression_metrics(all_preds.squeeze(), all_labels.squeeze())
        else:
            pred_classes = all_preds.argmax(axis=1)
            if self.config.metrics_type == 'meld':
                # MELD专用指标：包含每个情感类别的ACC/F1
                metrics = self.metrics_calc.calc_meld_metrics(pred_classes, all_labels.squeeze().astype(int))
            elif self.config.metrics_type == 'iemocap':
                # IEMOCAP专用指标：4分类情感，包含每个类别的ACC/F1
                metrics = self.metrics_calc.calc_iemocap_metrics(pred_classes, all_labels.squeeze().astype(int))
            else:
                metrics = self.metrics_calc.calc_classification_metrics(pred_classes, all_labels.squeeze(), self.config.num_labels)
        
        metrics['loss'] = loss_meter.avg
        metrics['sphere_loss'] = sphere_loss_meter.avg
        metrics['moe_loss'] = moe_loss_meter.avg  # ⭐ 记录MoE负载均衡损失
        
        return metrics
    
    def evaluate(self, dataloader, split='valid') -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, 
                       desc=f'{split.capitalize()} ', 
                       leave=False,
                       ncols=120,
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
            
            for batch in pbar:
                text_seq = batch['text'].to(self.device)
                audio_seq = batch['audio'].to(self.device)
                video_seq = batch['vision'].to(self.device)
                text_global = batch['text_global'].to(self.device)
                social = batch['social'].to(self.device)
                context = batch['context'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # ⭐ 获取 batch_dia_len：
                # - 对话级 batching 时：从 batch 中获取真实的对话长度列表
                # - utterance 级 batching 时：每个样本独立，使用 [1, 1, ...] 作为 fallback
                if 'batch_dia_len' in batch:
                    batch_dia_len_for_hypergraph = batch['batch_dia_len']
                elif self.config.model_config.use_hypergraph:
                    batch_dia_len_for_hypergraph = [1] * text_seq.size(0)
                else:
                    batch_dia_len_for_hypergraph = None
                
                logits, aux_outputs = self.model(
                    text_sequence=text_seq,
                    audio_sequence=audio_seq,
                    video_sequence=video_seq,
                    text_global=text_global,
                    social_embedding=social,
                    context_embedding=context,
                    batch_dia_len=batch_dia_len_for_hypergraph,
                )
                
                if self.config.task_type == 'regression':
                    loss = self.criterion(logits.squeeze(-1), labels.squeeze(-1))
                else:
                    loss = self.criterion(logits, labels.long().squeeze())
                
                loss_meter.update(loss.item(), text_seq.size(0))
                
                all_preds.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
        
        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        if self.config.task_type == 'regression':
            if self.config.metrics_type == 'chsims':
                # 在验证集第一次评估时启用调试输出
                debug_mode = (split == 'valid' and not hasattr(self, '_debug_done'))
                if debug_mode:
                    self._debug_done = True
                metrics = self.metrics_calc.calc_chsims_metrics(all_preds.squeeze(), all_labels.squeeze(), debug=debug_mode)
            else:
                metrics = self.metrics_calc.calc_regression_metrics(all_preds.squeeze(), all_labels.squeeze())
        else:
            pred_classes = all_preds.argmax(axis=1)
            if self.config.metrics_type == 'meld':
                # MELD专用指标：包含每个情感类别的ACC/F1
                metrics = self.metrics_calc.calc_meld_metrics(pred_classes, all_labels.squeeze().astype(int))
            elif self.config.metrics_type == 'iemocap':
                # IEMOCAP专用指标：4分类情感，包含每个类别的ACC/F1
                metrics = self.metrics_calc.calc_iemocap_metrics(pred_classes, all_labels.squeeze().astype(int))
            else:
                metrics = self.metrics_calc.calc_classification_metrics(pred_classes, all_labels.squeeze(), self.config.num_labels)
        
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            valid_metrics = self.evaluate(self.valid_loader, 'valid')
            
            # 日志
            self.logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            self.logger.info(f"Train: {dict_to_str(train_metrics)}")
            self.logger.info(f"Valid: {dict_to_str(valid_metrics)}")
            
            # 每个epoch后评估测试集（如果启用）
            test_metrics_epoch = None
            if getattr(self.config, 'eval_test_every_epoch', False):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Epoch {epoch} - 测试集评估（仅监控，不影响早停）")
                self.logger.info(f"{'='*60}")
                test_metrics_epoch = self.evaluate(self.test_loader, split='test')
                self.logger.info(f"Test: {dict_to_str(test_metrics_epoch)}")
                self.logger.info(f"{'='*60}\n")
            
            # 记录指标到txt文件
            self.log_metrics_to_file(epoch, train_metrics, valid_metrics, test_metrics_epoch)
            
            # 记录指标到历史（用于绘图）
            self.metrics_history.update('train', epoch, train_metrics)
            self.metrics_history.update('valid', epoch, valid_metrics)
            if test_metrics_epoch is not None:
                self.metrics_history.update('test', epoch, test_metrics_epoch)
            
            # 学习率调度
            if self.scheduler is not None:
                if self.config.scheduler_type == 'reduce_on_plateau':
                    monitor_metric = valid_metrics.get('mae', valid_metrics.get('loss'))
                    self.scheduler.step(monitor_metric)
                else:
                    self.scheduler.step()
            
            # 早停检查
            # 使用配置指定的指标，如果没有则回退到默认
            metric_name = getattr(self.config, 'early_stop_metric', 'mae')
            
            # ⭐ 新增：支持综合指标
            if metric_name == 'composite':
                # 计算综合分数（归一化后加权）
                mae_score = -valid_metrics.get('MAE', 1.0)  # 负值，因为越小越好
                corr_score = valid_metrics.get('Corr', 0.0)
                acc5_score = valid_metrics.get('Acc_5', 0.0)
                
                monitor_metric = (
                    self.config.composite_mae_weight * mae_score +
                    self.config.composite_corr_weight * corr_score +
                    self.config.composite_acc5_weight * acc5_score
                )
                
                self.logger.info(f"  Composite Score: {monitor_metric:.4f} "
                               f"(MAE={mae_score:.3f}, Corr={corr_score:.3f}, Acc5={acc5_score:.3f})")
            else:
                # 处理大小写不匹配：utils.py返回的键是Acc_2, F1_2等（首字母大写）
                metric_name_variants = [
                    metric_name,  # 原始：f1_weighted
                    metric_name.upper(),  # 全大写：F1_WEIGHTED
                    'Acc_2' if metric_name == 'acc_2' else metric_name,
                    'Acc_3' if metric_name == 'acc_3' else metric_name,
                    'F1_2' if metric_name == 'f1_2' else metric_name,
                    'F1_3' if metric_name == 'f1_3' else metric_name,
                    'F1_5' if metric_name == 'f1_5' else metric_name,
                    'Acc_5' if metric_name == 'acc_5' else metric_name,
                    'MAE' if metric_name == 'mae' else metric_name,
                    'Corr' if metric_name == 'corr' else metric_name,
                    'F1_weighted' if metric_name == 'f1_weighted' else metric_name,  # 分类任务
                    'F1_macro' if metric_name == 'f1_macro' else metric_name,
                    'F1_micro' if metric_name == 'f1_micro' else metric_name,
                    'Acc' if metric_name == 'acc' else metric_name,
                ]
                monitor_metric = None
                for variant in metric_name_variants:
                    if variant in valid_metrics:
                        monitor_metric = valid_metrics[variant]
                        break
                if monitor_metric is None:
                    monitor_metric = valid_metrics.get('MAE', valid_metrics.get('loss', 0.0))
            
            if self.config.metric_mode == 'min':
                is_best = monitor_metric < self.best_metric
            else:
                is_best = monitor_metric > self.best_metric
            
            if is_best:
                self.best_metric = monitor_metric
                self.best_epoch = epoch
                
                # 保存最佳模型
                save_path = os.path.join(self.config.save_dir, 'best_model.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=valid_metrics,
                    save_path=save_path,
                    is_best=True,
                )
                
                self.logger.info(f"✓ Best model saved! (metric: {self.best_metric:.4f})")
            
            # 早停
            self.early_stopping(monitor_metric, epoch)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # 测试
        self.logger.info("\nEvaluating on test set...")
        
        # 加载最佳模型
        checkpoint = torch.load(
            os.path.join(self.config.save_dir, 'best_model.pth'),
            map_location=self.device,
            weights_only=False  # PyTorch 2.6+ 需要此参数
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, 'test')
        self.logger.info(f"Test: {dict_to_str(test_metrics)}")
        
        self.logger.info(f"\nTraining completed! Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation metric: {self.best_metric:.4f}")
        
        # ====== 训练结束后绘制曲线图 ======
        if self.plotting_enabled and self.plotter is not None:
            self.logger.info("\n" + "="*60)
            self.logger.info("生成训练曲线图...")
            self.logger.info("="*60)
            
            # 判断是否包含测试集数据
            include_test = getattr(self.config, 'eval_test_every_epoch', False)
            
            # 绘制各个指标的独立图
            saved_paths = self.plotter.plot_all_metrics(
                self.metrics_history, 
                self.plot_config,
                include_test=include_test
            )
            
            # 绘制组合图
            combined_path = self.plotter.plot_combined_figure(
                self.metrics_history,
                self.plot_config,
                include_test=include_test
            )
            
            # 保存指标历史到JSON
            history_path = os.path.join(self.config.save_dir, 'metrics_history.json')
            self.metrics_history.save_to_json(history_path)
            self.logger.info(f"✓ 已保存指标历史: {history_path}")
            
            self.logger.info("="*60 + "\n")
        # ==========================================


def main():
    parser = argparse.ArgumentParser(description='统一训练脚本 - 重构版')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['chsims', 'chsimsv2', 'meld', 'iemocap'],
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小（可选，覆盖默认值）')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率（可选，覆盖默认值）')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数（可选，覆盖默认值）')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据集目录路径（覆盖默认路径）')
    parser.add_argument('--seq_length', type=int, default=None,
                        help='序列长度/最大帧数（建议：CH-SIMS/v2=70, MELD=80, IEMOCAP=110）')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='早停等待轮数（可选，0=禁用早停）')
    parser.add_argument('--early_stop_metric', type=str, default=None,
                        choices=['mae', 'loss', 'acc_2', 'acc_3', 'acc_5', 'f1_2', 'f1_3', 'f1_5', 'corr', 'composite'],
                        help='早停监控指标（composite为综合指标：0.4*MAE + 0.3*Corr + 0.3*Acc5）')
    parser.add_argument('--sphere_loss_weight', type=float, default=None,
                        help='超球体损失权重（可选，默认0.01）')
    parser.add_argument('--moe_loss_weight', type=float, default=None,
                        help='MoE负载均衡损失权重（可选，默认0.01，防止专家坍缩）')
    
    # CH-SIMSv2 MTL参数
    parser.add_argument('--chsimsv2_mtl_lambda', type=float, default=None,
                        help='CH-SIMSv2多任务学习的辅助损失权重（单模态标签），0.0=关闭')
    
    # 损失函数参数
    parser.add_argument('--loss_function', type=str, default=None,
                        choices=['mse', 'l1', 'mae', 'focal_mse', 'huber', 'focal', 'ce'],
                        help='损失函数类型')
    parser.add_argument('--focal_gamma', type=float, default=None,
                        help='Focal Loss gamma参数（初始值，如启用动态gamma）')
    parser.add_argument('--focal_dynamic_gamma', action='store_true',
                        help='启用动态gamma衰减（训练过程中gamma逐渐降低）')
    parser.add_argument('--focal_gamma_min', type=float, default=0.5,
                        help='动态gamma的最小值（默认0.5）')
    parser.add_argument('--focal_gamma_decay_mode', type=str, default='cosine',
                        choices=['linear', 'exponential', 'cosine', 'step'],
                        help='Gamma衰减模式（默认cosine）')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='启用类别权重')
    parser.add_argument('--no_class_weights', action='store_true',
                        help='禁用类别权重')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='Label Smoothing系数 (默认0.0)')

    # 采样参数
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='启用WeightedRandomSampler进行类别平衡采样')
    parser.add_argument('--no_weighted_sampler', action='store_true',
                        help='禁用WeightedRandomSampler')

    # ⭐ 课程学习参数
    parser.add_argument('--curriculum_mode', type=str, default=None,
                        choices=['none', 'freeze_backbone', 'alpha_blending'],
                        help='课程学习模式：none=关闭，freeze_backbone=冻结骨干网络，alpha_blending=渐进式MoE混合')
    parser.add_argument('--curriculum_epochs', type=int, default=None,
                        help='课程学习持续的Epoch数（默认：5）')
    
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='隐藏层维度（可选，默认256）')
    parser.add_argument('--dropout_p', type=float, default=None,
                        help='Dropout率（可选，默认0.1）')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler_type', type=str, default=None,
                        choices=['step', 'cosine', 'reduce_on_plateau', 'none'],
                        help='学习率调度器类型（可选，默认cosine）')
    parser.add_argument('--scheduler_gamma', type=float, default=None,
                        help='学习率衰减系数（默认0.5）')
    parser.add_argument('--scheduler_patience', type=int, default=None,
                        help='等待轮数（默认5，仅用于reduce_on_plateau）')
    parser.add_argument('--scheduler_step_size', type=int, default=None,
                        help='步长（默认10，仅用于step）')
    parser.add_argument('--warmup_ratio', type=float, default=None,
                        help='学习率预热比例（默认0.0关闭，0.1=前10%%步数用于预热）')
    
    # 组件参数
    parser.add_argument('--n_key_frames', type=int, default=None,
                        help='关键帧数量（可选，旧版参数，优先级低于n_segments/frame_ratio）')
    parser.add_argument('--key_frame_segment_size', type=int, default=None,
                        help='关键帧分段大小（可选，旧版参数，优先级低于n_segments/frame_ratio）')
    # 新的百分比模式参数
    parser.add_argument('--n_segments', type=int, default=None,
                        help='MDP3分段数（默认：config.model_config.n_segments）')
    parser.add_argument('--frame_ratio', type=int, default=None,
                        help='每段选择的帧百分比，1-100（默认：config.model_config.frame_ratio）')
    parser.add_argument('--num_film_experts', type=int, default=None,
                        help='MoE-FiLM专家数量（可选）')
    parser.add_argument('--film_top_k', type=int, default=None,
                        help='FiLM Top-K选择（可选）')
    parser.add_argument('--num_hypergraph_layers', type=int, default=None,
                        help='超图层数（可选）')
    parser.add_argument('--hypergraph_use_residue', action='store_true',
                        help='启用超图拼接残差（M3NET原始方式）')
    parser.add_argument('--hypergraph_no_residue', action='store_true',
                        help='禁用超图拼接残差')
    parser.add_argument('--num_fourier_layers', type=int, default=None,
                        help='傅里叶层数（可选）')
    parser.add_argument('--fgn_use_residue', action='store_true',
                        help='启用FGN拼接残差（GS-MCC原始方式）')
    parser.add_argument('--fgn_no_residue', action='store_true',
                        help='禁用FGN拼接残差')
    parser.add_argument('--fourier_sparsity_threshold', type=float, default=None,
                        help='傅里叶稀疏阈值（默认0.01）')
    parser.add_argument('--fourier_hidden_size_factor', type=int, default=None,
                        help='傅里叶隐藏层倍数（默认1）')
    
    # 混合回放池参数 ⭐ 新增
    parser.add_argument('--use_replay_buffer', action='store_true',
                        help='启用混合回放池（Experience Replay）')
    parser.add_argument('--no_replay_buffer', action='store_true',
                        help='禁用混合回放池')
    parser.add_argument('--replay_buffer_threshold', type=float, default=1.5,
                        help='回放池Loss阈值倍数（默认1.5，即avg_loss*1.5）')
    parser.add_argument('--replay_buffer_ratio', type=float, default=0.2,
                        help='回放训练比例（默认0.2，即额外训练20%%的batch）')
    parser.add_argument('--replay_buffer_max_size', type=int, default=500,
                        help='回放池最大容量（默认500）')
    
    # 模态和关键帧分析参数 ⭐ 新增
    parser.add_argument('--enable_modality_analysis', action='store_true',
                        help='启用模态贡献度分析（会影响训练速度）')
    parser.add_argument('--analyze_modality_every', type=int, default=None,
                        help='每N个batch进行模态分析（默认：10，适应小数据集）')
    parser.add_argument('--keyframe_log_every', type=int, default=None,
                        help='每N个utterance打印关键帧统计（默认：32）')
    parser.add_argument('--modality_analysis_epochs', type=int, default=None,
                        help='在前N个epoch进行模态分析（默认：3）')
    parser.add_argument('--enable_keyframe_logging', action='store_true',
                        help='启用关键帧统计打印')
    parser.add_argument('--eval_test_every_epoch', action='store_true',
                        help='每个epoch后在测试集上评估（仅监控，不影响早停）')
    
    # 组件开关（用于消融实验）
    parser.add_argument('--no_key_frame_selector', action='store_true',
                        help='禁用关键帧选择')
    parser.add_argument('--no_coupled_mamba', action='store_true',
                        help='禁用Coupled Mamba（使用独立Mamba）')
    parser.add_argument('--no_moe_film', action='store_true',
                        help='禁用MoE-FiLM调制')
    parser.add_argument('--no_hypergraph', action='store_true',
                        help='禁用超图建模')
    parser.add_argument('--no_frequency_decomp', action='store_true',
                        help='禁用频域分解')
    parser.add_argument('--no_sphere_reg', action='store_true',
                        help='禁用超球体正则化')
    
    # 组件开关（开启类）- 用于开启默认关闭的组件
    parser.add_argument('--use_dsps', action='store_true',
                        help='启用DSPS条件化SSM（在Mamba的dt/B/C路径注入条件）⚠️ 与MoE-FiLM互斥')
    parser.add_argument('--dsps_strength', type=float, default=0.1,
                        help='DSPS强度因子（0.0=无效果，1.0=完全效果，默认0.1）')
    parser.add_argument('--use_hypergraph', action='store_true',
                        help='启用超图建模（默认关闭）⚠️ 自动启用对话级batching')
    parser.add_argument('--use_frequency_decomp', action='store_true',
                        help='启用频域分解（默认关闭）')
    
    # ⭐ KL散度多任务学习 (GS-MCC)
    parser.add_argument('--use_kl_mtl', action='store_true',
                        help='启用KL散度多任务学习（让单模态预测接近融合预测）')
    parser.add_argument('--kl_mtl_weight', type=float, default=1.0,
                        help='KL散度损失的权重（默认1.0）')
    parser.add_argument('--unimodal_loss_weight', type=float, default=1.0,
                        help='单模态分类损失的权重（默认1.0）')
    parser.add_argument('--use_sphere_reg', action='store_true',
                        help='启用超球体正则化（默认关闭）')
    
    # 对话级 batching 参数（超图建模需要）
    parser.add_argument('--dialogue_batch_size', type=int, default=8,
                        help='[已弃用] 使用 --max_utterances_per_batch 代替')
    parser.add_argument('--max_dialogue_len', type=int, default=50,
                        help='单个对话最大utterance数量（默认50，超过的对话被跳过）')
    parser.add_argument('--max_utterances_per_batch', type=int, default=128,
                        help='每批最大utterance数（控制显存，默认128）')
    parser.add_argument('--no_direct_fusion_priors', action='store_true',
                        help='禁止social/context直接参与融合（只用于FiLM调制）')
    parser.add_argument('--use_improved_mlp', action='store_true',
                        help='使用改进版MLP（4层深层+GELU+残差+LayerNorm）')
    parser.add_argument('--mlp_dropout', type=float, default=0.2,
                        help='改进版MLP的Dropout比例（默认0.2）')
    parser.add_argument('--mlp_expansion_ratio', type=int, default=4,
                        help='改进版MLP中间层扩维倍数（默认4）')
    
    # 指标记录文件
    parser.add_argument('--metrics_file', type=str, default=None,
                        help='指标记录文件路径（txt文件）')
    
    # 多GPU支持：自定义保存目录
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录（用于多GPU并行训练避免冲突）')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志目录（用于多GPU并行训练避免冲突）')
    
    # ====== 训练曲线绘图开关（简化版）======
    parser.add_argument('--enable_plotting', action='store_true',
                        help='启用训练曲线绘图（自动根据数据集类型选择合适的指标）')
    parser.add_argument('--disable_plotting', action='store_true',
                        help='禁用训练曲线绘图')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.dataset)
    
    # 覆盖基础参数
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.seq_length is not None:
        config.seq_length = args.seq_length
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.early_stop_metric is not None:
        config.early_stop_metric = args.early_stop_metric
    if args.sphere_loss_weight is not None:
        config.sphere_loss_weight = args.sphere_loss_weight
    if args.moe_loss_weight is not None:
        config.moe_loss_weight = args.moe_loss_weight
    
    # CH-SIMSv2 MTL参数
    if args.chsimsv2_mtl_lambda is not None:
        if hasattr(config, 'chsimsv2_mtl_lambda'):
            config.chsimsv2_mtl_lambda = args.chsimsv2_mtl_lambda

    # 损失函数参数覆盖
    if args.loss_function is not None:
        config.loss_function = args.loss_function
    if args.focal_gamma is not None:
        config.focal_gamma = args.focal_gamma
    if args.focal_dynamic_gamma:
        config.focal_dynamic_gamma = True
        config.focal_gamma_min = args.focal_gamma_min
        config.focal_gamma_decay_mode = args.focal_gamma_decay_mode
    if args.use_class_weights:
        config.use_class_weights = True
    if args.no_class_weights:
        config.use_class_weights = False
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
    
    # 采样参数
    if args.use_weighted_sampler:
        config.use_weighted_sampler = True
    if args.no_weighted_sampler:
        config.use_weighted_sampler = False
    
    # ⭐ 课程学习参数覆盖
    if args.curriculum_mode is not None:
        config.curriculum_mode = args.curriculum_mode
    if args.curriculum_epochs is not None:
        config.curriculum_epochs = args.curriculum_epochs
    
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.dropout_p is not None:
        config.dropout_p = args.dropout_p
    
    # 学习率调度器参数
    if args.scheduler_type is not None:
        config.scheduler_type = args.scheduler_type
    if args.scheduler_gamma is not None:
        config.scheduler_gamma = args.scheduler_gamma
    if args.scheduler_patience is not None:
        config.scheduler_patience = args.scheduler_patience
    if args.scheduler_step_size is not None:
        config.scheduler_step_size = args.scheduler_step_size
    if args.warmup_ratio is not None:
        config.warmup_ratio = args.warmup_ratio
    
    # 模态和关键帧分析参数
    if args.enable_modality_analysis:
        config.enable_modality_analysis = True
    if args.analyze_modality_every is not None:
        config.analyze_modality_every = args.analyze_modality_every
    if args.keyframe_log_every is not None:
        config.keyframe_log_every = args.keyframe_log_every
    if args.modality_analysis_epochs is not None:
        config.modality_analysis_epochs = args.modality_analysis_epochs
    if args.enable_keyframe_logging:
        config.enable_keyframe_logging = True
    if args.eval_test_every_epoch:
        config.eval_test_every_epoch = True
    
    # 应用组件参数
    if args.n_segments is not None:
        config.model_config.n_segments = max(1, args.n_segments)
    if args.frame_ratio is not None:
        if not (1 <= args.frame_ratio <= 100):
            raise ValueError("--frame_ratio 必须在1到100之间")
        config.model_config.frame_ratio = args.frame_ratio
    # 兼容旧参数：若用户仍然使用n_key_frames/key_frame_segment_size，则回退到固定帧数模式
    if args.n_key_frames is not None:
        config.model_config.n_key_frames = args.n_key_frames
    if args.key_frame_segment_size is not None:
        config.model_config.key_frame_segment_size = args.key_frame_segment_size
    if args.num_film_experts is not None:
        config.model_config.num_film_experts = args.num_film_experts
    if args.film_top_k is not None:
        config.model_config.film_top_k = args.film_top_k
    if args.num_hypergraph_layers is not None:
        config.model_config.num_hypergraph_layers = args.num_hypergraph_layers
    if args.hypergraph_use_residue:
        config.model_config.hypergraph_use_residue = True
    if args.hypergraph_no_residue:
        config.model_config.hypergraph_use_residue = False
    if args.num_fourier_layers is not None:
        config.model_config.num_fourier_layers = args.num_fourier_layers
    if args.fgn_use_residue:
        config.model_config.fgn_use_residue = True
    if args.fgn_no_residue:
        config.model_config.fgn_use_residue = False
    if args.fourier_sparsity_threshold is not None:
        config.model_config.fourier_sparsity_threshold = args.fourier_sparsity_threshold
    if args.fourier_hidden_size_factor is not None:
        config.model_config.fourier_hidden_size_factor = args.fourier_hidden_size_factor
    
    # 应用混合回放池参数
    if args.use_replay_buffer:
        config.use_replay_buffer = True
    if args.no_replay_buffer:
        config.use_replay_buffer = False
    if args.replay_buffer_threshold is not None:
        config.replay_buffer_threshold = args.replay_buffer_threshold
    if args.replay_buffer_ratio is not None:
        config.replay_buffer_ratio = args.replay_buffer_ratio
    if args.replay_buffer_max_size is not None:
        config.replay_buffer_max_size = args.replay_buffer_max_size
    
    # 应用组件开关（关闭类）
    if args.no_key_frame_selector:
        config.model_config.use_key_frame_selector = False
    if args.no_coupled_mamba:
        config.model_config.use_coupled_mamba = False
    if args.no_moe_film:
        config.model_config.use_moe_film = False
    if args.no_hypergraph:
        config.model_config.use_hypergraph = False
    if args.no_frequency_decomp:
        config.model_config.use_frequency_decomp = False
    if args.no_sphere_reg:
        config.model_config.use_sphere_regularization = False
    if args.no_direct_fusion_priors:
        config.model_config.direct_fusion_priors = False
    
    # 应用组件开关（开启类）- 用于开启默认关闭的组件
    if args.use_dsps:
        config.model_config.use_dsps = True
    if hasattr(args, 'dsps_strength') and args.dsps_strength is not None:
        config.model_config.dsps_strength = args.dsps_strength
    
    # ⭐ KL散度多任务学习参数
    if args.use_kl_mtl:
        config.model_config.use_kl_mtl = True
        config.use_kl_mtl = True  # 也设置到config级别
    if hasattr(args, 'kl_mtl_weight') and args.kl_mtl_weight is not None:
        config.model_config.kl_mtl_weight = args.kl_mtl_weight
        config.kl_mtl_weight = args.kl_mtl_weight
    if hasattr(args, 'unimodal_loss_weight') and args.unimodal_loss_weight is not None:
        config.model_config.unimodal_loss_weight = args.unimodal_loss_weight
        config.unimodal_loss_weight = args.unimodal_loss_weight
    
    if args.use_hypergraph:
        config.model_config.use_hypergraph = True
        # ⭐ 超图建模需要对话级 batching
        config.use_dialogue_batching = True
        print("⚠️  启用超图建模，自动启用对话级 batching")
    if args.use_frequency_decomp:
        config.model_config.use_frequency_decomp = True
    if args.use_sphere_reg:
        config.model_config.use_sphere_regularization = True
    if args.use_improved_mlp:
        config.model_config.use_improved_mlp = True
    
    # 对话级 batching 参数
    if hasattr(args, 'dialogue_batch_size') and args.dialogue_batch_size:
        config.dialogue_batch_size = args.dialogue_batch_size
    if hasattr(args, 'max_dialogue_len') and args.max_dialogue_len:
        config.model_config.max_dialogue_len = args.max_dialogue_len
    if hasattr(args, 'max_utterances_per_batch') and args.max_utterances_per_batch:
        config.max_utterances_per_batch = args.max_utterances_per_batch
    if args.mlp_dropout is not None:
        config.model_config.mlp_dropout = args.mlp_dropout
    if args.mlp_expansion_ratio is not None:
        config.model_config.mlp_expansion_ratio = args.mlp_expansion_ratio
    
    # 指标记录文件
    if args.metrics_file is not None:
        config.metrics_file = args.metrics_file
    
    # 多GPU支持：自定义保存目录和日志目录
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    
    # ====== 绘图开关（自动根据任务类型选择指标）======
    if args.disable_plotting:
        # 强制禁用绘图
        config.plotting_enabled = False
    elif args.enable_plotting:
        # 启用绘图，根据任务类型自动选择合适的指标
        config.plotting_enabled = True
        
        if config.task_type == 'regression':
            # 回归任务 (CH-SIMS, CH-SIMSv2): MAE, Loss, Corr, Acc-2/3/5
            config.plot_mae = True
            config.plot_loss = True
            config.plot_corr = True
            config.plot_acc2 = True
            config.plot_acc3 = True
            config.plot_acc5 = True
            # 分类指标不适用
            config.plot_acc = False
            config.plot_f1_weighted = False
            config.plot_f1_macro = False
        else:
            # 分类任务 (MELD, IEMOCAP): Loss, Acc, F1_weighted, F1_macro
            config.plot_loss = True
            config.plot_acc = True  # 整体准确率
            config.plot_f1_weighted = True
            config.plot_f1_macro = True
            # 回归指标不适用
            config.plot_mae = False
            config.plot_corr = False
            config.plot_acc2 = False
            config.plot_acc3 = False
            config.plot_acc5 = False
    else:
        # 默认禁用绘图
        config.plotting_enabled = False
    
    # ========================================
    # ⭐ MoE-FiLM 与 DSPS 互斥校验
    # ========================================
    # 这两种调制方式不能同时启用：
    # - MoE-FiLM 在 Mamba 之前对帧级特征进行调制
    # - DSPS 在 Mamba 内部的 dt/B/C 路径注入条件
    # 同时启用会导致实验结果难以解释，因此强制互斥
    if config.model_config.use_dsps and config.model_config.use_moe_film:
        raise ValueError(
            "\n" + "=" * 70 + "\n"
            "❌ 配置错误：MoE-FiLM 与 DSPS 不能同时启用！\n"
            "=" * 70 + "\n\n"
            "当前配置：\n"
            f"  use_moe_film = {config.model_config.use_moe_film}\n"
            f"  use_dsps = {config.model_config.use_dsps}\n\n"
            "这两种调制方式是互斥的实验对照：\n"
            "  • MoE-FiLM：在 Mamba 之前对帧级特征进行调制\n"
            "  • DSPS：在 Mamba 内部的 dt/B/C 路径注入条件\n\n"
            "解决方案：\n"
            "  方案1：使用 MoE-FiLM（默认）\n"
            "         → 设置 USE_DSPS=false (或不设置)\n"
            "         → 设置 NO_MOE_FILM=false (或不设置)\n\n"
            "  方案2：使用 DSPS\n"
            "         → 设置 USE_DSPS=true\n"
            "         → 设置 NO_MOE_FILM=true\n\n"
            "  方案3：两者都不用（纯 Mamba）\n"
            "         → 设置 USE_DSPS=false\n"
            "         → 设置 NO_MOE_FILM=true\n"
            + "=" * 70
        )
    
    # 训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

