# -*- coding: utf-8 -*-
"""
配置文件 - 重构版
支持chsims/chsimsv2/meld数据集
"""

import os
from smsa_refactored import ModelConfig


class BaseConfig:
    """基础配置类"""
    def __init__(self, dataset_name='chsims'):
        self.dataset_name = dataset_name
        
        # ===== 设备和随机种子 =====
        self.device = 'cuda'
        self.seed = 42
        
        # ===== 数据相关 =====
        self.data_dir = f'./data/{dataset_name}'
        self.seq_length = 50
        self.batch_size = 32
        self.num_workers = 4
        self.cache_size = 20
        
        # ===== 模型维度 =====
        # 根据预处理脚本使用的模型设置：
        # - ViT-g-14 (OpenCLIP): 1024维
        # - WavLM-large: 1024维  
        # - all-mpnet-base-v2: 768维
        self.text_input_dim = 768
        self.audio_input_dim = 1024
        self.video_input_dim = 1024  # 修改：ViT-g-14输出1024维
        self.text_global_dim = 768
        self.social_dim = 768
        self.context_dim = 768
        self.hidden_dim = 256
        self.fusion_hidden_dim = 256
        
        # ===== Mamba相关 =====
        self.num_ism_layers = 2
        self.num_coupled_layers = 2
        
        # ===== 训练相关 =====
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 5e-4  # 增强正则化（进一步提高到5e-4）⭐
        self.dropout_p = 0.3  # 增强正则化（从0.1提高）
        self.grad_clip = 1.0
        
        # ===== 损失函数 =====
        self.loss_function = 'l1'  # 'l1'(MAE), 'mse', 'focal_mse', 'huber'
        
        # ===== 学习率调度器 =====
        self.scheduler_type = 'cosine'  # 'step', 'cosine', 'reduce_on_plateau'
        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.5
        self.scheduler_patience = 5
        
        # ===== 早停 =====
        self.early_stop_patience = 10  # 启用早停
        self.early_stop_metric = 'mae'  # 监控的指标：mae, loss, acc_2, acc_3, f1_2, f1_3, corr, composite
        self.metric_mode = 'min'  # 'min' for MAE/loss, 'max' for Corr/Acc
        # ⭐ 新增：综合指标权重（当early_stop_metric='composite'时使用）
        self.composite_mae_weight = 0.4     # MAE权重（越小越好）
        self.composite_corr_weight = 0.3    # 相关系数权重（越大越好）
        self.composite_acc5_weight = 0.3    # 五分类准确率权重（越大越好）
        
        # ===== 测试集评估 =====
        self.eval_test_every_epoch = False  # 每个epoch后在测试集上评估（默认关闭）
        
        # ===== 模态和关键帧分析 =====
        self.enable_modality_analysis = False  # 启用模态贡献度分析（默认关闭，避免影响训练速度）
        self.analyze_modality_every = 10  # 每N个batch分析一次（适应小数据集）
        self.modality_analysis_epochs = 3  # 在前N个epoch进行分析
        self.enable_keyframe_logging = False  # 启用关键帧统计（默认关闭）
        self.keyframe_log_every = 32  # 每N个utterance打印关键帧统计
        
        # ===== 数据增强 =====
        self.augment_train = True
        self.noise_scale = 0.08  # 增强数据增强（进一步提高到0.08）⭐
        
        # ===== 损失权重 =====
        self.sphere_loss_weight = 0.0  # 暂时关闭超球体正则化 ⭐
        
        # ===== 日志和保存 =====
        self.log_dir = f'./logs/{dataset_name}'
        self.save_dir = f'./checkpoints/{dataset_name}'
        
        # ===== 模型组件配置 =====
        self.model_config = ModelConfig(
            # 关键帧选择 (MDP3) - 新的百分比模式
            use_key_frame_selector=True,
            n_segments=3,              # 分段数（调整为3段）⭐
            frame_ratio=60,            # 每段选择百分比（默认：60%）
            key_frame_lambda=0.2,
            
            # Coupled Mamba
            use_coupled_mamba=True,
            
            # MoE-FiLM调制 (MoFME)
            use_moe_film=True,
            num_film_experts=6,        # 降低专家数量到6 ⭐
            film_top_k=3,              # 调整top-k到3 ⭐
            
            # 超图建模 (M3NET)
            use_hypergraph=False,      # 默认关闭超图（减少复杂度）⭐
            num_hypergraph_layers=2,   # 降低层数到2 ⭐
            
            # 频域分解 (GS-MCC)
            use_frequency_decomp=False,  # 默认关闭频域分解 ⭐
            num_fourier_layers=3,      # 降低层数到3 ⭐
            
            # 超球体正则
            use_sphere_regularization=False,  # 默认关闭 ⭐
            sphere_radius=1.0,
            
            # ⭐ 先验模态是否直接参与融合
            direct_fusion_priors=False,  # 默认False：social/context只用于调制，不参与融合
            
            # ⭐ MLP架构选择
            use_improved_mlp=False,  # 默认False：使用原始2层简单MLP（更稳定）
        )
    
    def __str__(self):
        config_str = f"\n{'='*50}\n"
        config_str += f"Dataset: {self.dataset_name}\n"
        config_str += f"{'='*50}\n"
        
        config_str += "\n[Data Config]\n"
        config_str += f"  data_dir: {self.data_dir}\n"
        config_str += f"  seq_length: {self.seq_length}\n"
        config_str += f"  batch_size: {self.batch_size}\n"
        
        config_str += "\n[Model Config]\n"
        config_str += f"  hidden_dim: {self.hidden_dim}\n"
        config_str += f"  Components:\n"
        config_str += f"    - KeyFrameSelector: {self.model_config.use_key_frame_selector}\n"
        config_str += f"    - CoupledMamba: {self.model_config.use_coupled_mamba}\n"
        config_str += f"    - MoE-FiLM: {self.model_config.use_moe_film}\n"
        config_str += f"    - Hypergraph: {self.model_config.use_hypergraph}\n"
        config_str += f"    - FrequencyDecomp: {self.model_config.use_frequency_decomp}\n"
        config_str += f"    - SphereReg: {self.model_config.use_sphere_regularization}\n"
        
        config_str += "\n[Training Config]\n"
        config_str += f"  learning_rate: {self.learning_rate}\n"
        config_str += f"  num_epochs: {self.num_epochs}\n"
        config_str += f"  early_stop_patience: {self.early_stop_patience}\n"
        
        config_str += f"\n{'='*50}\n"
        return config_str


class CHSIMSConfig(BaseConfig):
    """CH-SIMS配置"""
    def __init__(self):
        super().__init__(dataset_name='chsims')
        self.task_type = 'regression'
        self.num_labels = 1
        self.metrics_type = 'chsims'
        
        # CH-SIMS特定参数
        self.seq_length = 50
        
        # 数据目录
        self.data_dir = './data/chsims_processed'


class CHSIMSV2Config(BaseConfig):
    """CH-SIMSV2配置"""
    def __init__(self):
        super().__init__(dataset_name='chsimsv2')
        self.task_type = 'regression'
        self.num_labels = 1
        self.metrics_type = 'chsims'
        
        # CH-SIMSV2特定参数
        self.seq_length = 50


class MELDConfig(BaseConfig):
    """MELD配置"""
    def __init__(self):
        super().__init__(dataset_name='meld')
        self.task_type = 'classification'
        self.num_labels = 7  # MELD has 7 emotion classes
        self.metrics_type = 'meld'  # 使用MELD专用指标（包含每个类别的ACC/F1）
        
        # MELD情感类别名称（用于日志输出）
        self.emotion_names = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
        
        # MELD特定参数
        self.seq_length = 50
        self.batch_size = 16  # MELD可能需要更小的batch size
        
        # MELD早停使用loss（分类任务）
        self.early_stop_metric = 'loss'
        self.metric_mode = 'min'


def get_config(dataset_name='chsims'):
    """根据数据集名称获取配置"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'chsims':
        config = CHSIMSConfig()
    elif dataset_name == 'chsimsv2' or dataset_name == 'chsimsv2s':
        config = CHSIMSV2Config()
    elif dataset_name == 'meld':
        config = MELDConfig()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 解析命令行参数并覆盖配置
    import argparse
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据集目录路径（覆盖默认路径）')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--dropout_p', type=float, default=None)
    parser.add_argument('--early_stop_patience', type=int, default=None)
    parser.add_argument('--early_stop_metric', type=str, default=None,
                        choices=['mae', 'loss', 'acc_2', 'acc_3', 'acc_5', 'f1_2', 'f1_3', 'f1_5', 'corr', 'composite'],
                        help='早停监控指标（composite为综合指标：0.4*MAE + 0.3*Corr + 0.3*Acc5）')
    parser.add_argument('--sphere_loss_weight', type=float, default=None)
    
    # MDP3参数（新的百分比模式）
    parser.add_argument('--n_segments', type=int, default=None,
                        help='MDP3分段数（默认：4）')
    parser.add_argument('--frame_ratio', type=int, default=None,
                        help='每段选择的帧百分比1-100（默认：60表示60%）')
    
    # MoE-FiLM参数
    parser.add_argument('--num_film_experts', type=int, default=None)
    parser.add_argument('--film_top_k', type=int, default=None)
    
    # 超图参数
    parser.add_argument('--num_hypergraph_layers', type=int, default=None)
    
    # 频域分解参数
    parser.add_argument('--num_fourier_layers', type=int, default=None)
    
    # 模态和关键帧分析参数 ⭐ 新增
    parser.add_argument('--enable_modality_analysis', action='store_true',
                        help='启用模态贡献度分析（会影响训练速度）')
    parser.add_argument('--analyze_modality_every', type=int, default=None,
                        help='每N个batch进行模态分析（默认：10）')
    parser.add_argument('--keyframe_log_every', type=int, default=None,
                        help='每N个utterance打印关键帧统计（默认：32）')
    parser.add_argument('--modality_analysis_epochs', type=int, default=None,
                        help='在前N个epoch进行模态分析（默认：3）')
    parser.add_argument('--enable_keyframe_logging', action='store_true',
                        help='启用关键帧统计打印')
    parser.add_argument('--eval_test_every_epoch', action='store_true',
                        help='每个epoch后在测试集上评估（仅监控，不影响早停）')
    
    # 组件开关
    parser.add_argument('--no_key_frame_selector', action='store_true')
    parser.add_argument('--no_coupled_mamba', action='store_true')
    parser.add_argument('--no_moe_film', action='store_true')
    parser.add_argument('--no_hypergraph', action='store_true')
    parser.add_argument('--no_frequency_decomp', action='store_true')
    parser.add_argument('--no_sphere_reg', action='store_true')
    parser.add_argument('--no_direct_fusion_priors', action='store_true',
                        help='禁止social/context直接参与融合（只用于FiLM调制）')
    parser.add_argument('--use_improved_mlp', action='store_true',
                        help='使用改进版MLP（4层深层+GELU+残差+LayerNorm）')
    
    args, unknown = parser.parse_known_args()
    
    # 用命令行参数覆盖配置
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
        config.fusion_hidden_dim = args.hidden_dim
    if args.dropout_p is not None:
        config.dropout_p = args.dropout_p
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.early_stop_metric is not None:
        config.early_stop_metric = args.early_stop_metric
        # 根据指标类型自动设置mode
        if args.early_stop_metric in ['mae', 'loss']:
            config.metric_mode = 'min'  # 越小越好
        elif args.early_stop_metric in ['acc_2', 'acc_3', 'f1_2', 'f1_3', 'corr']:
            config.metric_mode = 'max'  # 越大越好
    if args.sphere_loss_weight is not None:
        config.sphere_loss_weight = args.sphere_loss_weight
    
    # MDP3参数（新的百分比模式）
    if args.n_segments is not None:
        config.model_config.n_segments = args.n_segments
    if args.frame_ratio is not None:
        # 验证范围
        if not (1 <= args.frame_ratio <= 100):
            raise ValueError(f"frame_ratio必须在1-100之间，当前值：{args.frame_ratio}")
        config.model_config.frame_ratio = args.frame_ratio
    
    # MoE-FiLM参数
    if args.num_film_experts is not None:
        config.model_config.num_film_experts = args.num_film_experts
    if args.film_top_k is not None:
        config.model_config.film_top_k = args.film_top_k
    
    # 超图参数
    if args.num_hypergraph_layers is not None:
        config.model_config.num_hypergraph_layers = args.num_hypergraph_layers
    
    # 频域分解参数
    if args.num_fourier_layers is not None:
        config.model_config.num_fourier_layers = args.num_fourier_layers
    
    # 模态和关键帧分析参数 ⭐ 新增
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
    
    # 组件开关
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
    if args.use_improved_mlp:
        config.model_config.use_improved_mlp = True
    
    return config

