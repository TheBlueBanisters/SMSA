#!/bin/bash

# ========================================
# 统一训练脚本 - 支持多数据集 + 多GPU并行 + 训练曲线绘图
# 使用方法:
#   1. 【推荐】直接修改配置区域(第25行)的DATASET和GPU_ID变量，然后运行: bash train_unified.sh
#   2. 或者使用命令行参数指定数据集: bash train_unified.sh DATASET_NAME
#   3. 命令行参数会覆盖配置文件中的设置
# 
# 单卡训练示例:
#   bash train_unified.sh              # 使用配置文件中的DATASET和GPU_ID
#   bash train_unified.sh chsims       # 使用命令行指定的数据集
#   bash train_unified.sh --batch_size 64  # 使用配置文件数据集+额外参数
#
# 多GPU并行训练示例:
#   # 修改GPU_ID=0，然后运行第一个训练
#   bash train_unified.sh
#   # 修改GPU_ID=1，然后运行第二个训练（会自动创建独立目录避免冲突）
#   bash train_unified.sh
#   
# 输出目录结构（每次训练创建独立目录）:
#   logs/dataset/gpu_X/run_YYYYMMDD_HHMMSS/
#     ├── train.log           # 训练日志
#     ├── metrics.txt         # 训练配置和指标记录
#     └── train.pid           # 进程ID文件
#   
#   checkpoints/dataset/gpu_X/run_YYYYMMDD_HHMMSS/
#     ├── best_model.pth      # 最佳模型检查点
#     ├── metrics_history.json # 指标历史（JSON格式）
#     └── figures/            # 训练曲线图目录
#         ├── mae_curve.png
#         ├── acc_2_curve.png
#         ├── loss_curve.png
#         └── training_curves_combined.png
# ========================================

# ============================================================
# 配置区域 - 在这里修改参数
# ============================================================

# --- 数据集选择 ---
# DATASET="chsimsv2"
DATASET="meld"         # 数据集名称：chsims / chsimsv2 / meld / iemocap
                        # 会自动映射到对应的 _processed 子目录
# DATASET="chsimsv2"  
# --- 数据集根目录 ---
DATA_DIR="./meld_10"  
# DATA_DIR="./chsimsv2_101"           # 数据集根目录（一级目录，留空使用默认 ./data）
                        # 脚本会自动在此目录下查找 {dataset_name}_processed 子目录
                        # 示例：DATA_DIR="./data_new" + DATASET="meld" → ./data_new/meld_processed

# --- GPU设置 ---
# GPU_ID=0  
GPU_ID=1  
           # 使用的GPU编号，多GPU用逗号分隔，如：0,1

# --- 基础训练参数 ---
# 注意：不同数据集有默认值，留空("")则使用数据集默认值（推荐）
SEQ_LENGTH=""             # 留空使用数据集默认 (CH-SIMS/v2=70, MELD=80, IEMOCAP=110)
BATCH_SIZE=""             # 留空使用数据集默认 (CH-SIMS=32, MELD=16)
LEARNING_RATE="5e-5"          # 留空使用数据集默认 (MELD=5e-5)
NUM_EPOCHS="150"          # 训练轮数
HIDDEN_DIM="4096"             # 留空使用默认 256
DROPOUT="0.2"                # 留空使用默认 (MELD=0.4)
EARLY_STOP_PATIENCE="20"  # 早停等待轮数
EARLY_STOP_METRIC=""      # 留空自动选择 (Regression=mae, Classification=f1_weighted)
SPHERE_LOSS_WEIGHT="0.001"

# --- 测试集评估设置 ---
EVAL_TEST_EVERY_EPOCH=true    # 是否在每个epoch后评估测试集（true=启用，false=禁用）⭐
                                # 注意：仅用于监控，不影响早停和模型保存

# --- 学习率调度器参数 ---
SCHEDULER_TYPE="reduce_on_plateau"       # 学习率调度器类型（默认：cosine）⭐ 改为cosine，更稳定
                        # 可选：cosine（余弦退火）、reduce_on_plateau（自适应）、step（固定步长）、留空（使用默认）
SCHEDULER_GAMMA=""      # 学习率衰减系数（默认：0.5，用于step和reduce_on_plateau）
SCHEDULER_PATIENCE=""   # 等待轮数（默认：5，仅用于reduce_on_plateau）⭐
SCHEDULER_STEP_SIZE=""  # 步长（默认：10，仅用于step）
WARMUP_RATIO="0.1"      # 学习率预热比例（默认0.0关闭，0.1=前10%用于预热）⭐

# --- 关键帧选择（MDP3）参数 ---
# 新的百分比模式：根据实际帧数自适应选择
N_SEGMENTS="4"           # ✓ 改进: 将视频分成4段（保留更多信息）
FRAME_RATIO="70"         # ✓ 改进: 每段选择70%的帧（提高信息保留率）
                         # 实际选择帧数 = max(1, floor(每段帧数 * FRAME_RATIO / 100)) * N_SEGMENTS
                         # 示例：100帧视频，4段，60% -> 每段25帧选15帧 -> 总共60帧

# --- MoE-FiLM 参数 ---
NUM_FILM_EXPERTS="4"     # FiLM专家数量（默认：8）   16
FILM_TOP_K="2"           # Top-K路由选择（默认：4）  8
MOE_LOSS_WEIGHT="0.1"   # ⭐ MoE负载均衡损失权重（默认：0.01，防止专家坍缩）
                         # 设为0关闭此损失

# --- CH-SIMSV2 多任务学习 (MTL) 参数 ---
# 仅对 CH-SIMSV2 数据集生效
CHSIMSV2_MTL_LAMBDA="0.1"  # 辅助任务（单模态标签）的损失权重
                           # 0.0 = 不使用单模态辅助损失
                           # 建议范围：0.1 - 0.5

# --- 超图建模（M3NET）参数 ---
NUM_HG_LAYERS="1"            # 超图卷积层数（M3NET原始使用3层）
HG_USE_RESIDUE=true          # 拼接残差（M3NET原始方式）

# --- 频域分解（GS-MCC / FGN）参数 ---
NUM_FOURIER_LAYERS="2"       # 傅里叶卷积层数（GS-MCC原始使用3层）
FGN_USE_RESIDUE=true         # FGN拼接残差（GS-MCC原始方式）
F_SPARSITY_THRESHOLD="0.01"  # 傅里叶稀疏阈值
F_HIDDEN_SIZE_FACTOR="2"     # 傅里叶隐藏层倍数

# --- KL散度多任务学习（GS-MCC）参数 ---
# ⭐ 让单模态预测去"模仿"融合后的预测（软标签蒸馏）
USE_KL_MTL=true             # 是否启用KL散度多任务学习（true=启用，false=禁用）
KL_MTL_WEIGHT="1.0"          # KL散度损失的权重（默认1.0）
UNIMODAL_LOSS_WEIGHT="1.0"   # 单模态分类损失的权重（默认1.0）

# --- 模态和关键帧分析参数 ---
ENABLE_MODALITY_ANALYSIS=false  # 是否启用模态分析（true=启用，false=禁用）⭐
ANALYZE_MODALITY_EVERY="1"     # 每N个batch进行模态分析（默认：10）
KEYFRAME_LOG_EVERY="3"         # 每N个utterance打印关键帧统计（默认：32）
MODALITY_ANALYSIS_EPOCHS="10"    # 在前N个epoch进行模态分析（默认：3）
ENABLE_KEYFRAME_LOGGING=false   # 是否启用关键帧统计（true=启用，false=禁用）⭐

# --- 组件开关（true=关闭该组件，false或留空=使用该组件）---
NO_KEY_FRAME_SELECTOR=true    # 关闭关键帧选择
NO_COUPLED_MAMBA=false         # 关闭Coupled Mamba（使用独立Mamba）
NO_MOE_FILM=true             # 关闭MoE-FiLM调制
NO_HYPERGRAPH=false            # 关闭超图建模
                               # ⚠️ 启用超图建模(false)会自动启用对话级batching
MAX_UTTERANCES_PER_BATCH=64   # ⭐ 每批最大utterance数（控制显存，防止OOM）
MAX_DIALOGUE_LEN=200           # 单个对话最大utterance数（超过的对话会被跳过）
NO_FREQUENCY_DECOMP=false      # 关闭频域分解
NO_SPHERE_REG=true         # 关闭超球体正则化
NO_DIRECT_FUSION_PRIORS=true  # ⭐ 禁止social/context直接参与融合（只用于调制）
                               # true=social/context只用于FiLM调制，不送入最终MLP融合
                               # false=social/context既用于调制，也参与融合（原始行为）

# --- DSPS（条件化SSM）开关 ---
NO_DSPS=false                  # ⭐ DSPS条件化SSM开关（与其他开关风格一致）
                               # true=关闭DSPS（使用标准Mamba，默认）
                               # false=启用DSPS（在Mamba内部的dt/B/C路径注入条件）
                               # ⚠️ 重要：NO_DSPS=false 时必须同时设置 NO_MOE_FILM=true
                               #         MoE-FiLM 与 DSPS 互斥，不能同时启用！
DSPS_STRENGTH=0.5              # DSPS强度因子（0.0=无效果，1.0=完全效果）
                               # 建议从0.1开始，根据实验调整

# --- MLP架构选择 ---
USE_IMPROVED_MLP=false
         # ⭐ MLP架构选择（true=改进版，false=原始版）
                               # true=改进版：4层深层MLP + GELU + 残差 + LayerNorm（容量大，性能可能更好）
                               # false=原始版：2层简单MLP + ReLU（容量小，更稳定，当前使用）
MLP_DROPOUT=0.2              # 改进版MLP的Dropout比例（默认0.2）
MLP_EXPANSION_RATIO=4        # 改进版MLP中间层扩维倍数（默认4，即中间层=输入×4）

# --- 混合回放池 (Experience Replay) ---
USE_REPLAY_BUFFER=true        # ⭐ 混合回放池开关
                               # true=启用（类似DQN的经验回放，关注高loss难样本）
                               # false=禁用（默认）
REPLAY_BUFFER_THRESHOLD="1.8"  # Loss阈值倍数（batch_loss > avg_loss * threshold 时加入池）
REPLAY_BUFFER_RATIO="0.2"      # 回放比例（每epoch额外训练20%的难样本batch）
REPLAY_BUFFER_MAX_SIZE="500"   # 回放池最大容量

# --- 类别权重设置（分类任务） ---
# ⭐ 用于处理 MELD 等数据集的类别不平衡问题
USE_CLASS_WEIGHTS="true"         # 类别权重开关：
                            #   ""（留空）= 使用数据集默认值（MELD默认开启）
                            #   true      = 强制开启（给少数类更高权重）
                            #   false     = 强制关闭（所有类别权重相等）

# --- 采样与损失函数增强 (Updated) ---
# 针对类别不平衡（特别是IEMOCAP负样本多）的优化
USE_WEIGHTED_SAMPLER="true"     # 是否使用加权随机采样（解决类别不平衡）
                            #   "" (留空) = 使用数据集默认（IEMOCAP默认开启）
                            #   true      = 强制开启 (注意：会禁用shuffle)
                            #   false     = 强制关闭

LABEL_SMOOTHING="0.1"          # 标签平滑系数 (防止过拟合)
                            #   "" (留空) = 使用数据集默认（IEMOCAP默认0.1）
                            #   0.0       = 关闭
                            #   0.1       = 推荐值

# --- Focal Loss 设置（分类任务） ---
# ⭐ Focal Loss 用于处理类别不平衡，让模型更关注难分类的样本
# 仅对 MELD 和 IEMOCAP 等分类任务生效
USE_FOCAL_LOSS="true"              # Focal Loss 开关：
                            #   "" (留空) = 使用数据集默认配置（MELD默认开启）
                            #   true      = 强制开启 (loss_function=focal)
                            #   false     = 强制关闭 (loss_function=ce)
FOCAL_LOSS_GAMMA="5.0"       # Focal Loss 的 gamma 参数（默认：2.0
                            # 如果启用动态gamma，这是初始值

# --- 动态 Focal Loss Gamma 衰减 (新功能) ---
# ⭐ 让 gamma 随训练进度逐渐衰减，平衡少数类和多数类的学习
# 训练初期：高 gamma → 快速学习少数类特征
# 训练后期：低 gamma → 恢复多数类（neutral）的置信度
FOCAL_DYNAMIC_GAMMA=true     # 是否启用动态gamma衰减（false=关闭，true=开启）
FOCAL_GAMMA_MIN="0.5"         # gamma 衰减的最小值（默认：0.5）
FOCAL_GAMMA_DECAY_MODE="exponential"  # gamma 衰减模式：
                            #   linear      = 线性衰减（匀速）
                            #   exponential = 指数衰减（先快后慢）
                            #   cosine      = 余弦衰减（先慢后快再慢，推荐）
                            #   step        = 阶梯式衰减（分3段降低）

# --- 课程学习 (Curriculum Learning) 设置 ---
# ⭐ 用于解决 Mamba Backbone 与 MoE 模块协同训练时的不稳定问题（特别是 Acc-5 掉点问题）
CURRICULUM_MODE="freeze_backbone"       # 课程学习模式：
                             #   none           = 关闭课程学习（默认行为，与之前完全兼容）
                             #   freeze_backbone = 策略A：冻结骨干网络，只训练MoE/FiLM/Head
                             #   alpha_blending  = 策略B：渐进式MoE混合（alpha从0.2→1.0）
CURRICULUM_EPOCHS=10          # 课程学习持续的Epoch数（默认：5）
                             # freeze_backbone: 前N个epoch冻结Backbone，之后解冻
                             # alpha_blending:  alpha = min(1.0, epoch / N)

# --- 后台运行设置 ---
USE_NOHUP=true          # 使用nohup后台运行（SSH断开后继续训练）
                        # true=后台运行，false=前台运行

# --- 训练曲线绘图设置 ---
ENABLE_PLOTTING=true    # 是否启用训练曲线绘图（自动根据数据集选择合适指标）
                        # 回归任务(CH-SIMS/v2): MAE, Loss, Corr, Acc-2/3/5
                        # 分类任务(MELD/IEMOCAP): Loss, Acc, F1_weighted, F1_macro

# --- 其他参数 ---
# 注意：以下参数在配置文件(config_refactored.py)中设置，不通过命令行传递
NUM_WORKERS=4           # 数据加载器的工作进程数（在配置文件中生效）
SEED=42                 # 随机种子（在配置文件中生效）

# ============================================================
# 以下是脚本逻辑，一般不需要修改
# ============================================================

# 处理命令行参数
if [ $# -ge 1 ]; then
    # 如果提供了命令行参数，检查第一个参数是否是数据集名称
    if [[ "$1" =~ ^(chsims|chsimsv2|meld|iemocap)$ ]]; then
        DATASET=$1
        shift  # 移除第一个参数，剩余的作为额外选项
        echo "使用命令行指定的数据集: $DATASET（覆盖配置文件）"
    else
        # 第一个参数不是数据集名称，使用配置文件中的数据集
        echo "使用配置文件中的数据集: $DATASET"
    fi
else
    # 没有命令行参数，使用配置文件中的数据集
    echo "使用配置文件中的数据集: $DATASET"
fi

# 验证数据集名称
if [[ ! "$DATASET" =~ ^(chsims|chsimsv2|meld|iemocap)$ ]]; then
    echo ""
    echo "=========================================="
    echo "错误：无效的数据集名称: '$DATASET'"
    echo "=========================================="
    echo "可用的数据集: chsims, chsimsv2, meld, iemocap"
    echo ""
    echo "请在脚本第20行修改 DATASET 变量，或使用命令行参数："
    echo "  bash train_unified.sh chsims"
    echo "  bash train_unified.sh chsimsv2"
    echo "  bash train_unified.sh meld"
    echo "  bash train_unified.sh iemocap"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo ""
    echo "=========================================="
    echo "❌ 错误：未检测到conda环境"
    echo "=========================================="
    echo "请先激活一个conda环境（非base环境）"
    echo ""
    echo "示例："
    echo "  conda create -n kopa python=3.8"
    echo "  conda activate kopa"
    echo "=========================================="
    exit 1
elif [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo ""
    echo "=========================================="
    echo "❌ 错误：不允许在base环境中运行"
    echo "=========================================="
    echo "请创建并激活一个独立的conda环境"
    echo ""
    echo "示例："
    echo "  conda create -n kopa python=3.8"
    echo "  conda activate kopa"
    echo "  bash train_unified.sh"
    echo "=========================================="
    exit 1
else
    echo "✅ 当前环境: $CONDA_DEFAULT_ENV"
    
    # 检查关键依赖包
    echo "检查依赖包..."
    MISSING_PACKAGES=()
    
    # 检查torch
    python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
    
    # 检查其他关键包
    python -c "import transformers" 2>/dev/null || MISSING_PACKAGES+=("transformers")
    python -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
    python -c "import tqdm" 2>/dev/null || MISSING_PACKAGES+=("tqdm")
    python -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  检测到缺失的依赖包"
        echo "=========================================="
        echo "缺失的包: ${MISSING_PACKAGES[*]}"
        echo ""
        echo "正在尝试自动安装..."
        
        # 检查是否有requirements.txt
        if [ -f "requirements.txt" ]; then
            echo "使用requirements.txt安装依赖..."
            pip install -r requirements.txt -q
        else
            echo "逐个安装缺失的包..."
            for pkg in "${MISSING_PACKAGES[@]}"; do
                echo "安装 $pkg..."
                if [ "$pkg" = "torch" ]; then
                    pip install torch torchvision torchaudio -q
                else
                    pip install "$pkg" -q
                fi
            done
        fi
        
        # 再次检查是否安装成功
        echo ""
        echo "验证安装..."
        ALL_OK=true
        for pkg in "${MISSING_PACKAGES[@]}"; do
            pkg_import="${pkg//-/_}"  # 处理包名如scikit-learn -> sklearn
            [ "$pkg" = "scikit-learn" ] && pkg_import="sklearn"
            
            if ! python -c "import $pkg_import" 2>/dev/null; then
                echo "❌ $pkg 安装失败"
                ALL_OK=false
            else
                echo "✅ $pkg 安装成功"
            fi
        done
        
        if [ "$ALL_OK" = false ]; then
            echo ""
            echo "=========================================="
            echo "❌ 部分依赖包安装失败"
            echo "=========================================="
            echo "请手动安装："
            echo "  pip install -r requirements.txt"
            echo "或者："
            echo "  pip install torch transformers numpy tqdm scikit-learn"
            echo "=========================================="
            exit 1
        fi
        
        echo ""
        echo "✅ 所有依赖包已安装"
        echo "=========================================="
    else
        echo "✅ 所有依赖包已就绪"
    fi
fi

# 基础参数（所有数据集通用）
COMMON_ARGS="--dataset $DATASET"

# 数据集默认参数
case $DATASET in
    chsims)
        echo "=========================================="
        echo "Training on CH-SIMS dataset"
        echo "=========================================="
        DEFAULT_BATCH_SIZE=32
        DEFAULT_LR=1e-4
        DEFAULT_EPOCHS=50
        ;;
    chsimsv2)
        echo "=========================================="
        echo "Training on CH-SIMSV2 dataset"
        echo "=========================================="
        DEFAULT_BATCH_SIZE=32
        DEFAULT_LR=1e-4
        DEFAULT_EPOCHS=50
        ;;
    meld)
        echo "=========================================="
        echo "Training on MELD dataset"
        echo "=========================================="
        DEFAULT_BATCH_SIZE=16
        DEFAULT_LR=5e-5
        DEFAULT_EPOCHS=40
        ;;
    iemocap)
        echo "=========================================="
        echo "Training on IEMOCAP dataset"
        echo "=========================================="
        DEFAULT_BATCH_SIZE=16
        DEFAULT_LR=5e-5
        DEFAULT_EPOCHS=50
        ;;
esac

# 使用配置变量或默认值
FINAL_BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
FINAL_LR=${LEARNING_RATE:-$DEFAULT_LR}
FINAL_EPOCHS=${NUM_EPOCHS:-$DEFAULT_EPOCHS}

# 构建参数字符串
DATASET_ARGS="--batch_size $FINAL_BATCH_SIZE --learning_rate $FINAL_LR --num_epochs $FINAL_EPOCHS"

# 构建完整的数据目录路径
# DATA_DIR 是一级目录，自动拼接 {dataset_name}_processed
if [ -n "$DATA_DIR" ]; then
    # 数据集名称到目录名的映射
    case $DATASET in
        chsims)
            DATASET_SUBDIR="chsims_processed"
            ;;
        chsimsv2)
            DATASET_SUBDIR="chsimsv2_processed"
            ;;
        meld)
            DATASET_SUBDIR="meld"
            ;;
        iemocap)
            DATASET_SUBDIR="iemocap"
            ;;
    esac
    FULL_DATA_DIR="${DATA_DIR}/${DATASET_SUBDIR}"
    echo "数据目录: $FULL_DATA_DIR"
fi

# 添加可选参数（只有在设置了值时才添加）
[ -n "$FULL_DATA_DIR" ] && DATASET_ARGS="$DATASET_ARGS --data_dir $FULL_DATA_DIR"
[ -n "$SEQ_LENGTH" ] && DATASET_ARGS="$DATASET_ARGS --seq_length $SEQ_LENGTH"
[ -n "$HIDDEN_DIM" ] && DATASET_ARGS="$DATASET_ARGS --hidden_dim $HIDDEN_DIM"
[ -n "$DROPOUT" ] && DATASET_ARGS="$DATASET_ARGS --dropout_p $DROPOUT"
[ -n "$EARLY_STOP_PATIENCE" ] && DATASET_ARGS="$DATASET_ARGS --early_stop_patience $EARLY_STOP_PATIENCE"
[ -n "$EARLY_STOP_METRIC" ] && DATASET_ARGS="$DATASET_ARGS --early_stop_metric $EARLY_STOP_METRIC"
[ -n "$SPHERE_LOSS_WEIGHT" ] && DATASET_ARGS="$DATASET_ARGS --sphere_loss_weight $SPHERE_LOSS_WEIGHT"
[ -n "$MOE_LOSS_WEIGHT" ] && DATASET_ARGS="$DATASET_ARGS --moe_loss_weight $MOE_LOSS_WEIGHT"
[ -n "$CHSIMSV2_MTL_LAMBDA" ] && DATASET_ARGS="$DATASET_ARGS --chsimsv2_mtl_lambda $CHSIMSV2_MTL_LAMBDA"

# 学习率调度器参数
[ -n "$SCHEDULER_TYPE" ] && DATASET_ARGS="$DATASET_ARGS --scheduler_type $SCHEDULER_TYPE"
[ -n "$SCHEDULER_GAMMA" ] && DATASET_ARGS="$DATASET_ARGS --scheduler_gamma $SCHEDULER_GAMMA"
[ -n "$SCHEDULER_PATIENCE" ] && DATASET_ARGS="$DATASET_ARGS --scheduler_patience $SCHEDULER_PATIENCE"
[ -n "$SCHEDULER_STEP_SIZE" ] && DATASET_ARGS="$DATASET_ARGS --scheduler_step_size $SCHEDULER_STEP_SIZE"
[ -n "$WARMUP_RATIO" ] && DATASET_ARGS="$DATASET_ARGS --warmup_ratio $WARMUP_RATIO"
[ -n "$N_SEGMENTS" ] && DATASET_ARGS="$DATASET_ARGS --n_segments $N_SEGMENTS"
[ -n "$FRAME_RATIO" ] && DATASET_ARGS="$DATASET_ARGS --frame_ratio $FRAME_RATIO"
[ -n "$NUM_FILM_EXPERTS" ] && DATASET_ARGS="$DATASET_ARGS --num_film_experts $NUM_FILM_EXPERTS"
[ -n "$FILM_TOP_K" ] && DATASET_ARGS="$DATASET_ARGS --film_top_k $FILM_TOP_K"
[ -n "$NUM_HG_LAYERS" ] && DATASET_ARGS="$DATASET_ARGS --num_hypergraph_layers $NUM_HG_LAYERS"
[ "$HG_USE_RESIDUE" = true ] && DATASET_ARGS="$DATASET_ARGS --hypergraph_use_residue"
[ "$HG_USE_RESIDUE" = false ] && DATASET_ARGS="$DATASET_ARGS --hypergraph_no_residue"
# 频域分解（GS-MCC / FGN）参数
[ -n "$NUM_FOURIER_LAYERS" ] && DATASET_ARGS="$DATASET_ARGS --num_fourier_layers $NUM_FOURIER_LAYERS"
[ "$FGN_USE_RESIDUE" = true ] && DATASET_ARGS="$DATASET_ARGS --fgn_use_residue"
[ "$FGN_USE_RESIDUE" = false ] && DATASET_ARGS="$DATASET_ARGS --fgn_no_residue"
[ -n "$F_SPARSITY_THRESHOLD" ] && DATASET_ARGS="$DATASET_ARGS --fourier_sparsity_threshold $F_SPARSITY_THRESHOLD"
[ -n "$F_HIDDEN_SIZE_FACTOR" ] && DATASET_ARGS="$DATASET_ARGS --fourier_hidden_size_factor $F_HIDDEN_SIZE_FACTOR"

# KL散度多任务学习参数
[ "$USE_KL_MTL" = true ] && DATASET_ARGS="$DATASET_ARGS --use_kl_mtl"
[ -n "$KL_MTL_WEIGHT" ] && DATASET_ARGS="$DATASET_ARGS --kl_mtl_weight $KL_MTL_WEIGHT"
[ -n "$UNIMODAL_LOSS_WEIGHT" ] && DATASET_ARGS="$DATASET_ARGS --unimodal_loss_weight $UNIMODAL_LOSS_WEIGHT"

# 测试集评估参数
[ "$EVAL_TEST_EVERY_EPOCH" = true ] && DATASET_ARGS="$DATASET_ARGS --eval_test_every_epoch"

# 模态和关键帧分析参数
[ "$ENABLE_MODALITY_ANALYSIS" = true ] && DATASET_ARGS="$DATASET_ARGS --enable_modality_analysis"
[ -n "$ANALYZE_MODALITY_EVERY" ] && DATASET_ARGS="$DATASET_ARGS --analyze_modality_every $ANALYZE_MODALITY_EVERY"
[ -n "$KEYFRAME_LOG_EVERY" ] && DATASET_ARGS="$DATASET_ARGS --keyframe_log_every $KEYFRAME_LOG_EVERY"
[ -n "$MODALITY_ANALYSIS_EPOCHS" ] && DATASET_ARGS="$DATASET_ARGS --modality_analysis_epochs $MODALITY_ANALYSIS_EPOCHS"
[ "$ENABLE_KEYFRAME_LOGGING" = true ] && DATASET_ARGS="$DATASET_ARGS --enable_keyframe_logging"
# 注意：NUM_WORKERS 和 SEED 在配置文件中设置，不需要命令行传递

# 添加组件开关（关闭类：NO_XXX=true 时传递 --no_xxx）
[ "$NO_KEY_FRAME_SELECTOR" = true ] && DATASET_ARGS="$DATASET_ARGS --no_key_frame_selector"
[ "$NO_COUPLED_MAMBA" = true ] && DATASET_ARGS="$DATASET_ARGS --no_coupled_mamba"
[ "$NO_MOE_FILM" = true ] && DATASET_ARGS="$DATASET_ARGS --no_moe_film"
[ "$NO_DIRECT_FUSION_PRIORS" = true ] && DATASET_ARGS="$DATASET_ARGS --no_direct_fusion_priors"

# 添加组件开关（开启类：NO_XXX=false 时传递 --use_xxx，用于开启默认关闭的组件）
[ "$NO_HYPERGRAPH" = false ] && DATASET_ARGS="$DATASET_ARGS --use_hypergraph"
[ "$NO_FREQUENCY_DECOMP" = false ] && DATASET_ARGS="$DATASET_ARGS --use_frequency_decomp"
[ "$NO_SPHERE_REG" = false ] && DATASET_ARGS="$DATASET_ARGS --use_sphere_reg"

# 对话级 batching 参数（超图建模时自动启用）
if [ "$NO_HYPERGRAPH" = false ]; then
    DATASET_ARGS="$DATASET_ARGS --max_utterances_per_batch $MAX_UTTERANCES_PER_BATCH"
    DATASET_ARGS="$DATASET_ARGS --max_dialogue_len $MAX_DIALOGUE_LEN"
fi

# MLP 参数
[ "$USE_IMPROVED_MLP" = true ] && DATASET_ARGS="$DATASET_ARGS --use_improved_mlp"
[ -n "$MLP_DROPOUT" ] && DATASET_ARGS="$DATASET_ARGS --mlp_dropout $MLP_DROPOUT"
[ -n "$MLP_EXPANSION_RATIO" ] && DATASET_ARGS="$DATASET_ARGS --mlp_expansion_ratio $MLP_EXPANSION_RATIO"

# 混合回放池参数
[ "$USE_REPLAY_BUFFER" = true ] && DATASET_ARGS="$DATASET_ARGS --use_replay_buffer"
[ "$USE_REPLAY_BUFFER" = false ] && DATASET_ARGS="$DATASET_ARGS --no_replay_buffer"
[ -n "$REPLAY_BUFFER_THRESHOLD" ] && DATASET_ARGS="$DATASET_ARGS --replay_buffer_threshold $REPLAY_BUFFER_THRESHOLD"
[ -n "$REPLAY_BUFFER_RATIO" ] && DATASET_ARGS="$DATASET_ARGS --replay_buffer_ratio $REPLAY_BUFFER_RATIO"
[ -n "$REPLAY_BUFFER_MAX_SIZE" ] && DATASET_ARGS="$DATASET_ARGS --replay_buffer_max_size $REPLAY_BUFFER_MAX_SIZE"

# 类别权重参数
[ "$USE_CLASS_WEIGHTS" = true ] && DATASET_ARGS="$DATASET_ARGS --use_class_weights"
[ "$USE_CLASS_WEIGHTS" = false ] && DATASET_ARGS="$DATASET_ARGS --no_class_weights"

# 加权采样参数
[ "$USE_WEIGHTED_SAMPLER" = true ] && DATASET_ARGS="$DATASET_ARGS --use_weighted_sampler"
[ "$USE_WEIGHTED_SAMPLER" = false ] && DATASET_ARGS="$DATASET_ARGS --no_weighted_sampler"

# 标签平滑参数
[ -n "$LABEL_SMOOTHING" ] && DATASET_ARGS="$DATASET_ARGS --label_smoothing $LABEL_SMOOTHING"

# Focal Loss 参数（分类任务）
if [ "$USE_FOCAL_LOSS" = true ]; then
    DATASET_ARGS="$DATASET_ARGS --loss_function focal"
    [ -n "$FOCAL_LOSS_GAMMA" ] && DATASET_ARGS="$DATASET_ARGS --focal_gamma $FOCAL_LOSS_GAMMA"
    
    # 动态 Gamma 参数
    if [ "$FOCAL_DYNAMIC_GAMMA" = true ]; then
        DATASET_ARGS="$DATASET_ARGS --focal_dynamic_gamma"
        [ -n "$FOCAL_GAMMA_MIN" ] && DATASET_ARGS="$DATASET_ARGS --focal_gamma_min $FOCAL_GAMMA_MIN"
        [ -n "$FOCAL_GAMMA_DECAY_MODE" ] && DATASET_ARGS="$DATASET_ARGS --focal_gamma_decay_mode $FOCAL_GAMMA_DECAY_MODE"
    fi
elif [ "$USE_FOCAL_LOSS" = false ]; then
    # 如果强制关闭，对于分类任务使用标准交叉熵
    if [[ "$DATASET" == "meld" || "$DATASET" == "iemocap" ]]; then
        DATASET_ARGS="$DATASET_ARGS --loss_function ce"
    fi
fi

# 课程学习参数
[ -n "$CURRICULUM_MODE" ] && [ "$CURRICULUM_MODE" != "none" ] && DATASET_ARGS="$DATASET_ARGS --curriculum_mode $CURRICULUM_MODE"
[ -n "$CURRICULUM_EPOCHS" ] && [ "$CURRICULUM_MODE" != "none" ] && DATASET_ARGS="$DATASET_ARGS --curriculum_epochs $CURRICULUM_EPOCHS"

# DSPS（条件化SSM）参数（注意：NO_DSPS=false 表示启用）
if [ "$NO_DSPS" = false ]; then
    DATASET_ARGS="$DATASET_ARGS --use_dsps"
    DATASET_ARGS="$DATASET_ARGS --dsps_strength $DSPS_STRENGTH"
fi

# 为多GPU并行训练创建独立目录
# 使用GPU_ID作为目录后缀来避免冲突
GPU_SUFFIX=$(echo $GPU_ID | tr ',' '_')  # 将逗号转为下划线，例如 "0,1" -> "0_1"

# ====== 每次训练创建独立目录 ======
# 格式: logs/dataset/gpu_X/run_YYYYMMDD_HHMMSS/
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="run_${TIMESTAMP}"

# 创建本次训练的独立目录
LOG_BASE_DIR="./logs/$DATASET/gpu_${GPU_SUFFIX}/${RUN_DIR}"
CHECKPOINT_BASE_DIR="./checkpoints/$DATASET/gpu_${GPU_SUFFIX}/${RUN_DIR}"
mkdir -p "$LOG_BASE_DIR"
mkdir -p "$CHECKPOINT_BASE_DIR"

echo "=========================================="
echo "📁 本次训练输出目录:"
echo "   日志: $LOG_BASE_DIR"
echo "   模型: $CHECKPOINT_BASE_DIR"
echo "=========================================="

# 检查是否有相同GPU的训练正在运行（避免GPU资源冲突）
PID_FILE="$LOG_BASE_DIR/train.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "⚠️  警告：GPU $GPU_ID 上已有训练在运行"
        echo "=========================================="
        echo "进程ID: $OLD_PID"
        echo ""
        echo "如果要强制启动新训练，请先停止旧训练："
        echo "  kill $OLD_PID"
        echo ""
        echo "或者删除PID文件后重试："
        echo "  rm $PID_FILE"
        echo "=========================================="
        exit 1
    else
        # 旧进程已结束，删除过期的PID文件
        rm -f "$PID_FILE"
    fi
fi

# 显示最终配置
echo ""
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "数据集: $DATASET"
echo "GPU: $GPU_ID"
echo "运行目录: $RUN_DIR"
echo ""
echo "基础参数:"
echo "  序列长度: ${SEQ_LENGTH:-'(默认)'}"
echo "  批大小: $FINAL_BATCH_SIZE"
echo "  学习率: $FINAL_LR"
echo "  训练轮数: $FINAL_EPOCHS"
[ -n "$FULL_DATA_DIR" ] && echo "  数据目录: $FULL_DATA_DIR" || echo "  数据目录: (默认)"
[ -n "$HIDDEN_DIM" ] && echo "  隐藏层维度: $HIDDEN_DIM"
[ -n "$DROPOUT" ] && echo "  Dropout: $DROPOUT"
echo ""
echo "组件状态:"
echo "  关键帧选择: $([ "$NO_KEY_FRAME_SELECTOR" = true ] && echo '关闭' || echo '开启')"
echo "  Coupled Mamba: $([ "$NO_COUPLED_MAMBA" = true ] && echo '关闭' || echo '开启')"
echo "  MoE-FiLM: $([ "$NO_MOE_FILM" = true ] && echo '关闭' || echo '开启')"
echo "  DSPS (条件化SSM): $([ "$NO_DSPS" = true ] && echo '关闭' || echo "开启 (strength=$DSPS_STRENGTH)")"
if [ "$DATASET" = "chsimsv2" ]; then
    echo "  CH-SIMSv2 MTL: $([ "$CHSIMSV2_MTL_LAMBDA" != "0.0" ] && echo "开启 (lambda=$CHSIMSV2_MTL_LAMBDA)" || echo '关闭')"
fi
echo "  超图建模: $([ "$NO_HYPERGRAPH" = true ] && echo '关闭' || echo '开启')"
if [ "$NO_HYPERGRAPH" = false ]; then
    echo "    - 对话级batching: 开启（动态BatchSampler）"
    echo "    - 每批最大utterances: $MAX_UTTERANCES_PER_BATCH"
    echo "    - 最大对话长度: $MAX_DIALOGUE_LEN"
fi
echo "  频域分解: $([ "$NO_FREQUENCY_DECOMP" = true ] && echo '关闭' || echo '开启')"
if [ "$NO_FREQUENCY_DECOMP" = false ]; then
    echo "    - 傅里叶层数: $NUM_FOURIER_LAYERS"
    echo "    - 拼接残差: $([ "$FGN_USE_RESIDUE" = true ] && echo '开启' || echo '关闭')"
fi
echo "  KL散度多任务: $([ "$USE_KL_MTL" = true ] && echo "开启 (kl=$KL_MTL_WEIGHT, uni=$UNIMODAL_LOSS_WEIGHT)" || echo '关闭')"
echo "  超球体正则: $([ "$NO_SPHERE_REG" = true ] && echo '关闭' || echo '开启')"
echo "  先验直接融合: $([ "$NO_DIRECT_FUSION_PRIORS" = true ] && echo '关闭(只调制)' || echo '开启')"
echo ""
echo "类别不平衡处理:"
if [ -z "$USE_CLASS_WEIGHTS" ]; then
    echo "  类别权重: 使用数据集默认值（MELD默认开启）"
elif [ "$USE_CLASS_WEIGHTS" = true ]; then
    echo "  类别权重: 强制开启"
else
    echo "  类别权重: 强制关闭"
fi
if [ -z "$USE_FOCAL_LOSS" ]; then
    echo "  Focal Loss: 使用数据集默认值（MELD默认开启）"
elif [ "$USE_FOCAL_LOSS" = true ]; then
    if [ "$FOCAL_DYNAMIC_GAMMA" = true ]; then
        echo "  Focal Loss: 强制开启 (动态gamma: $FOCAL_LOSS_GAMMA→$FOCAL_GAMMA_MIN, mode=$FOCAL_GAMMA_DECAY_MODE)"
    else
        echo "  Focal Loss: 强制开启 (gamma=$FOCAL_LOSS_GAMMA)"
    fi
else
    echo "  Focal Loss: 强制关闭"
fi
if [ -z "$USE_WEIGHTED_SAMPLER" ]; then
    echo "  加权采样: 使用数据集默认值（IEMOCAP默认开启）"
elif [ "$USE_WEIGHTED_SAMPLER" = true ]; then
    echo "  加权采样: 强制开启"
else
    echo "  加权采样: 强制关闭"
fi
if [ -n "$LABEL_SMOOTHING" ]; then
    echo "  标签平滑: $LABEL_SMOOTHING"
else
    echo "  标签平滑: 使用数据集默认值"
fi
echo ""
echo "学习率调度:"
echo "  调度器类型: ${SCHEDULER_TYPE:-cosine}"
echo "  预热比例: ${WARMUP_RATIO:-0.0}"
echo ""
echo "MLP架构:"
echo "  版本: $([ "$USE_IMPROVED_MLP" = true ] && echo '改进版(4层深层+GELU+残差)' || echo '原始版(2层简单+ReLU)')"
echo "  Dropout: $MLP_DROPOUT"
echo "  扩维倍数: $MLP_EXPANSION_RATIO"
echo ""
echo "混合回放池 (Experience Replay):"
echo "  启用: $([ "$USE_REPLAY_BUFFER" = true ] && echo '是' || echo '否')"
if [ "$USE_REPLAY_BUFFER" = true ]; then
    echo "  Loss阈值倍数: $REPLAY_BUFFER_THRESHOLD"
    echo "  回放比例: $REPLAY_BUFFER_RATIO"
    echo "  最大容量: $REPLAY_BUFFER_MAX_SIZE"
fi
echo ""
echo "课程学习 (Curriculum Learning):"
if [ "$CURRICULUM_MODE" = "none" ] || [ -z "$CURRICULUM_MODE" ]; then
    echo "  状态: 关闭"
elif [ "$CURRICULUM_MODE" = "freeze_backbone" ]; then
    echo "  状态: 开启 - 冻结骨干网络"
    echo "  策略: 前${CURRICULUM_EPOCHS}个epoch只训练MoE/FiLM/Head，之后解冻全部参数"
elif [ "$CURRICULUM_MODE" = "alpha_blending" ]; then
    echo "  状态: 开启 - 渐进式Alpha混合"
    echo "  策略: MoE影响力从0.2渐进增加到1.0（持续${CURRICULUM_EPOCHS}个epoch）"
fi
echo ""
echo "分析和日志:"
echo "  每epoch测试集评估: $([ "$EVAL_TEST_EVERY_EPOCH" = true ] && echo '开启' || echo '关闭')"
echo "  模态贡献度分析: $([ "$ENABLE_MODALITY_ANALYSIS" = true ] && echo "开启 (每${ANALYZE_MODALITY_EVERY}个batch)" || echo '关闭')"
echo "  关键帧统计: $([ "$ENABLE_KEYFRAME_LOGGING" = true ] && echo "开启 (每${KEYFRAME_LOG_EVERY}个utterance)" || echo '关闭')"
echo ""
echo "训练曲线绘图:"
if [ "$ENABLE_PLOTTING" = true ]; then
    echo "  状态: 开启（自动选择指标）"
    if [[ "$DATASET" == "meld" || "$DATASET" == "iemocap" ]]; then
        echo "  指标: Loss, Acc, F1_weighted, F1_macro"
    else
        echo "  指标: MAE, Loss, Corr, Acc-2/3/5"
    fi
else
    echo "  状态: 关闭"
fi
echo ""
[ $# -gt 0 ] && echo "额外命令行参数: $@" && echo ""
echo "=========================================="
echo ""

# 日志文件路径（使用本次训练的独立目录）
LOG_FILE="$LOG_BASE_DIR/train.log"
METRICS_FILE="$LOG_BASE_DIR/metrics.txt"

# 创建指标记录文件并写入配置
echo "========================================" > "$METRICS_FILE"
echo "训练参数配置" >> "$METRICS_FILE"
echo "========================================" >> "$METRICS_FILE"
echo "训练时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$METRICS_FILE"
echo "GPU设备: $GPU_ID" >> "$METRICS_FILE"
echo "运行目录: $RUN_DIR" >> "$METRICS_FILE"
echo "日志目录: $LOG_BASE_DIR" >> "$METRICS_FILE"
echo "检查点目录: $CHECKPOINT_BASE_DIR" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 数据集选择 ---" >> "$METRICS_FILE"
echo "DATASET=\"$DATASET\"" >> "$METRICS_FILE"
echo "DATA_DIR=\"$DATA_DIR\"" >> "$METRICS_FILE"
echo "FULL_DATA_DIR=\"$FULL_DATA_DIR\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- GPU设置 ---" >> "$METRICS_FILE"
echo "GPU_ID=$GPU_ID" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 基础训练参数 ---" >> "$METRICS_FILE"
echo "SEQ_LENGTH=\"$SEQ_LENGTH\"" >> "$METRICS_FILE"
echo "BATCH_SIZE=\"$BATCH_SIZE\"" >> "$METRICS_FILE"
echo "LEARNING_RATE=\"$LEARNING_RATE\"" >> "$METRICS_FILE"
echo "NUM_EPOCHS=\"$NUM_EPOCHS\"" >> "$METRICS_FILE"
echo "HIDDEN_DIM=\"$HIDDEN_DIM\"" >> "$METRICS_FILE"
echo "DROPOUT=\"$DROPOUT\"" >> "$METRICS_FILE"
echo "EARLY_STOP_PATIENCE=\"$EARLY_STOP_PATIENCE\"" >> "$METRICS_FILE"
echo "EARLY_STOP_METRIC=\"$EARLY_STOP_METRIC\"" >> "$METRICS_FILE"
echo "SPHERE_LOSS_WEIGHT=\"$SPHERE_LOSS_WEIGHT\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 测试集评估设置 ---" >> "$METRICS_FILE"
echo "EVAL_TEST_EVERY_EPOCH=$EVAL_TEST_EVERY_EPOCH" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 学习率调度器参数 ---" >> "$METRICS_FILE"
echo "SCHEDULER_TYPE=\"$SCHEDULER_TYPE\"" >> "$METRICS_FILE"
echo "SCHEDULER_GAMMA=\"$SCHEDULER_GAMMA\"" >> "$METRICS_FILE"
echo "SCHEDULER_PATIENCE=\"$SCHEDULER_PATIENCE\"" >> "$METRICS_FILE"
echo "SCHEDULER_STEP_SIZE=\"$SCHEDULER_STEP_SIZE\"" >> "$METRICS_FILE"
echo "WARMUP_RATIO=\"$WARMUP_RATIO\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 关键帧选择（MDP3）参数 ---" >> "$METRICS_FILE"
echo "N_KEY_FRAMES=\"$N_KEY_FRAMES\"" >> "$METRICS_FILE"
echo "KEY_FRAME_SEGMENT=\"$KEY_FRAME_SEGMENT\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- MoE-FiLM 参数 ---" >> "$METRICS_FILE"
echo "NUM_FILM_EXPERTS=\"$NUM_FILM_EXPERTS\"" >> "$METRICS_FILE"
echo "FILM_TOP_K=\"$FILM_TOP_K\"" >> "$METRICS_FILE"
echo "MOE_LOSS_WEIGHT=\"$MOE_LOSS_WEIGHT\"" >> "$METRICS_FILE"
echo "CHSIMSV2_MTL_LAMBDA=\"$CHSIMSV2_MTL_LAMBDA\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 超图建模（M3NET）参数 ---" >> "$METRICS_FILE"
echo "NUM_HG_LAYERS=\"$NUM_HG_LAYERS\"" >> "$METRICS_FILE"
echo "HG_USE_RESIDUE=$HG_USE_RESIDUE" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 频域分解（GS-MCC / FGN）参数 ---" >> "$METRICS_FILE"
echo "NUM_FOURIER_LAYERS=\"$NUM_FOURIER_LAYERS\"" >> "$METRICS_FILE"
echo "FGN_USE_RESIDUE=$FGN_USE_RESIDUE" >> "$METRICS_FILE"
echo "F_SPARSITY_THRESHOLD=\"$F_SPARSITY_THRESHOLD\"" >> "$METRICS_FILE"
echo "F_HIDDEN_SIZE_FACTOR=\"$F_HIDDEN_SIZE_FACTOR\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- KL散度多任务学习（GS-MCC）参数 ---" >> "$METRICS_FILE"
echo "USE_KL_MTL=$USE_KL_MTL" >> "$METRICS_FILE"
echo "KL_MTL_WEIGHT=\"$KL_MTL_WEIGHT\"" >> "$METRICS_FILE"
echo "UNIMODAL_LOSS_WEIGHT=\"$UNIMODAL_LOSS_WEIGHT\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 模态和关键帧分析参数 ---" >> "$METRICS_FILE"
echo "ENABLE_MODALITY_ANALYSIS=$ENABLE_MODALITY_ANALYSIS" >> "$METRICS_FILE"
echo "ANALYZE_MODALITY_EVERY=\"$ANALYZE_MODALITY_EVERY\"" >> "$METRICS_FILE"
echo "KEYFRAME_LOG_EVERY=\"$KEYFRAME_LOG_EVERY\"" >> "$METRICS_FILE"
echo "MODALITY_ANALYSIS_EPOCHS=\"$MODALITY_ANALYSIS_EPOCHS\"" >> "$METRICS_FILE"
echo "ENABLE_KEYFRAME_LOGGING=$ENABLE_KEYFRAME_LOGGING" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 组件开关 ---" >> "$METRICS_FILE"
echo "NO_KEY_FRAME_SELECTOR=$NO_KEY_FRAME_SELECTOR" >> "$METRICS_FILE"
echo "NO_COUPLED_MAMBA=$NO_COUPLED_MAMBA" >> "$METRICS_FILE"
echo "NO_MOE_FILM=$NO_MOE_FILM" >> "$METRICS_FILE"
echo "NO_DSPS=$NO_DSPS" >> "$METRICS_FILE"
echo "DSPS_STRENGTH=$DSPS_STRENGTH" >> "$METRICS_FILE"
echo "NO_HYPERGRAPH=$NO_HYPERGRAPH" >> "$METRICS_FILE"
echo "NO_FREQUENCY_DECOMP=$NO_FREQUENCY_DECOMP" >> "$METRICS_FILE"
echo "NO_SPHERE_REG=$NO_SPHERE_REG" >> "$METRICS_FILE"
echo "NO_DIRECT_FUSION_PRIORS=$NO_DIRECT_FUSION_PRIORS" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 类别权重设置 ---" >> "$METRICS_FILE"
echo "USE_CLASS_WEIGHTS=\"$USE_CLASS_WEIGHTS\"" >> "$METRICS_FILE"
echo "USE_FOCAL_LOSS=$USE_FOCAL_LOSS" >> "$METRICS_FILE"
echo "FOCAL_LOSS_GAMMA=\"$FOCAL_LOSS_GAMMA\"" >> "$METRICS_FILE"
echo "FOCAL_DYNAMIC_GAMMA=$FOCAL_DYNAMIC_GAMMA" >> "$METRICS_FILE"
echo "FOCAL_GAMMA_MIN=\"$FOCAL_GAMMA_MIN\"" >> "$METRICS_FILE"
echo "FOCAL_GAMMA_DECAY_MODE=\"$FOCAL_GAMMA_DECAY_MODE\"" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- MLP架构选择 ---" >> "$METRICS_FILE"
echo "USE_IMPROVED_MLP=$USE_IMPROVED_MLP" >> "$METRICS_FILE"
echo "MLP_DROPOUT=$MLP_DROPOUT" >> "$METRICS_FILE"
echo "MLP_EXPANSION_RATIO=$MLP_EXPANSION_RATIO" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 混合回放池 ---" >> "$METRICS_FILE"
echo "USE_REPLAY_BUFFER=$USE_REPLAY_BUFFER" >> "$METRICS_FILE"
echo "REPLAY_BUFFER_THRESHOLD=$REPLAY_BUFFER_THRESHOLD" >> "$METRICS_FILE"
echo "REPLAY_BUFFER_RATIO=$REPLAY_BUFFER_RATIO" >> "$METRICS_FILE"
echo "REPLAY_BUFFER_MAX_SIZE=$REPLAY_BUFFER_MAX_SIZE" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 课程学习 (Curriculum Learning) 设置 ---" >> "$METRICS_FILE"
echo "CURRICULUM_MODE=\"$CURRICULUM_MODE\"" >> "$METRICS_FILE"
echo "CURRICULUM_EPOCHS=$CURRICULUM_EPOCHS" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 训练曲线绘图设置 ---" >> "$METRICS_FILE"
echo "ENABLE_PLOTTING=$ENABLE_PLOTTING" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 后台运行设置 ---" >> "$METRICS_FILE"
echo "USE_NOHUP=$USE_NOHUP" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "# --- 其他参数 ---" >> "$METRICS_FILE"
echo "NUM_WORKERS=$NUM_WORKERS" >> "$METRICS_FILE"
echo "SEED=$SEED" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

echo "========================================" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"
echo "训练指标记录" >> "$METRICS_FILE"
echo "========================================" >> "$METRICS_FILE"
echo "" >> "$METRICS_FILE"

# 构建绘图参数（简化为单个开关）
PLOT_ARGS=""
[ "$ENABLE_PLOTTING" = true ] && PLOT_ARGS="--enable_plotting"

# 训练命令（使用本次训练的独立目录）
TRAIN_CMD="python train_refactored.py $COMMON_ARGS $DATASET_ARGS --metrics_file $METRICS_FILE --save_dir $CHECKPOINT_BASE_DIR --log_dir $LOG_BASE_DIR $PLOT_ARGS $@"

# 根据配置选择运行方式
if [ "$USE_NOHUP" = true ]; then
    # ========== 后台运行模式 ==========
    echo "=========================================="
    echo "后台训练模式（nohup）"
    echo "=========================================="
    echo "日志文件: $LOG_FILE"
    echo ""
    
    # 启动后台训练
    # 注意：不使用 conda run，因为：
    # 1. 脚本开头已经验证并激活了 conda 环境
    # 2. conda run 会 fork 子进程，导致 $! 捕获的 PID 不是实际 Python 进程的 PID
    # 3. 直接运行可以正确捕获真正的训练进程 PID
    nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    
    # 等待一下确认启动成功
    sleep 3
    
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "✅ 训练已在后台启动！"
        echo ""
        echo "=========================================="
        echo "后台训练信息"
        echo "=========================================="
        echo "GPU设备: $GPU_ID"
        echo "进程ID (PID): $TRAIN_PID"
        echo "运行目录: $RUN_DIR"
        echo ""
        echo "📁 输出文件位置:"
        echo "   日志: $LOG_FILE"
        echo "   指标: $METRICS_FILE"
        echo "   模型: $CHECKPOINT_BASE_DIR/"
        echo "   曲线图: $CHECKPOINT_BASE_DIR/figures/"
        echo ""
        echo "实用命令:"
        echo "  # 查看实时日志"
        echo "  tail -f $LOG_FILE"
        echo ""
        echo "  # 查看训练进度（最后20行）"
        echo "  tail -20 $LOG_FILE"
        echo ""
        echo "  # 查看训练指标（精简版）"
        echo "  tail -50 $METRICS_FILE"
        echo ""
        echo "  # 检查进程状态"
        echo "  ps -p $TRAIN_PID"
        echo ""
        echo "  # 停止训练"
        echo "  kill $TRAIN_PID"
        echo ""
        echo "  # 如果PID失效，用这个命令查找实际进程"
        echo "  pgrep -f 'train_refactored.py.*$RUN_DIR'"
        echo ""
        echo "  # 查看GPU使用"
        echo "  watch -n 1 nvidia-smi"
        echo ""
        echo "=========================================="
        echo "💡 多GPU并行训练提示:"
        echo "   可以修改GPU_ID后再次运行此脚本"
        echo "   不同GPU的训练会自动隔离，互不干扰"
        echo "=========================================="
        echo "提示: 现在可以安全地断开SSH连接"
        echo "      训练将在后台继续运行"
        echo "=========================================="
        
        # 保存PID到文件
        echo $TRAIN_PID > "$PID_FILE"
        echo ""
        echo "PID已保存到: $PID_FILE"
    else
        echo ""
        echo "=========================================="
        echo "❌ 启动训练失败"
        echo "=========================================="
        echo "请检查日志: $LOG_FILE"
        exit 1
    fi
else
    # ========== 前台运行模式 ==========
    echo "=========================================="
    echo "前台训练模式"
    echo "=========================================="
    echo "提示: SSH断开会导致训练中断"
    echo "      如需后台运行，设置 USE_NOHUP=true"
    echo ""
    
    # 前台执行训练
    $TRAIN_CMD
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "训练成功完成！"
        echo "=========================================="
        echo "📁 结果保存位置:"
        echo "   日志: $LOG_BASE_DIR"
        echo "   模型: $CHECKPOINT_BASE_DIR"
        echo "   曲线图: $CHECKPOINT_BASE_DIR/figures/"
        echo "   指标历史: $CHECKPOINT_BASE_DIR/metrics_history.json"
        echo ""
        echo "评估模型:"
        echo "  bash eval_refactored.sh $DATASET"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "训练失败！请检查上面的错误信息。"
        echo "=========================================="
        exit 1
    fi
fi
