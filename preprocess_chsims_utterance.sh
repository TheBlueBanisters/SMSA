#!/bin/bash
# CH-SIMS 数据集预处理脚本 - Utterance 级别特征版本
# 每个 utterance 提取一个特征向量（非帧级别）
# 只提取三种基本模态：视觉、音频、文本

# 配置参数
DATASET="chsims"
INPUT_DIR="./chsims"
OUTPUT_DIR="./data_unified/chsims_utterance"

# 模型路径
CLIP_MODEL="ViT-g-14"
CLIP_CKPT="./models/open_clip/vit_g14_laion2b/open_clip_pytorch_model.bin"
WAVLM_PATH="./models/wavlm/wavlm-large"
TEXT_MODEL_PATH="./models/sent/paraphrase-multilingual-mpnet-base-v2"

# 处理参数
DEVICE="cuda:1"
FRAME_RATE=3.0
BATCH_SIZE_IMG=32
TARGET_SR=16000

# 创建日志目录
LOG_DIR="./logs/preprocess_${DATASET}_utterance"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/preprocess_${TIMESTAMP}.log"

echo "========================================"
echo "CH-SIMS 数据集预处理 - Utterance 级别"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "特征级别: Utterance 级别（单个向量）"
echo "提取模态: 视觉 + 音频 + 文本"
echo "日志文件: $LOG_FILE"
echo "========================================"

PREPROCESS_CMD=(
    python preprocess_unified_utterance.py
    --dataset $DATASET
    --input_dir $INPUT_DIR
    --output_dir $OUTPUT_DIR
    --clip_model $CLIP_MODEL
    --clip_ckpt $CLIP_CKPT
    --wavlm_path $WAVLM_PATH
    --text_model_path $TEXT_MODEL_PATH
    --device $DEVICE
    --frame_rate $FRAME_RATE
    --batch_size_img $BATCH_SIZE_IMG
    --target_sr $TARGET_SR
)

echo "启动命令: ${PREPROCESS_CMD[*]}"
echo ""

nohup "${PREPROCESS_CMD[@]}" >> "$LOG_FILE" 2>&1 &
PREPROCESS_PID=$!

sleep 2

if ps -p $PREPROCESS_PID >/dev/null 2>&1; then
    echo "========================================="
    echo "✅ 预处理进程已在后台运行"
    echo "========================================="
    echo "PID: $PREPROCESS_PID"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "实用命令:"
    echo "  tail -f $LOG_FILE           # 查看实时日志"
    echo "  ps -p $PREPROCESS_PID       # 检查进程状态"
    echo "  kill $PREPROCESS_PID        # 终止进程"
    echo ""
    echo "预处理将在 SSH 断开后继续运行。"
    echo ""
    echo "说明:"
    echo "  - 每个 utterance 提取一个特征向量（非帧级别）"
    echo "  - 视觉特征: 对所有采样帧取平均"
    echo "  - 音频特征: 对所有音频片段取平均"
    echo "  - 文本特征: 直接编码"
    exit 0
else
    echo "========================================="
    echo "❌ 启动预处理进程失败"
    echo "========================================="
    echo "请检查日志: $LOG_FILE"
    exit 1
fi

