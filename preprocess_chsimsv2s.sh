#!/bin/bash
# CH-SIMSV2 数据集预处理脚本
# 使用方法:
#   1. 直接运行（使用默认路径）: bash preprocess_chsimsv2s.sh
#   2. 指定输入目录: bash preprocess_chsimsv2s.sh --input ./path/to/chsimsv2
#   3. 指定输出目录: bash preprocess_chsimsv2s.sh --output ./path/to/output
#   4. 同时指定: bash preprocess_chsimsv2s.sh --input ./chsimsv2 --output ./data/chsimsv2_new

# 配置参数
DATASET="chsimsv2"
INPUT_DIR="./chsimsv2/ch-simsv2s"           # CH-SIMSV2原始数据目录
OUTPUT_DIR="./chsimsv2_10"     # 处理后的数据输出目录

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash preprocess_chsimsv2s.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --input, -i DIR    CH-SIMSV2原始数据目录（默认：./chsimsv2）"
            echo "  --output, -o DIR   输出目录（默认：./data/chsimsv2_processed）"
            echo "  --help, -h         显示帮助信息"
            echo ""
            echo "示例:"
            echo "  bash preprocess_chsimsv2s.sh"
            echo "  bash preprocess_chsimsv2s.sh --input ./chsimsv2_raw --output ./data/chsimsv2_v2"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 模型路径
CLIP_MODEL="ViT-g-14"
CLIP_CKPT="./models/open_clip/vit_g14_laion2b/open_clip_pytorch_model.bin"
WAVLM_PATH="./models/wavlm/wavlm-large"
TEXT_MODEL_PATH="./models/sent/paraphrase-multilingual-mpnet-base-v2"

# 处理参数
DEVICE="cuda:0"
FRAME_RATE=10.0
BATCH_SIZE_IMG=32
TARGET_SR=16000

# MLLM 参数（用于社会关系和情境特征提取）
API_KEY="sk-CopXuPMUxmJY7UNSXrjyBA"
BASE_URL="https://llm.rekeymed.com/v1/"
MODEL_NAME="Qwen/Qwen3-Omni-30B-A3B"

# 提取模式：all=完整提取, basic=仅基本模态, social=仅社会关系, context=仅情境, prior=社会关系+情境
EXTRACT_MODE="all"

# 提示词模板（可选，留空则使用默认）
SOCIAL_PROMPT="请以简洁的自然语言描述场景中的社会关系，包括显性的关系类型（例如：家庭成员、朋友、同事等）以及隐性的关系特征（例如：角色间的权力高低、情感亲密程度、互动频率等）。用不带格式分点的自然文段输出，100字左右。"
CONTEXT_PROMPT="请以简洁的自然语言描述当前情境的背景和氛围，包括：发生的时间、地点、事件或活动类型，以及交流的气氛（例如对话的情感基调、心理张力程度，是轻松随意还是紧张正式，是否带有仪式性等）。用不带格式分点的自然文段输出，100字左右。"

# 示例：自定义中文提示词
# SOCIAL_PROMPT="请分析这个视频中的社会关系：角色之间的关系、互动模式、社交场景等。用简洁的中文描述，不超过150字。"
# CONTEXT_PROMPT="请分析这个视频的情境信息：场景环境、整体氛围、事件背景等。用简洁的中文描述，不超过150字。"

# 创建日志目录
LOG_DIR="./logs/preprocess_${DATASET}"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/preprocess_${TIMESTAMP}.log"

echo "========================================"
echo "CH-SIMSV2S 数据集预处理"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "提取模式: $EXTRACT_MODE"
echo "先验文本CSV: ${OUTPUT_DIR}/prior_texts/"
echo "日志文件: $LOG_FILE"
echo "========================================"

PREPROCESS_CMD=(
    python preprocess_unified.py
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
    --api_key $API_KEY
    --base_url $BASE_URL
    --model_name $MODEL_NAME
    --extract_mode $EXTRACT_MODE
)

# 添加自定义提示词（如果设置了）
if [ -n "$SOCIAL_PROMPT" ]; then
    PREPROCESS_CMD+=(--social_prompt "$SOCIAL_PROMPT")
    echo "使用自定义社会关系提示词"
fi

if [ -n "$CONTEXT_PROMPT" ]; then
    PREPROCESS_CMD+=(--context_prompt "$CONTEXT_PROMPT")
    echo "使用自定义情境提示词"
fi

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
    exit 0
else
    echo "========================================="
    echo "❌ 启动预处理进程失败"
    echo "========================================="
    echo "请检查日志: $LOG_FILE"
    exit 1
fi

