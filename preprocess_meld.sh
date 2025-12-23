#!/bin/bash
# MELD 数据集预处理脚本
# 使用方法:
#   1. 直接运行（使用默认路径）: bash preprocess_meld.sh
#   2. 指定输入目录: bash preprocess_meld.sh --input ./path/to/MELD
#   3. 指定输出目录: bash preprocess_meld.sh --output ./path/to/output
#   4. 同时指定: bash preprocess_meld.sh --input ./MELD --output ./data/meld_new

# 配置参数
DATASET="meld"
INPUT_DIR="./MELD"                  # MELD原始数据目录（包含train/dev/test子目录和csv文件）
OUTPUT_DIR="./meld_10"  # 处理后的数据输出目录

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
            echo "用法: bash preprocess_meld.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --input, -i DIR    MELD原始数据目录（默认：./MELD）"
            echo "  --output, -o DIR   输出目录（默认：./data/meld_processed）"
            echo "  --help, -h         显示帮助信息"
            echo ""
            echo "示例:"
            echo "  bash preprocess_meld.sh"
            echo "  bash preprocess_meld.sh --input ./MELD_raw --output ./data/meld_v2"
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
DEVICE="cuda:1"
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
SOCIAL_PROMPT="Please describe the social relationships in the scene using concise natural language, including explicit relationship types (e.g., family members, friends, colleagues, etc.) as well as implicit relationship characteristics (e.g., power dynamics between characters, emotional intimacy, frequency of interaction, etc.). Output in a natural paragraph without bullet points, approximately 100 words. Please answer in English."
CONTEXT_PROMPT="Please describe the background and atmosphere of the current situation using concise natural language, including: the time, place, type of event or activity, and the atmosphere of the communication (e.g., emotional tone of the conversation, degree of psychological tension, whether it is relaxed and casual or tense and formal, whether it has a ceremonial nature, etc.). Output in a natural paragraph without bullet points, approximately 100 words. Please answer in English."

# 示例：自定义提示词
# SOCIAL_PROMPT="Analyze the social relationships in this video: who are the characters and how do they interact?"
# CONTEXT_PROMPT="Describe the contextual setting and atmosphere of this video scene."

# 创建日志目录
LOG_DIR="./logs/preprocess_${DATASET}"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/preprocess_${TIMESTAMP}.log"

echo "========================================"
echo "MELD 数据集预处理"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "提取模式: $EXTRACT_MODE"
echo "标签类型: Emotion（7分类情感识别）"
echo "  - neutral(0), joy(1), sadness(2), anger(3)"
echo "  - surprise(4), fear(5), disgust(6)"
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

