#!/bin/bash

# 模型仓库名称
# MODEL_NAME="laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
# DIR_PATH="models/open_clip/vit_g14_laion2b"

# 修改为 BERT 中文模型
MODEL_NAME="google-bert/bert-base-chinese"
DIR_PATH="models/bert-base-chinese"


# 使用 HF Mirror 镜像站（国内访问更快）
# HF_ENDPOINT="https://hf-mirror.com"
HF_ENDPOINT="https://huggingface.com"

# API 获取文件列表
API_URL="${HF_ENDPOINT}/api/models/${MODEL_NAME}"

# 创建目标目录
mkdir -p "${DIR_PATH}"

echo "📥 正在获取文件列表: ${MODEL_NAME} ..."
FILES=$(curl -sL "${API_URL}" | jq -r '.siblings[].rfilename')

if [ -z "$FILES" ]; then
  echo "❌ 未能获取到文件列表，请检查模型名称是否正确。"
  exit 1
fi

# 生成 aria2 下载列表
LIST_FILE="download_list.txt"
rm -f "$LIST_FILE"

for file in $FILES; do
  echo "${HF_ENDPOINT}/${MODEL_NAME}/resolve/main/${file}" >> "$LIST_FILE"
  echo "  dir=${DIR_PATH}" >> "$LIST_FILE"
  echo "  out=${file}" >> "$LIST_FILE"
done

# 用 aria2c 批量下载，支持断点续传
echo "🚀 开始下载..."
aria2c -c -x 16 -s 16 -i "$LIST_FILE"

echo "✅ 模型已下载完成，路径: ${DIR_PATH}"
