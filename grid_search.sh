#!/bin/bash

# ========================================
# Grid Search 超参数搜索脚本
# ========================================
# 使用方法:
#   1. 单参数搜索:
#      bash grid_search.sh --param dropout --values "0.1 0.2 0.3 0.4 0.5"
#      bash grid_search.sh --param batch_size --values "8 16 32"
#      bash grid_search.sh --param learning_rate --values "1e-4 5e-5 1e-5"
#      bash grid_search.sh --param num_hg_layers --values "4 6 8 10"
#
#   2. 成对参数搜索（两个列表需要元素个数一致）:
#      bash grid_search.sh --paired --param1 n_segments --param2 frame_ratio \
#                          --values1 "2 4 6" --values2 "50 70 90"
#      bash grid_search.sh --paired --param1 num_film_experts --param2 film_top_k \
#                          --values1 "8 16 32" --values2 "4 8 16"
#
#   3. 指定监控指标:
#      --metric acc          # 监控准确率 Acc/Acc_2（越大越好）
#      --metric f1           # 监控F1分数 F1_weighted/F1_2（越大越好）
#      --metric mae          # 监控MAE（越小越好，仅回归任务）
#      --metric corr         # 监控相关系数Corr（越大越好，仅回归任务）
#      --metric combined     # 综合指标（默认，自动适配分类/回归任务）
#
#   4. 指定数据集:
#      --dataset meld        # 默认使用 train_unified.sh 中的设置
#
#   5. 指定GPU:
#      --gpu 0               # 默认使用 GPU 0
#
#   6. 运行模式:
#      # ⭐ 默认后台运行（推荐，SSH断开后继续运行）
#      bash grid_search.sh --param dropout --values "0.1 0.2 0.3"
#      
#      # 前台运行（调试用）
#      bash grid_search.sh --foreground --param dropout --values "0.1 0.2 0.3"
#
#   7. 停止 Grid Search:
#      # 方法1: 使用 PID
#      kill $(cat ./logs/xxx/grid_search.pid)
#      
#      # 方法2: 使用进程组（杀死所有子进程）
#      kill -- -$(cat ./logs/xxx/grid_search.pgid)
#      
#      # 方法3: 使用停止脚本
#      bash stop_grid_search.sh ./logs/xxx/
#
# ========================================

# 不使用 set -e，因为我们需要手动处理子进程的退出状态
# set -e

# ============================================================
# 子进程管理和信号处理
# ============================================================
# 核心思路：
# 1. grid_search 主进程管理所有训练子进程
# 2. 使用进程组 (PGID) 来统一管理
# 3. 当主进程被杀时，自动杀死所有子进程
# ============================================================

set -m  # 启用作业控制

# 全局变量
CHILD_PID=""                    # 当前正在运行的训练子进程 PID
CHILD_PIDS=()                   # 所有启动过的子进程 PID
GRID_SEARCH_PID=$$              # 主进程 PID
GRID_SEARCH_PGID=""             # 主进程的进程组 ID
CLEANUP_DONE=false              # 防止重复清理

# 获取当前进程组 ID
get_pgid() {
    ps -o pgid= -p $$ 2>/dev/null | tr -d ' '
}

# 清理函数：终止所有子进程
cleanup() {
    # 防止重复清理
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    local exit_code=${1:-130}
    
    echo ""
    echo "=========================================="
    echo "⚠️  收到终止信号，正在清理所有子进程..."
    echo "=========================================="
    echo "主进程 PID: $$"
    echo "进程组 PGID: $GRID_SEARCH_PGID"
    
    # 方法1: 终止当前正在运行的训练子进程
    if [ -n "$CHILD_PID" ]; then
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "终止当前训练子进程 (PID: $CHILD_PID)..."
            
            # 首先尝试终止子进程的所有后代
            local descendants=$(pgrep -P "$CHILD_PID" 2>/dev/null || true)
            if [ -n "$descendants" ]; then
                echo "  终止后代进程: $descendants"
                echo "$descendants" | xargs -r kill -TERM 2>/dev/null || true
            fi
            
            # 然后终止子进程本身
            kill -TERM "$CHILD_PID" 2>/dev/null || true
            
            # 等待最多5秒
            local wait_count=0
            while kill -0 "$CHILD_PID" 2>/dev/null && [ $wait_count -lt 5 ]; do
                sleep 1
                wait_count=$((wait_count + 1))
            done
            
            # 强制终止
            if kill -0 "$CHILD_PID" 2>/dev/null; then
                echo "  强制终止..."
                kill -9 "$CHILD_PID" 2>/dev/null || true
                [ -n "$descendants" ] && echo "$descendants" | xargs -r kill -9 2>/dev/null || true
            fi
        fi
    fi
    
    # 方法2: 终止所有记录的子进程
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止已记录的子进程: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # 方法3: 终止此进程的所有直接子进程
    local children=$(pgrep -P $$ 2>/dev/null || true)
    if [ -n "$children" ]; then
        echo "终止直接子进程: $children"
        echo "$children" | xargs -r kill -TERM 2>/dev/null || true
        sleep 1
        echo "$children" | xargs -r kill -9 2>/dev/null || true
    fi
    
    # 方法4: 根据 PID 文件中记录的进程组 ID 终止（针对 nohup 模式）
    if [ -n "$GRID_SEARCH_PGID" ] && [ "$GRID_SEARCH_PGID" != "$$" ]; then
        echo "终止进程组 $GRID_SEARCH_PGID 中的所有进程..."
        kill -TERM -"$GRID_SEARCH_PGID" 2>/dev/null || true
        sleep 1
        kill -9 -"$GRID_SEARCH_PGID" 2>/dev/null || true
    fi
    
    # 方法5: 查找并终止所有由此 grid_search 启动的 Python 训练进程
    # 使用更精确的匹配，只终止在我们的实验目录中的进程
    if [ -n "$GRID_SEARCH_DIR" ]; then
        local our_python_pids=$(pgrep -f "train_refactored.py.*$GRID_SEARCH_DIR" 2>/dev/null || true)
        if [ -n "$our_python_pids" ]; then
            echo "终止相关 Python 进程: $our_python_pids"
            echo "$our_python_pids" | xargs -r kill -TERM 2>/dev/null || true
            sleep 1
            echo "$our_python_pids" | xargs -r kill -9 2>/dev/null || true
        fi
    fi
    
    echo "清理完成"
    
    # 删除 PID 文件
    [ -n "$PID_FILE" ] && [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    
    exit $exit_code
}

# 注册信号处理（只处理中断信号，不处理 EXIT，避免正常结束时误杀进程）
trap 'cleanup 130' SIGINT   # Ctrl+C
trap 'cleanup 143' SIGTERM  # kill 命令
trap 'cleanup 129' SIGHUP   # 终端断开

# 记录进程组 ID
GRID_SEARCH_PGID=$(get_pgid)

# 正常结束时的清理（只删除 PID 文件，不杀进程）
normal_exit_cleanup() {
    [ -n "$PID_FILE" ] && [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    [ -n "$PGID_FILE" ] && [ -f "$PGID_FILE" ] && rm -f "$PGID_FILE"
}

# ============================================================
# 默认配置
# ============================================================
PAIRED_MODE=false
PARAM="batch_size"
PARAM1=""
PARAM2=""
VALUES="8 16 32 64"
VALUES1=""
VALUES2=""
METRIC="acc"  # acc, f1, mae, combined
DATASET=""  # 留空则使用train_unified.sh中的默认值
GPU_ID=2
DRY_RUN=false  # 干运行模式，只打印命令不执行
USE_NOHUP=true   # ⭐ 默认后台运行（SSH断开后继续运行）
FOREGROUND=false # 前台运行模式（调试用）
NOHUP_INTERNAL=false  # 内部标记，用于区分是否已经在nohup子进程中
OUTPUT_DIR=""    # 内部使用：nohup 启动时传入的输出目录

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --paired)
            PAIRED_MODE=true
            shift
            ;;
        --param)
            PARAM="$2"
            shift 2
            ;;
        --param1)
            PARAM1="$2"
            shift 2
            ;;
        --param2)
            PARAM2="$2"
            shift 2
            ;;
        --values)
            VALUES="$2"
            shift 2
            ;;
        --values1)
            VALUES1="$2"
            shift 2
            ;;
        --values2)
            VALUES2="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --nohup)
            # 保留兼容性，但现在默认就是后台运行
            USE_NOHUP=true
            shift
            ;;
        --foreground|--fg)
            # 前台运行模式（调试用）
            USE_NOHUP=false
            FOREGROUND=true
            shift
            ;;
        --nohup-internal)
            # 内部使用：标记已经在nohup子进程中运行
            NOHUP_INTERNAL=true
            shift
            ;;
        --output-dir)
            # 内部使用：指定输出目录（避免重复创建）
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash grid_search.sh [OPTIONS]"
            echo ""
            echo "单参数搜索:"
            echo "  --param NAME          参数名称 (dropout, batch_size, learning_rate, num_hg_layers)"
            echo "  --values \"V1 V2 ...\"  参数值列表（空格分隔）"
            echo ""
            echo "成对参数搜索:"
            echo "  --paired              启用成对参数模式"
            echo "  --param1 NAME         第一个参数名称"
            echo "  --param2 NAME         第二个参数名称"
            echo "  --values1 \"V1 V2 ...\" 第一个参数值列表"
            echo "  --values2 \"V1 V2 ...\" 第二个参数值列表（元素个数需与values1一致）"
            echo ""
            echo "通用选项:"
            echo "  --metric MODE         监控指标: acc, f1, mae, corr, combined (默认: acc)"
            echo "  --dataset NAME        数据集名称 (默认使用train_unified.sh中的设置)"
            echo "  --gpu ID              GPU编号 (默认: 0)"
            echo "  --foreground, --fg    前台运行模式（调试用，默认是后台运行）"
            echo "  --dry-run             干运行模式，只打印命令不执行"
            echo "  -h, --help            显示帮助信息"
            echo ""
            echo "⭐ 默认行为："
            echo "  脚本默认在后台运行，SSH断开后继续执行"
            echo "  可以用 kill PID 或 kill -- -PGID 停止"
            echo ""
            echo "支持的单参数:"
            echo "  dropout, batch_size, learning_rate, num_hg_layers"
            echo ""
            echo "支持的成对参数:"
            echo "  n_segments + frame_ratio"
            echo "  num_film_experts + film_top_k"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Nohup 后台运行处理
# ============================================================
if [ "$USE_NOHUP" = true ] && [ "$NOHUP_INTERNAL" = false ]; then
    # 获取数据集名称（用于日志目录）
    if [ -n "$DATASET" ]; then
        DS_NAME_FOR_LOG="$DATASET"
    else
        DS_NAME_FOR_LOG=$(grep '^DATASET=' train_unified.sh 2>/dev/null | head -1 | cut -d'"' -f2)
        [ -z "$DS_NAME_FOR_LOG" ] && DS_NAME_FOR_LOG="unknown"
    fi
    
    # 生成搜索名称
    if [ "$PAIRED_MODE" = true ]; then
        SEARCH_NAME_FOR_LOG="${PARAM1}_${PARAM2}"
    else
        SEARCH_NAME_FOR_LOG="${PARAM}"
    fi
    
    # 创建日志目录
    TIMESTAMP_FOR_LOG=$(date +%Y%m%d_%H%M%S)
    NOHUP_LOG_DIR="./logs/${DS_NAME_FOR_LOG}/grid_search_${SEARCH_NAME_FOR_LOG}_${TIMESTAMP_FOR_LOG}"
    mkdir -p "$NOHUP_LOG_DIR"
    
    NOHUP_LOG="${NOHUP_LOG_DIR}/grid_search.log"
    PID_FILE="${NOHUP_LOG_DIR}/grid_search.pid"
    PGID_FILE="${NOHUP_LOG_DIR}/grid_search.pgid"
    
    echo ""
    echo "=========================================="
    echo "🚀 后台运行模式 (nohup)"
    echo "=========================================="
    echo "日志目录: $NOHUP_LOG_DIR"
    echo "日志文件: $NOHUP_LOG"
    echo ""
    
    # 重建命令行参数（添加 --nohup-internal 标记）
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    WORK_DIR="$(pwd)"
    
    # 构建参数数组（避免引号问题）
    SCRIPT_ARGS=()
    [ "$PAIRED_MODE" = true ] && SCRIPT_ARGS+=("--paired")
    [ -n "$PARAM" ] && SCRIPT_ARGS+=("--param" "$PARAM")
    [ -n "$PARAM1" ] && SCRIPT_ARGS+=("--param1" "$PARAM1")
    [ -n "$PARAM2" ] && SCRIPT_ARGS+=("--param2" "$PARAM2")
    [ -n "$VALUES" ] && SCRIPT_ARGS+=("--values" "$VALUES")
    [ -n "$VALUES1" ] && SCRIPT_ARGS+=("--values1" "$VALUES1")
    [ -n "$VALUES2" ] && SCRIPT_ARGS+=("--values2" "$VALUES2")
    [ -n "$METRIC" ] && SCRIPT_ARGS+=("--metric" "$METRIC")
    [ -n "$DATASET" ] && SCRIPT_ARGS+=("--dataset" "$DATASET")
    SCRIPT_ARGS+=("--gpu" "$GPU_ID")
    [ "$DRY_RUN" = true ] && SCRIPT_ARGS+=("--dry-run")
    SCRIPT_ARGS+=("--nohup-internal")
    SCRIPT_ARGS+=("--output-dir" "$NOHUP_LOG_DIR")  # 传递目录给内部脚本
    
    # 使用 setsid 创建新的会话和进程组，这样可以通过 PGID 统一管理
    # nohup + setsid 确保：
    # 1. SSH 断开后进程继续运行
    # 2. 所有子进程都属于同一个进程组
    # 3. 可以通过 kill -PGID 一次性杀死所有进程
    
    CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
    if [ -n "$CONDA_ENV" ] && [ "$CONDA_ENV" != "base" ]; then
        echo "将在 conda 环境 '$CONDA_ENV' 中运行"
        # 使用 setsid 创建新会话，并记录 PGID
        setsid bash -c "
            cd '$WORK_DIR'
            source ~/.bashrc 2>/dev/null
            conda activate $CONDA_ENV 2>/dev/null
            # 写入 PGID（新会话中 PGID 等于 PID）
            echo \$\$ > '$PGID_FILE'
            exec bash '$SCRIPT_PATH' ${SCRIPT_ARGS[*]@Q}
        " > "$NOHUP_LOG" 2>&1 &
        NOHUP_PID=$!
    else
        setsid bash -c "
            cd '$WORK_DIR'
            echo \$\$ > '$PGID_FILE'
            exec bash '$SCRIPT_PATH' ${SCRIPT_ARGS[*]@Q}
        " > "$NOHUP_LOG" 2>&1 &
        NOHUP_PID=$!
    fi
    
    # 保存启动时的 PID（可能不是实际的进程 PID，因为 setsid 会 fork）
    echo $NOHUP_PID > "$PID_FILE"
    
    # 等待 PGID 文件生成
    sleep 2
    
    # 读取实际的 PGID
    ACTUAL_PGID=""
    if [ -f "$PGID_FILE" ]; then
        ACTUAL_PGID=$(cat "$PGID_FILE" 2>/dev/null)
    fi
    
    # 检查进程状态
    STARTED_OK=false
    RUNNING_PID=""
    
    # 首先检查 PGID 文件中的进程
    if [ -n "$ACTUAL_PGID" ] && kill -0 "$ACTUAL_PGID" 2>/dev/null; then
        STARTED_OK=true
        RUNNING_PID="$ACTUAL_PGID"
    # 然后检查原始 PID
    elif kill -0 $NOHUP_PID 2>/dev/null; then
        STARTED_OK=true
        RUNNING_PID="$NOHUP_PID"
    # 最后检查日志文件
    elif [ -s "$NOHUP_LOG" ]; then
        # 日志有内容，可能进程已经开始工作
        STARTED_OK=true
        RUNNING_PID="$ACTUAL_PGID"
    fi
    
    if [ "$STARTED_OK" = true ]; then
        # 更新 PID 文件为实际运行的 PID
        [ -n "$RUNNING_PID" ] && echo "$RUNNING_PID" > "$PID_FILE"
        
        echo "✅ Grid Search 已在后台启动！"
        echo ""
        echo "=========================================="
        echo "后台运行信息"
        echo "=========================================="
        echo "主进程 PID: ${RUNNING_PID:-$NOHUP_PID}"
        echo "进程组 PGID: ${ACTUAL_PGID:-未知}"
        echo "PID 文件: $PID_FILE"
        echo "日志文件: $NOHUP_LOG"
        echo ""
        echo "📋 常用命令:"
        echo ""
        echo "  # 查看实时日志"
        echo "  tail -f $NOHUP_LOG"
        echo ""
        echo "  # 查看进度（最后30行）"
        echo "  tail -30 $NOHUP_LOG"
        echo ""
        echo "  # 检查进程状态"
        echo "  ps -p ${RUNNING_PID:-$NOHUP_PID} -o pid,pgid,stat,cmd"
        echo ""
        echo "  # ⚠️ 停止 Grid Search 及所有子进程（推荐）"
        if [ -n "$ACTUAL_PGID" ]; then
            echo "  kill -- -$ACTUAL_PGID"
        else
            echo "  kill -- -\$(cat $PGID_FILE)"
        fi
        echo ""
        echo "  # 或者使用停止脚本"
        echo "  bash stop_grid_search.sh $NOHUP_LOG_DIR"
        echo ""
        echo "=========================================="
        echo "💡 提示: 现在可以安全地断开 SSH 连接"
        echo "         Grid Search 将在后台继续运行"
        echo "         杀死主进程时会自动杀死所有训练子进程"
        echo "=========================================="
    else
        echo "❌ 后台启动失败，请检查日志: $NOHUP_LOG"
        exit 1
    fi
    
    exit 0
fi

# ============================================================
# 参数验证
# ============================================================
echo ""
echo "=========================================="
echo "Grid Search 超参数搜索"
echo "=========================================="

if [ "$PAIRED_MODE" = true ]; then
    # 成对参数模式验证
    if [ -z "$PARAM1" ] || [ -z "$PARAM2" ]; then
        echo "❌ 错误：成对参数模式需要指定 --param1 和 --param2"
        exit 1
    fi
    if [ -z "$VALUES1" ] || [ -z "$VALUES2" ]; then
        echo "❌ 错误：成对参数模式需要指定 --values1 和 --values2"
        exit 1
    fi
    
    # 转换为数组
    read -ra ARR1 <<< "$VALUES1"
    read -ra ARR2 <<< "$VALUES2"
    
    if [ ${#ARR1[@]} -ne ${#ARR2[@]} ]; then
        echo "❌ 错误：values1 和 values2 的元素个数不一致"
        echo "  values1 (${#ARR1[@]}个): ${VALUES1}"
        echo "  values2 (${#ARR2[@]}个): ${VALUES2}"
        exit 1
    fi
    
    NUM_EXPERIMENTS=${#ARR1[@]}
    echo "模式: 成对参数搜索"
    echo "参数对: $PARAM1 + $PARAM2"
    echo "实验组数: $NUM_EXPERIMENTS"
    echo "  $PARAM1: ${VALUES1}"
    echo "  $PARAM2: ${VALUES2}"
else
    # 单参数模式验证
    if [ -z "$PARAM" ]; then
        echo "❌ 错误：单参数模式需要指定 --param"
        echo "使用 --help 查看帮助信息"
        exit 1
    fi
    if [ -z "$VALUES" ]; then
        echo "❌ 错误：单参数模式需要指定 --values"
        exit 1
    fi
    
    read -ra ARR_VALUES <<< "$VALUES"
    NUM_EXPERIMENTS=${#ARR_VALUES[@]}
    echo "模式: 单参数搜索"
    echo "参数: $PARAM"
    echo "实验组数: $NUM_EXPERIMENTS"
    echo "参数值: ${VALUES}"
fi

echo "监控指标: $METRIC"
[ -n "$DATASET" ] && echo "数据集: $DATASET" || echo "数据集: (使用默认)"
echo "GPU: $GPU_ID"
echo "=========================================="
echo ""

# ============================================================
# 创建Grid Search输出目录
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$PAIRED_MODE" = true ]; then
    SEARCH_NAME="${PARAM1}_${PARAM2}"
else
    SEARCH_NAME="${PARAM}"
fi

# 获取数据集名称（用于目录）
if [ -n "$DATASET" ]; then
    DS_NAME="$DATASET"
else
    # 从train_unified.sh中读取默认数据集
    DS_NAME=$(grep '^DATASET=' train_unified.sh | head -1 | cut -d'"' -f2)
    [ -z "$DS_NAME" ] && DS_NAME="unknown"
fi

# 如果传入了 OUTPUT_DIR（nohup 模式），使用它；否则创建新目录
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    GRID_SEARCH_DIR="$OUTPUT_DIR"
    echo "使用已创建的输出目录: $GRID_SEARCH_DIR"
else
    GRID_SEARCH_DIR="./logs/${DS_NAME}/grid_search_${SEARCH_NAME}_${TIMESTAMP}"
    mkdir -p "$GRID_SEARCH_DIR"
fi

# 如果是 nohup internal 模式，记录 PID 文件路径供 cleanup 函数使用
if [ "$NOHUP_INTERNAL" = true ]; then
    PID_FILE="${GRID_SEARCH_DIR}/grid_search.pid"
    PGID_FILE="${GRID_SEARCH_DIR}/grid_search.pgid"
    # 写入当前进程的 PID 和 PGID（可能会被启动脚本覆盖，但作为备份）
    echo $$ > "$PID_FILE"
    echo "$(ps -o pgid= -p $$ | tr -d ' ')" > "$PGID_FILE"
fi

# 结果汇总文件
SUMMARY_FILE="${GRID_SEARCH_DIR}/summary.txt"
RESULTS_CSV="${GRID_SEARCH_DIR}/results.csv"

# 初始化汇总文件
echo "========================================" > "$SUMMARY_FILE"
echo "Grid Search 超参数搜索报告" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "搜索时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "数据集: $DS_NAME" >> "$SUMMARY_FILE"
echo "GPU: $GPU_ID" >> "$SUMMARY_FILE"
echo "监控指标: $METRIC" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ "$PAIRED_MODE" = true ]; then
    echo "参数对: $PARAM1 + $PARAM2" >> "$SUMMARY_FILE"
    echo "搜索空间:" >> "$SUMMARY_FILE"
    for i in "${!ARR1[@]}"; do
        echo "  组$((i+1)): $PARAM1=${ARR1[$i]}, $PARAM2=${ARR2[$i]}" >> "$SUMMARY_FILE"
    done
    # CSV头
    echo "experiment_id,${PARAM1},${PARAM2},best_epoch,best_acc,best_f1,best_loss,combined_score" > "$RESULTS_CSV"
else
    echo "参数: $PARAM" >> "$SUMMARY_FILE"
    echo "搜索空间: ${VALUES}" >> "$SUMMARY_FILE"
    # CSV头
    echo "experiment_id,${PARAM},best_epoch,best_acc,best_f1,best_loss,combined_score" > "$RESULTS_CSV"
fi
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "实验详情" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ============================================================
# 参数名称映射（脚本变量 -> train_unified.sh变量）
# ============================================================
get_param_var_name() {
    local param_name="$1"
    case "$param_name" in
        dropout)          echo "DROPOUT" ;;
        batch_size)       echo "BATCH_SIZE" ;;
        learning_rate)    echo "LEARNING_RATE" ;;
        num_hg_layers)    echo "NUM_HG_LAYERS" ;;
        n_segments)       echo "N_SEGMENTS" ;;
        frame_ratio)      echo "FRAME_RATIO" ;;
        num_film_experts) echo "NUM_FILM_EXPERTS" ;;
        film_top_k)       echo "FILM_TOP_K" ;;
        *)
            echo "❌ 未知参数: $param_name" >&2
            exit 1
            ;;
    esac
}

# ============================================================
# 解析测试集指标的函数
# ============================================================
parse_test_metrics() {
    local metrics_file="$1"
    local output_file="$2"
    
    # 使用Python解析指标文件
    # 支持两种格式：
    # 1. MELD分类任务：Acc, F1_weighted, loss
    # 2. CH-SIMS/CH-SIMSv2回归任务：MAE, Corr, Acc_2, F1_2, loss
    python3 << 'PYTHON_EOF'
import re
import sys
import os

metrics_file = os.environ.get('METRICS_FILE', '')
output_file = os.environ.get('OUTPUT_FILE', '')

if not metrics_file or not output_file:
    print("ERROR: 缺少参数")
    sys.exit(1)

# 读取文件
try:
    with open(metrics_file, 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    print(f"ERROR: 无法读取文件 {metrics_file}: {e}")
    sys.exit(1)

# 检测数据集类型（通过查找特征性指标）
is_regression = 'MAE:' in content or 'Acc_2:' in content
print(f"检测到数据类型: {'回归任务(CH-SIMS)' if is_regression else '分类任务(MELD)'}")

results = []

# 解析测试集指标块
# 格式示例：
# ============================================================
# Epoch 1/50
# ============================================================
# ...
# 测试集 (Test):
#   MAE: 0.1234
#   Acc_2: 0.7500
#   F1_2: 0.7200
#   loss: 0.0500

lines = content.split('\n')
current_epoch = 0
in_test_section = False
test_data = {}

for line in lines:
    # 检测Epoch（匹配 "Epoch N/M" 格式）
    epoch_match = re.match(r'^Epoch (\d+)/\d+', line.strip())
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        continue
    
    # 检测测试集section开始
    if '测试集 (Test):' in line or re.match(r'^\s*Test:', line):
        in_test_section = True
        test_data = {}
        continue
    
    # 在测试集section中收集数据
    if in_test_section:
        # 去除前导空格
        stripped = line.strip()
        
        if is_regression:
            # 回归任务指标：MAE, Acc_2, F1_2
            if stripped.startswith('MAE:'):
                match = re.search(r'MAE:\s*([\d\.]+)', stripped)
                if match:
                    test_data['mae'] = float(match.group(1))
            elif stripped.startswith('Acc_2:'):
                match = re.search(r'Acc_2:\s*([\d\.]+)', stripped)
                if match:
                    test_data['acc'] = float(match.group(1))
            elif stripped.startswith('F1_2:'):
                match = re.search(r'F1_2:\s*([\d\.]+)', stripped)
                if match:
                    test_data['f1'] = float(match.group(1))
            elif stripped.startswith('Corr:'):
                match = re.search(r'Corr:\s*([\d\.\-]+)', stripped)
                if match:
                    test_data['corr'] = float(match.group(1))
        else:
            # 分类任务指标：Acc, F1_weighted
            if stripped.startswith('Acc:') and 'Acc_' not in stripped:
                match = re.search(r'Acc:\s*([\d\.]+)', stripped)
                if match:
                    test_data['acc'] = float(match.group(1))
            elif stripped.startswith('F1_weighted:'):
                match = re.search(r'F1_weighted:\s*([\d\.]+)', stripped)
                if match:
                    test_data['f1'] = float(match.group(1))
        
        # loss是通用的
        if stripped.startswith('loss:'):
            match = re.search(r'loss:\s*([\d\.]+)', stripped)
            if match:
                test_data['loss'] = float(match.group(1))
                
                # loss通常是最后一个指标，保存结果
                if 'acc' in test_data and current_epoch > 0:
                    # 如果没有f1，使用acc作为默认值
                    if 'f1' not in test_data:
                        test_data['f1'] = test_data['acc']
                    
                    results.append({
                        'epoch': current_epoch,
                        'acc': test_data['acc'],
                        'f1': test_data['f1'],
                        'loss': test_data['loss'],
                        'mae': test_data.get('mae', 0),
                        'corr': test_data.get('corr', 0)
                    })
                test_data = {}
                in_test_section = False
        
        # 如果遇到空行或新的section，结束当前test section
        elif stripped == '' or stripped.startswith('===') or stripped.startswith('---'):
            if stripped.startswith('===') or stripped.startswith('---'):
                in_test_section = False

# 去重（按epoch，保留最后一个）
seen_epochs = {}
for r in results:
    seen_epochs[r['epoch']] = r

results = sorted(seen_epochs.values(), key=lambda x: x['epoch'])

# 写入结果（增加mae和corr列以支持回归任务）
try:
    with open(output_file, 'w') as f:
        for r in results:
            f.write(f"{r['epoch']},{r['acc']},{r['f1']},{r['loss']},{r.get('mae', 0)},{r.get('corr', 0)}\n")
    print(f"解析到 {len(results)} 个epoch的测试集结果")
except Exception as e:
    print(f"ERROR: 无法写入文件 {output_file}: {e}")
    sys.exit(1)
PYTHON_EOF
}

# ============================================================
# 计算最优epoch和综合分数
# ============================================================
find_best_epoch() {
    local parsed_file="$1"
    local metric_type="$2"
    
    PARSED_FILE="$parsed_file" METRIC_TYPE="$metric_type" python3 << 'PYTHON_EOF'
import sys
import os

parsed_file = os.environ.get('PARSED_FILE', '')
metric = os.environ.get('METRIC_TYPE', 'combined')

if not parsed_file:
    print("ERROR,0,0,0,0")
    sys.exit(1)

results = []
try:
    with open(parsed_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 支持新格式（6列）和旧格式（4列）
            if len(parts) >= 4:
                try:
                    epoch = int(parts[0])
                    acc = float(parts[1])
                    f1 = float(parts[2])
                    loss = float(parts[3])
                    mae = float(parts[4]) if len(parts) > 4 else 0
                    corr = float(parts[5]) if len(parts) > 5 else 0
                    
                    # 计算综合分数
                    # 对于回归任务（有MAE），使用MAE和Corr
                    # 对于分类任务，使用acc和f1
                    if mae > 0:  # 回归任务
                        # MAE越小越好（通常在0-1范围），Corr越大越好（-1到1）
                        # acc_2和f1_2越大越好
                        mae_normalized = max(0, 1 - mae)  # 归一化MAE
                        corr_normalized = (corr + 1) / 2   # 归一化Corr到0-1
                        combined = acc * 0.25 + f1 * 0.25 + mae_normalized * 0.25 + corr_normalized * 0.25
                    else:  # 分类任务
                        # acc和f1越大越好（范围0-1），loss越小越好
                        loss_normalized = max(0, 1 - loss / 3)  # 假设loss一般在0-3范围
                        combined = acc * 0.35 + f1 * 0.35 + loss_normalized * 0.3
                    
                    results.append({
                        'epoch': epoch,
                        'acc': acc,
                        'f1': f1,
                        'loss': loss,
                        'mae': mae,
                        'corr': corr,
                        'combined': combined
                    })
                except (ValueError, IndexError):
                    continue
except Exception as e:
    print(f"ERROR,0,0,0,0")
    sys.exit(1)

if not results:
    print("ERROR,0,0,0,0")
    sys.exit(1)

# 根据指标类型选择最优
if metric == 'acc':
    best = max(results, key=lambda x: x['acc'])
elif metric == 'f1':
    best = max(results, key=lambda x: x['f1'])
elif metric == 'mae':
    # 对于MAE，如果有mae值就用mae，否则用loss
    if any(r['mae'] > 0 for r in results):
        best = min(results, key=lambda x: x['mae'])
    else:
        best = min(results, key=lambda x: x['loss'])
elif metric == 'corr':
    best = max(results, key=lambda x: x['corr'])
else:  # combined
    best = max(results, key=lambda x: x['combined'])

# 输出: epoch,acc,f1,loss,combined
print(f"{best['epoch']},{best['acc']:.4f},{best['f1']:.4f},{best['loss']:.4f},{best['combined']:.4f}")
PYTHON_EOF
}

# ============================================================
# 修改train_unified.sh参数并运行训练
# ============================================================
run_experiment() {
    local exp_id="$1"
    local param_settings="$2"  # 格式: "PARAM1=VALUE1 PARAM2=VALUE2 ..."
    
    echo ""
    echo "=========================================="
    echo "实验 $exp_id / $NUM_EXPERIMENTS"
    echo "参数设置: $param_settings"
    echo "=========================================="
    
    # 保存当前工作目录
    local ORIGINAL_DIR=$(pwd)
    
    # 创建实验专用目录
    local exp_dir="${GRID_SEARCH_DIR}/exp_${exp_id}"
    mkdir -p "$exp_dir"
    
    # 复制train_unified.sh到实验目录并修改参数
    local exp_script="${exp_dir}/train_unified.sh"
    cp train_unified.sh "$exp_script"
    
    # 修改参数
    for setting in $param_settings; do
        local var_name="${setting%%=*}"
        local var_value="${setting#*=}"
        
        # 使用sed替换变量值
        # 匹配模式: VAR_NAME="value" 或 VAR_NAME=value
        sed -i "s/^${var_name}=.*$/${var_name}=\"${var_value}\"/" "$exp_script"
    done
    
    # 如果指定了数据集，修改DATASET
    if [ -n "$DATASET" ]; then
        sed -i "s/^DATASET=.*$/DATASET=\"${DATASET}\"/" "$exp_script"
    fi
    
    # 修改GPU_ID
    sed -i "s/^GPU_ID=.*$/GPU_ID=${GPU_ID}/" "$exp_script"
    
    # 强制使用前台运行（便于脚本控制流程）
    sed -i "s/^USE_NOHUP=.*$/USE_NOHUP=false/" "$exp_script"
    
    # 强制启用测试集评估
    sed -i "s/^EVAL_TEST_EVERY_EPOCH=.*$/EVAL_TEST_EVERY_EPOCH=true/" "$exp_script"
    
    # 获取GPU后缀
    local gpu_suffix=$(echo $GPU_ID | tr ',' '_')
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] 将执行: bash $exp_script"
        echo "[DRY RUN] 参数设置:"
        for setting in $param_settings; do
            echo "  $setting"
        done
        
        # 创建模拟结果
        echo "1,0.6500,0.5800,1.2000,0.4500" > "${exp_dir}/best_result.txt"
        return 0
    fi
    
    # 记录开始时间
    local start_time=$(date +%s)
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 运行训练
    echo ""
    echo "正在运行训练..."
    
    # 捕获训练脚本的输出
    local train_log="${exp_dir}/train_output.log"
    
    # 进入实验目录执行（这样train_unified.sh能找到项目文件）
    cd "$ORIGINAL_DIR"
    
    # 执行训练脚本
    local train_exit_code=0
    
    # 获取当前 conda 环境
    local CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
    local CONDA_PREFIX_VAL="${CONDA_PREFIX:-}"
    
    # 构建启动命令
    # 关键：禁用 job control 以防止子进程创建新的进程组
    # 这样所有子进程都会继承主进程的 PGID，便于统一管理
    set +m  # 禁用 job control
    
    if [ -n "$CONDA_ENV" ] && [ "$CONDA_ENV" != "base" ]; then
        echo "使用 conda 环境: $CONDA_ENV"
        bash "$exp_script" > "$train_log" 2>&1 &
        CHILD_PID=$!
    else
        bash "$exp_script" > "$train_log" 2>&1 &
        CHILD_PID=$!
    fi
    
    set -m  # 重新启用 job control
    
    CHILD_PIDS+=($CHILD_PID)
    echo "训练进程已启动 (PID: $CHILD_PID, PGID: $(ps -o pgid= -p $CHILD_PID 2>/dev/null | tr -d ' '))"
    
    # 等待训练完成
    wait $CHILD_PID 2>/dev/null || train_exit_code=$?
    
    # 清空当前子进程记录
    CHILD_PID=""
    
    if [ $train_exit_code -eq 0 ]; then
        echo "✅ 训练完成"
    elif [ $train_exit_code -eq 130 ] || [ $train_exit_code -eq 143 ]; then
        echo "⚠️ 训练被中断 (exit code: $train_exit_code)"
        return 1
    else
        echo "⚠️ 训练可能出错 (exit code: $train_exit_code)，请检查日志: $train_log"
    fi
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "耗时: $((duration / 60))分$((duration % 60))秒"
    
    # 查找生成的指标文件
    # 首先从train_output.log中提取指标文件路径
    local metrics_file=""
    
    # 方法1: 从日志中提取
    if [ -f "$train_log" ]; then
        metrics_file=$(grep -o '\./logs/[^[:space:]]*\.txt' "$train_log" 2>/dev/null | tail -1)
        # 转换为绝对路径
        if [ -n "$metrics_file" ]; then
            metrics_file="${ORIGINAL_DIR}/${metrics_file#./}"
        fi
    fi
    
    # 方法2: 在标准位置查找最新的文件
    if [ -z "$metrics_file" ] || [ ! -f "$metrics_file" ]; then
        metrics_file=$(ls -t "${ORIGINAL_DIR}/logs/${DS_NAME}/gpu_${gpu_suffix}"/train_*.txt 2>/dev/null | head -1)
    fi
    
    if [ -n "$metrics_file" ] && [ -f "$metrics_file" ]; then
        echo "找到指标文件: $metrics_file"
        
        # 复制到实验目录
        cp "$metrics_file" "${exp_dir}/metrics.txt"
        
        # 解析指标
        local parsed_file="${exp_dir}/parsed_metrics.csv"
        export METRICS_FILE="${exp_dir}/metrics.txt"
        export OUTPUT_FILE="$parsed_file"
        parse_test_metrics
        unset METRICS_FILE OUTPUT_FILE
        
        # 检查解析结果
        if [ -f "$parsed_file" ] && [ -s "$parsed_file" ]; then
            # 找到最优epoch
            local best_result=$(find_best_epoch "$parsed_file" "$METRIC")
            echo "最优结果: $best_result"
            
            # 保存到实验结果文件
            echo "$best_result" > "${exp_dir}/best_result.txt"
            
            return 0
        else
            echo "❌ 指标解析失败"
            echo "ERROR,0,0,0,0" > "${exp_dir}/best_result.txt"
            return 1
        fi
    else
        echo "❌ 未找到指标文件"
        echo "尝试查找路径: ${ORIGINAL_DIR}/logs/${DS_NAME}/gpu_${gpu_suffix}/train_*.txt"
        echo "ERROR,0,0,0,0" > "${exp_dir}/best_result.txt"
        return 1
    fi
}

# ============================================================
# 浮点数比较辅助函数
# ============================================================
float_gt() {
    # 返回0如果 $1 > $2，否则返回1
    python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"
}

# ============================================================
# 主循环：执行Grid Search
# ============================================================
echo ""
echo "=========================================="
echo "开始 Grid Search"
echo "=========================================="

# 全局最优结果变量
BEST_SCORE="-999999"
BEST_EXP_ID="0"
BEST_PARAMS=""
BEST_EPOCH=""
BEST_ACC=""
BEST_F1=""
BEST_LOSS=""

if [ "$PAIRED_MODE" = true ]; then
    # 成对参数模式
    VAR1=$(get_param_var_name "$PARAM1")
    VAR2=$(get_param_var_name "$PARAM2")
    
    for i in "${!ARR1[@]}"; do
        exp_id=$((i + 1))
        val1="${ARR1[$i]}"
        val2="${ARR2[$i]}"
        
        param_settings="${VAR1}=${val1} ${VAR2}=${val2}"
        
        echo "" >> "$SUMMARY_FILE"
        echo "--- 实验 $exp_id ---" >> "$SUMMARY_FILE"
        echo "$PARAM1 = $val1" >> "$SUMMARY_FILE"
        echo "$PARAM2 = $val2" >> "$SUMMARY_FILE"
        
        if run_experiment "$exp_id" "$param_settings"; then
            # 读取结果
            result_file="${GRID_SEARCH_DIR}/exp_${exp_id}/best_result.txt"
            if [ -f "$result_file" ]; then
                result=$(cat "$result_file")
                IFS=',' read -r best_epoch best_acc best_f1 best_loss combined_score <<< "$result"
                
                # 检查是否是有效结果
                if [ "$best_epoch" != "ERROR" ]; then
                    echo "最优Epoch: $best_epoch" >> "$SUMMARY_FILE"
                    echo "Accuracy: $best_acc" >> "$SUMMARY_FILE"
                    echo "F1 Score: $best_f1" >> "$SUMMARY_FILE"
                    echo "Loss: $best_loss" >> "$SUMMARY_FILE"
                    echo "综合得分: $combined_score" >> "$SUMMARY_FILE"
                    
                    # 写入CSV
                    echo "${exp_id},${val1},${val2},${best_epoch},${best_acc},${best_f1},${best_loss},${combined_score}" >> "$RESULTS_CSV"
                    
                    # 检查是否是全局最优
                    if float_gt "$combined_score" "$BEST_SCORE"; then
                        BEST_SCORE="$combined_score"
                        BEST_EXP_ID="$exp_id"
                        BEST_PARAMS="$PARAM1=$val1, $PARAM2=$val2"
                        BEST_EPOCH="$best_epoch"
                        BEST_ACC="$best_acc"
                        BEST_F1="$best_f1"
                        BEST_LOSS="$best_loss"
                    fi
                else
                    echo "状态: 解析失败" >> "$SUMMARY_FILE"
                    echo "${exp_id},${val1},${val2},ERROR,0,0,0,0" >> "$RESULTS_CSV"
                fi
            fi
        else
            echo "状态: 训练失败" >> "$SUMMARY_FILE"
            echo "${exp_id},${val1},${val2},ERROR,0,0,0,0" >> "$RESULTS_CSV"
        fi
        
        echo "" >> "$SUMMARY_FILE"
    done
else
    # 单参数模式
    VAR_NAME=$(get_param_var_name "$PARAM")
    
    for i in "${!ARR_VALUES[@]}"; do
        exp_id=$((i + 1))
        val="${ARR_VALUES[$i]}"
        
        param_settings="${VAR_NAME}=${val}"
        
        echo "" >> "$SUMMARY_FILE"
        echo "--- 实验 $exp_id ---" >> "$SUMMARY_FILE"
        echo "$PARAM = $val" >> "$SUMMARY_FILE"
        
        if run_experiment "$exp_id" "$param_settings"; then
            # 读取结果
            result_file="${GRID_SEARCH_DIR}/exp_${exp_id}/best_result.txt"
            if [ -f "$result_file" ]; then
                result=$(cat "$result_file")
                IFS=',' read -r best_epoch best_acc best_f1 best_loss combined_score <<< "$result"
                
                # 检查是否是有效结果
                if [ "$best_epoch" != "ERROR" ]; then
                    echo "最优Epoch: $best_epoch" >> "$SUMMARY_FILE"
                    echo "Accuracy: $best_acc" >> "$SUMMARY_FILE"
                    echo "F1 Score: $best_f1" >> "$SUMMARY_FILE"
                    echo "Loss: $best_loss" >> "$SUMMARY_FILE"
                    echo "综合得分: $combined_score" >> "$SUMMARY_FILE"
                    
                    # 写入CSV
                    echo "${exp_id},${val},${best_epoch},${best_acc},${best_f1},${best_loss},${combined_score}" >> "$RESULTS_CSV"
                    
                    # 检查是否是全局最优
                    if float_gt "$combined_score" "$BEST_SCORE"; then
                        BEST_SCORE="$combined_score"
                        BEST_EXP_ID="$exp_id"
                        BEST_PARAMS="$PARAM=$val"
                        BEST_EPOCH="$best_epoch"
                        BEST_ACC="$best_acc"
                        BEST_F1="$best_f1"
                        BEST_LOSS="$best_loss"
                    fi
                else
                    echo "状态: 解析失败" >> "$SUMMARY_FILE"
                    echo "${exp_id},${val},ERROR,0,0,0,0" >> "$RESULTS_CSV"
                fi
            fi
        else
            echo "状态: 训练失败" >> "$SUMMARY_FILE"
            echo "${exp_id},${val},ERROR,0,0,0,0" >> "$RESULTS_CSV"
        fi
        
        echo "" >> "$SUMMARY_FILE"
    done
fi

# ============================================================
# 生成最终报告
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "🏆 最优参数配置" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ "$BEST_EXP_ID" != "0" ]; then
    echo "实验编号: $BEST_EXP_ID" >> "$SUMMARY_FILE"
    echo "参数设置: $BEST_PARAMS" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "最优指标 (在Epoch $BEST_EPOCH):" >> "$SUMMARY_FILE"
    echo "  Accuracy:     $BEST_ACC" >> "$SUMMARY_FILE"
    echo "  F1 Score:     $BEST_F1" >> "$SUMMARY_FILE"
    echo "  Loss:         $BEST_LOSS" >> "$SUMMARY_FILE"
    echo "  综合得分:     $BEST_SCORE" >> "$SUMMARY_FILE"
else
    echo "没有成功完成的实验" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "搜索完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "结果目录: $GRID_SEARCH_DIR" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

# ============================================================
# 控制台输出最终结果
# ============================================================
echo ""
echo "=========================================="
echo "✅ Grid Search 完成!"
echo "=========================================="
echo ""
echo "📊 结果汇总:"
echo "  总实验数: $NUM_EXPERIMENTS"
echo ""

if [ "$BEST_EXP_ID" != "0" ]; then
    echo "🏆 最优参数配置:"
    echo "  $BEST_PARAMS"
    echo ""
    echo "📈 最优指标 (Epoch $BEST_EPOCH):"
    echo "  Accuracy:  $BEST_ACC"
    echo "  F1 Score:  $BEST_F1"
    echo "  Loss:      $BEST_LOSS"
    echo "  综合得分:  $BEST_SCORE"
else
    echo "❌ 没有成功完成的实验"
fi

echo ""
echo "📁 输出文件:"
echo "  汇总报告: $SUMMARY_FILE"
echo "  CSV结果:  $RESULTS_CSV"
echo ""
echo "查看报告: cat $SUMMARY_FILE"
echo "=========================================="

# 清理 PID/PGID 文件（如果是 nohup 模式运行的）
if [ "$NOHUP_INTERNAL" = true ]; then
    [ -f "${GRID_SEARCH_DIR}/grid_search.pid" ] && rm -f "${GRID_SEARCH_DIR}/grid_search.pid"
    [ -f "${GRID_SEARCH_DIR}/grid_search.pgid" ] && rm -f "${GRID_SEARCH_DIR}/grid_search.pgid"
fi

# 标记正常完成，防止信号处理误触发
CLEANUP_DONE=true

