#!/bin/bash

# ========================================
# 停止 Grid Search 及其所有子进程
# ========================================
# 使用方法:
#   bash stop_grid_search.sh                    # 停止所有 grid_search 进程
#   bash stop_grid_search.sh <PID>              # 停止指定 PID 的进程
#   bash stop_grid_search.sh <LOG_DIR>          # 停止指定日志目录对应的进程
#   bash stop_grid_search.sh --force            # 强制停止（使用 SIGKILL）
#   bash stop_grid_search.sh --list             # 仅列出进程，不停止
#
# 示例:
#   bash stop_grid_search.sh ./logs/chsimsv2/grid_search_dropout_20251222_123456
#   bash stop_grid_search.sh 12345
#   bash stop_grid_search.sh --force
# ========================================

FORCE=false
LIST_ONLY=false
TARGET_PID=""
TARGET_DIR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        [0-9]*)
            TARGET_PID="$1"
            shift
            ;;
        */logs/*|./logs/*)
            TARGET_DIR="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: bash stop_grid_search.sh [OPTIONS] [PID|LOG_DIR]"
            echo ""
            echo "Options:"
            echo "  PID         指定要停止的进程ID"
            echo "  LOG_DIR     Grid Search 日志目录（包含 grid_search.pid 文件）"
            echo "  --force     强制停止（使用 SIGKILL）"
            echo "  --list      仅列出进程，不停止"
            echo ""
            echo "示例:"
            echo "  # 停止所有 grid_search 进程"
            echo "  bash stop_grid_search.sh"
            echo ""
            echo "  # 停止指定日志目录对应的进程（推荐）"
            echo "  bash stop_grid_search.sh ./logs/chsimsv2/grid_search_dropout_20251222_123456"
            echo ""
            echo "  # 强制停止指定 PID"
            echo "  bash stop_grid_search.sh 12345 --force"
            exit 0
            ;;
        *)
            # 检查是否是目录
            if [ -d "$1" ]; then
                TARGET_DIR="$1"
            else
                echo "未知参数: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

echo ""
echo "=========================================="
echo "停止 Grid Search"
echo "=========================================="

# 当前脚本的PID（排除自己）
SELF_PID=$$
SELF_PPID=$PPID

# 从目录中读取 PID 和 PGID
read_pids_from_dir() {
    local dir="$1"
    local pid=""
    local pgid=""
    
    if [ -f "$dir/grid_search.pid" ]; then
        pid=$(cat "$dir/grid_search.pid" 2>/dev/null)
    fi
    if [ -f "$dir/grid_search.pgid" ]; then
        pgid=$(cat "$dir/grid_search.pgid" 2>/dev/null)
    fi
    
    echo "$pid $pgid"
}

# 查找所有相关进程（排除自己和父进程）
find_processes() {
    local pids=""
    
    # 1. 查找 grid_search.sh 进程（排除 stop_grid_search.sh）
    local gs_pids=$(pgrep -f "grid_search\.sh" 2>/dev/null || true)
    for pid in $gs_pids; do
        # 检查是不是 stop_grid_search.sh
        local cmd=$(ps -p $pid -o args= 2>/dev/null || true)
        if [[ "$cmd" != *"stop_grid_search"* ]] && [ -n "$cmd" ]; then
            pids="$pids $pid"
        fi
    done
    
    # 2. 查找 train_unified.sh 进程
    local tu_pids=$(pgrep -f "train_unified\.sh" 2>/dev/null || true)
    [ -n "$tu_pids" ] && pids="$pids $tu_pids"
    
    # 3. 查找 train_refactored.py 进程
    local tr_pids=$(pgrep -f "train_refactored\.py" 2>/dev/null || true)
    [ -n "$tr_pids" ] && pids="$pids $tr_pids"
    
    # 过滤掉自己、父进程、和无效PID
    local valid_pids=""
    for pid in $pids; do
        # 排除自己和父进程
        [ "$pid" = "$SELF_PID" ] && continue
        [ "$pid" = "$SELF_PPID" ] && continue
        # 检查进程是否存在
        if kill -0 "$pid" 2>/dev/null; then
            valid_pids="$valid_pids $pid"
        fi
    done
    
    echo "$valid_pids" | tr ' ' '\n' | sort -u | grep -v '^$' || true
}

# 停止进程及其子进程
stop_process() {
    local pid=$1
    local signal=$2
    
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  进程 $pid 已不存在"
        return 0
    fi
    
    echo "  停止进程 $pid..."
    
    # 获取所有子进程（递归）
    get_descendants() {
        local parent=$1
        local children=$(pgrep -P "$parent" 2>/dev/null || true)
        echo "$children"
        for child in $children; do
            get_descendants "$child"
        done
    }
    
    local all_children=$(get_descendants "$pid" | sort -u)
    
    # 先停止子进程
    for child in $all_children; do
        if kill -0 "$child" 2>/dev/null; then
            echo "    停止子进程 $child"
            kill $signal "$child" 2>/dev/null || true
        fi
    done
    
    # 再停止主进程
    kill $signal "$pid" 2>/dev/null || true
    
    return 0
}

# 停止整个进程组
stop_process_group() {
    local pgid=$1
    local signal=$2
    
    echo "  停止进程组 PGID=$pgid..."
    
    # 发送信号给整个进程组
    kill $signal -"$pgid" 2>/dev/null || true
    
    return 0
}

# 主逻辑

# 如果指定了目录，从目录中读取 PID/PGID
if [ -n "$TARGET_DIR" ]; then
    echo "从目录读取进程信息: $TARGET_DIR"
    
    if [ ! -d "$TARGET_DIR" ]; then
        echo "❌ 目录不存在: $TARGET_DIR"
        exit 1
    fi
    
    read -r TARGET_PID TARGET_PGID <<< $(read_pids_from_dir "$TARGET_DIR")
    
    if [ -z "$TARGET_PID" ] && [ -z "$TARGET_PGID" ]; then
        echo "❌ 未找到 PID 或 PGID 文件"
        exit 1
    fi
    
    echo "  PID: ${TARGET_PID:-未知}"
    echo "  PGID: ${TARGET_PGID:-未知}"
    echo ""
    
    # 优先使用 PGID 停止整个进程组
    if [ -n "$TARGET_PGID" ]; then
        if [ "$LIST_ONLY" = true ]; then
            echo "进程组 $TARGET_PGID 中的进程:"
            ps -o pid,pgid,stat,cmd --no-headers -g "$TARGET_PGID" 2>/dev/null || echo "  (无法获取进程列表)"
        else
            echo "停止进程组 $TARGET_PGID..."
            if [ "$FORCE" = true ]; then
                stop_process_group "$TARGET_PGID" "-9"
            else
                stop_process_group "$TARGET_PGID" "-TERM"
                sleep 3
                # 检查是否还有进程在运行
                if ps -g "$TARGET_PGID" --no-headers 2>/dev/null | grep -q .; then
                    echo "部分进程未响应 SIGTERM，使用 SIGKILL..."
                    stop_process_group "$TARGET_PGID" "-9"
                fi
            fi
        fi
    elif [ -n "$TARGET_PID" ]; then
        if ! kill -0 "$TARGET_PID" 2>/dev/null; then
            echo "进程 $TARGET_PID 不存在"
            exit 1
        fi
        
        if [ "$LIST_ONLY" = true ]; then
            echo "进程 $TARGET_PID 及其子进程:"
            pstree -p "$TARGET_PID" 2>/dev/null || ps -p "$TARGET_PID" -o pid,ppid,pgid,cmd
        else
            echo "停止进程 $TARGET_PID 及其子进程..."
            if [ "$FORCE" = true ]; then
                stop_process "$TARGET_PID" "-9"
            else
                stop_process "$TARGET_PID" "-TERM"
                sleep 3
                if kill -0 "$TARGET_PID" 2>/dev/null; then
                    echo "进程未响应 SIGTERM，使用 SIGKILL..."
                    stop_process "$TARGET_PID" "-9"
                fi
            fi
        fi
    fi
    
elif [ -n "$TARGET_PID" ]; then
    # 停止指定 PID 的进程
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        echo "进程 $TARGET_PID 不存在"
        exit 1
    fi
    
    if [ "$LIST_ONLY" = true ]; then
        echo "进程 $TARGET_PID 信息:"
        ps -p "$TARGET_PID" -o pid,ppid,pgid,stat,cmd
        echo ""
        echo "子进程树:"
        pstree -p "$TARGET_PID" 2>/dev/null || echo "  (无法获取进程树)"
    else
        echo "停止进程 $TARGET_PID 及其子进程..."
        
        if [ "$FORCE" = true ]; then
            stop_process "$TARGET_PID" "-9"
        else
            stop_process "$TARGET_PID" "-TERM"
            sleep 3
            if kill -0 "$TARGET_PID" 2>/dev/null; then
                echo "进程未响应 SIGTERM，使用 SIGKILL..."
                stop_process "$TARGET_PID" "-9"
            fi
        fi
    fi
else
    # 停止所有相关进程
    all_pids=$(find_processes)
    
    if [ -z "$all_pids" ]; then
        echo "没有找到正在运行的 Grid Search 进程"
        exit 0
    fi
    
    echo "找到以下进程:"
    for pid in $all_pids; do
        cmd=$(ps -p $pid -o pid,ppid,pgid,args= 2>/dev/null || echo "(已结束)")
        echo "  $cmd"
    done
    echo ""
    
    if [ "$LIST_ONLY" = true ]; then
        echo "使用 --force 或不带 --list 选项来停止这些进程"
        exit 0
    fi
    
    echo "正在停止进程..."
    for pid in $all_pids; do
        if [ "$FORCE" = true ]; then
            stop_process "$pid" "-9"
        else
            stop_process "$pid" "-TERM"
        fi
    done
    
    # 等待并检查
    sleep 3
    remaining=$(find_processes)
    
    if [ -n "$remaining" ] && [ "$FORCE" != true ]; then
        echo ""
        echo "部分进程未响应 SIGTERM，使用 SIGKILL 强制停止..."
        for pid in $remaining; do
            stop_process "$pid" "-9"
        done
    fi
fi

# 最终检查（除非只是列出）
if [ "$LIST_ONLY" != true ]; then
    sleep 1
    remaining=$(find_processes)
    if [ -z "$remaining" ]; then
        echo ""
        echo "✅ 所有 Grid Search 进程已停止"
        
        # 清理 PID 文件
        if [ -n "$TARGET_DIR" ]; then
            [ -f "$TARGET_DIR/grid_search.pid" ] && rm -f "$TARGET_DIR/grid_search.pid"
            [ -f "$TARGET_DIR/grid_search.pgid" ] && rm -f "$TARGET_DIR/grid_search.pgid"
        fi
    else
        echo ""
        echo "⚠️ 以下进程仍在运行:"
        for pid in $remaining; do
            cmd=$(ps -p $pid -o args= 2>/dev/null || echo "(未知)")
            echo "  PID $pid: $cmd"
        done
        echo ""
        echo "请尝试: bash stop_grid_search.sh --force"
    fi
fi

echo "=========================================="
