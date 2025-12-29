#!/usr/bin/env python3
"""
IEMOCAP数据集预处理脚本

功能：
1. 将AVI视频转换为MP4格式
2. 整理视频到新目录结构
3. 生成统一的CSV标签文件

使用方法：
    conda activate kopa
    python preprocess_iemocap.py
"""

import os
import re
import csv
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 配置 =====================
IEMOCAP_ROOT = Path("/home/kemove/wsy2/IEMOCAP/IEMOCAP_full_release")
OUTPUT_ROOT = Path("/home/kemove/wsy2/IEMOCAP_processed")
NUM_SESSIONS = 5
NUM_WORKERS = 4  # 并行转换视频的线程数

# ===================== 路径设置 =====================
VIDEO_OUTPUT_DIR = OUTPUT_ROOT / "videos"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audios"  # 可选：也整理音频
META_CSV_PATH = OUTPUT_ROOT / "meta.csv"


def ensure_dirs():
    """创建输出目录"""
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录已创建: {OUTPUT_ROOT}")


def convert_avi_to_mp4(avi_path: Path, mp4_path: Path) -> bool:
    """
    使用ffmpeg将AVI转换为MP4
    
    Args:
        avi_path: 输入AVI文件路径
        mp4_path: 输出MP4文件路径
    
    Returns:
        转换是否成功
    """
    if mp4_path.exists():
        return True  # 跳过已存在的文件
    
    try:
        cmd = [
            "ffmpeg",
            "-i", str(avi_path),
            "-c:v", "libx264",      # 视频编码器
            "-preset", "medium",     # 编码速度/质量平衡
            "-crf", "23",           # 质量因子 (0-51, 越低质量越好)
            "-c:a", "aac",          # 音频编码器
            "-b:a", "128k",         # 音频比特率
            "-y",                   # 覆盖输出文件
            "-loglevel", "error",   # 只显示错误
            str(mp4_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 转换失败: {avi_path}")
        print(f"  错误: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except FileNotFoundError:
        print("✗ 错误: 未找到ffmpeg，请确保已安装ffmpeg")
        return False


def parse_emotion_file(emo_file: Path) -> dict:
    """
    解析情感标注文件
    
    Args:
        emo_file: EmoEvaluation文件路径
    
    Returns:
        字典 {utterance_id: {emotion, valence, activation, dominance, start, end}}
    """
    annotations = {}
    
    if not emo_file.exists():
        return annotations
    
    with open(emo_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 匹配格式: [start - end] utterance_id emotion [V, A, D]
    pattern = r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\w+)\s+\[([^\]]+)\]'
    
    for match in re.finditer(pattern, content):
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        utterance_id = match.group(3)
        emotion = match.group(4)
        vad = match.group(5).split(',')
        
        try:
            valence = float(vad[0].strip())
            activation = float(vad[1].strip())
            dominance = float(vad[2].strip())
        except (ValueError, IndexError):
            valence = activation = dominance = 0.0
        
        annotations[utterance_id] = {
            'emotion': emotion,
            'valence': valence,
            'activation': activation,
            'dominance': dominance,
            'start_time': start_time,
            'end_time': end_time
        }
    
    return annotations


def parse_transcription_file(trans_file: Path) -> dict:
    """
    解析文本转录文件
    
    Args:
        trans_file: transcriptions文件路径
    
    Returns:
        字典 {utterance_id: text}
    """
    transcriptions = {}
    
    if not trans_file.exists():
        return transcriptions
    
    with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 格式: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
            match = re.match(r'(\S+)\s+\[[^\]]+\]:\s*(.*)$', line)
            if match:
                utterance_id = match.group(1)
                text = match.group(2).strip()
                transcriptions[utterance_id] = text
    
    return transcriptions


def get_all_dialogues():
    """
    获取所有对话的信息
    
    Returns:
        列表 [(session_id, dialogue_id, avi_path_f, avi_path_m), ...]
    """
    dialogues = []
    
    for session_num in range(1, NUM_SESSIONS + 1):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        avi_dir = session_dir / "dialog" / "avi" / "DivX"
        
        if not avi_dir.exists():
            print(f"⚠ 警告: 目录不存在 {avi_dir}")
            continue
        
        # 收集所有AVI文件（排除macOS元数据文件）
        avi_files = [f for f in avi_dir.glob("*.avi") if not f.name.startswith('._')]
        
        # 按对话名分组（去掉F/M后缀）
        dialogue_groups = {}
        for avi_file in avi_files:
            name = avi_file.stem
            # Ses01F_impro01 -> 对话标识
            # 判断是F还是M的视频
            if name.startswith(f"Ses0{session_num}F_"):
                base_name = name  # 保持原名作为对话ID
                dialogue_groups.setdefault(base_name, {})['F'] = avi_file
            elif name.startswith(f"Ses0{session_num}M_"):
                # 对应的F版本名字
                base_name = name.replace(f"Ses0{session_num}M_", f"Ses0{session_num}F_")
                dialogue_groups.setdefault(base_name, {})['M'] = avi_file
        
        for dialogue_id, videos in dialogue_groups.items():
            dialogues.append({
                'session': session_num,
                'dialogue_id': dialogue_id,
                'video_f': videos.get('F'),
                'video_m': videos.get('M')
            })
    
    return dialogues


def process_videos():
    """处理所有视频：AVI -> MP4"""
    print("\n" + "=" * 50)
    print("步骤 1: 转换视频格式 (AVI -> MP4)")
    print("=" * 50)
    
    # 收集所有需要转换的视频
    conversion_tasks = []
    
    for session_num in range(1, NUM_SESSIONS + 1):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        avi_dir = session_dir / "dialog" / "avi" / "DivX"
        
        if not avi_dir.exists():
            continue
        
        # 创建session子目录
        session_output_dir = VIDEO_OUTPUT_DIR / f"Session{session_num}"
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        for avi_file in avi_dir.glob("*.avi"):
            # 跳过 macOS 元数据文件 (以 ._ 开头)
            if avi_file.name.startswith('._'):
                continue
            mp4_file = session_output_dir / f"{avi_file.stem}.mp4"
            conversion_tasks.append((avi_file, mp4_file))
    
    print(f"找到 {len(conversion_tasks)} 个视频文件需要处理")
    
    # 并行转换
    success_count = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(convert_avi_to_mp4, avi, mp4): (avi, mp4)
            for avi, mp4 in conversion_tasks
        }
        
        with tqdm(total=len(futures), desc="转换视频") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)
    
    print(f"✓ 视频转换完成: {success_count}/{len(conversion_tasks)}")


def copy_audios():
    """复制分割好的音频文件到新目录"""
    print("\n" + "=" * 50)
    print("步骤 2: 整理音频文件")
    print("=" * 50)
    
    audio_count = 0
    
    for session_num in range(1, NUM_SESSIONS + 1):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        wav_dir = session_dir / "sentences" / "wav"
        
        if not wav_dir.exists():
            continue
        
        session_output_dir = AUDIO_OUTPUT_DIR / f"Session{session_num}"
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历每个对话的音频目录
        for dialogue_dir in wav_dir.iterdir():
            if not dialogue_dir.is_dir():
                continue
            
            dialogue_output_dir = session_output_dir / dialogue_dir.name
            dialogue_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建符号链接或复制文件
            for wav_file in dialogue_dir.glob("*.wav"):
                target = dialogue_output_dir / wav_file.name
                if not target.exists():
                    # 使用符号链接节省空间
                    target.symlink_to(wav_file)
                audio_count += 1
    
    print(f"✓ 音频整理完成: {audio_count} 个文件")


def generate_meta_csv():
    """生成统一的CSV标签文件"""
    print("\n" + "=" * 50)
    print("步骤 3: 生成标签CSV文件")
    print("=" * 50)
    
    all_records = []
    
    for session_num in range(1, NUM_SESSIONS + 1):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        
        if not emo_dir.exists():
            print(f"⚠ 警告: 目录不存在 {emo_dir}")
            continue
        
        # 处理每个情感标注文件（排除macOS元数据文件和隐藏文件）
        for emo_file in emo_dir.glob("*.txt"):
            if emo_file.name.startswith('.') or emo_file.name.startswith('._'):
                continue
            
            dialogue_id = emo_file.stem
            
            # 解析情感标注
            annotations = parse_emotion_file(emo_file)
            
            # 解析文本转录
            trans_file = trans_dir / f"{dialogue_id}.txt"
            transcriptions = parse_transcription_file(trans_file)
            
            # 合并信息
            for utterance_id, anno in annotations.items():
                # 提取说话人 (F/M)
                speaker_match = re.search(r'_([FM])\d+$', utterance_id)
                speaker = speaker_match.group(1) if speaker_match else 'Unknown'
                
                # 确定视频文件路径
                # 根据说话人选择对应的视频
                video_filename = f"{dialogue_id}.mp4"
                video_path = f"videos/Session{session_num}/{video_filename}"
                
                # 音频文件路径
                audio_filename = f"{utterance_id}.wav"
                audio_path = f"audios/Session{session_num}/{dialogue_id}/{audio_filename}"
                
                record = {
                    'utterance_id': utterance_id,
                    'dialogue_id': dialogue_id,
                    'session': session_num,
                    'speaker': speaker,
                    'text': transcriptions.get(utterance_id, ''),
                    'emotion': anno['emotion'],
                    'valence': anno['valence'],
                    'activation': anno['activation'],
                    'dominance': anno['dominance'],
                    'start_time': anno['start_time'],
                    'end_time': anno['end_time'],
                    'video_path': video_path,
                    'audio_path': audio_path
                }
                all_records.append(record)
    
    # 按session和utterance_id排序
    all_records.sort(key=lambda x: (x['session'], x['dialogue_id'], x['start_time']))
    
    # 写入CSV
    fieldnames = [
        'utterance_id', 'dialogue_id', 'session', 'speaker', 'text',
        'emotion', 'valence', 'activation', 'dominance',
        'start_time', 'end_time', 'video_path', 'audio_path'
    ]
    
    with open(META_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    
    print(f"✓ CSV文件已生成: {META_CSV_PATH}")
    print(f"  总记录数: {len(all_records)}")
    
    # 统计情感分布
    emotion_counts = {}
    for record in all_records:
        emo = record['emotion']
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    print("\n情感标签分布:")
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count}")


def print_summary():
    """打印处理摘要"""
    print("\n" + "=" * 50)
    print("处理完成！")
    print("=" * 50)
    print(f"""
输出目录结构:
{OUTPUT_ROOT}/
├── videos/                    # 转换后的MP4视频
│   ├── Session1/
│   │   ├── Ses01F_impro01.mp4
│   │   ├── Ses01M_impro01.mp4
│   │   └── ...
│   ├── Session2/
│   └── ...
├── audios/                    # 分割好的音频 (符号链接)
│   ├── Session1/
│   │   ├── Ses01F_impro01/
│   │   │   ├── Ses01F_impro01_F000.wav
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── meta.csv                   # 统一标签文件

CSV字段说明:
- utterance_id: 语句唯一标识 (如 Ses01F_impro01_F000)
- dialogue_id:  对话标识 (如 Ses01F_impro01)
- session:      Session编号 (1-5)
- speaker:      说话人 (F=Female, M=Male)
- text:         语句文本
- emotion:      情感标签 (neu/ang/hap/sad/fru/exc/xxx等)
- valence:      效价 (1-5)
- activation:   激活度 (1-5)
- dominance:    支配度 (1-5)
- start_time:   开始时间 (秒)
- end_time:     结束时间 (秒)
- video_path:   视频相对路径
- audio_path:   音频相对路径
""")


def main():
    print("=" * 50)
    print("IEMOCAP 数据集预处理脚本")
    print("=" * 50)
    
    # 检查源目录
    if not IEMOCAP_ROOT.exists():
        print(f"✗ 错误: IEMOCAP目录不存在: {IEMOCAP_ROOT}")
        return
    
    # 创建输出目录
    ensure_dirs()
    
    # 步骤1: 转换视频
    process_videos()
    
    # 步骤2: 整理音频
    copy_audios()
    
    # 步骤3: 生成CSV
    generate_meta_csv()
    
    # 打印摘要
    print_summary()


if __name__ == "__main__":
    main()

