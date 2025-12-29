# -*- coding: utf-8 -*-
"""
统一数据集预处理脚本
支持 CH-SIMSV2S, CH-SIMS, MELD 数据集
将它们统一处理成相同的格式（utterances.pkl + splits.pkl）

数据集结构：
1. CH-SIMSV2S / CH-SIMS:
   - Raw/video_xxxx/xxxx.mp4
   - meta.csv (包含所有标注和 train/valid/test 划分)

2. MELD:
   - train/train/dia{x}_utt{y}.mp4
   - dev/dev/dia{x}_utt{y}.mp4
   - test/test/dia{x}_utt{y}.mp4
   - 各自的 CSV 文件

目标格式（与原始 chsimsv2 一致）：
  output_dir/
    video_xxxx/ (或 dia_xxxx/)
      - utterances.pkl
        {
          'video_id': str,
          'utterance_ids': [str, ...],
          'utterances': {
            'utt_id': {
              'vision': [T, D_v],
              'audio': [T, D_a],
              'text': [D_t],
              'social': [D_s],       # utterance 级别
              'context': [D_c],      # utterance 级别
              'social_text': str,    # 原始文本
              'context_text': str,   # 原始文本
              'label': float,
            },
            ...
          },
        }
    splits.pkl  # {'train': [...], 'valid': [...], 'test': [...]}
"""

import os
import sys
import pickle
import argparse
import traceback
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import cv2
import ffmpeg
from tqdm import tqdm
from PIL import Image
from openai import OpenAI

# ========== 导入依赖 ==========
import open_clip
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sentence_transformers import SentenceTransformer

# ========== 默认配置 ==========

DEFAULT_SOCIAL_PROMPT = """请分析这个视频中的社会关系信息：
1. 视频中有多少个角色？他们之间是什么关系？
2. 角色之间的互动模式是什么？（如：友好、对立、上下级等）
3. 社交场景的特征是什么？（如：正式会议、家庭聚会、商业谈判等）

请用简洁的语言描述，不要超过150字。将结果用<answer></answer>标签包裹。"""

DEFAULT_CONTEXT_PROMPT = """请分析这个视频的情境信息：
1. 场景环境是什么？（如：室内/室外、办公室/家庭等）
2. 整体氛围如何？（如：紧张、轻松、严肃、欢快等）
3. 事件的背景和上下文是什么？

请用简洁的语言描述，不要超过150字。将结果用<answer></answer>标签包裹。"""

# ========== 工具函数 ==========

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def extract_audio_waveform(video_path: str, sr: int) -> Tuple[np.ndarray, int]:
    """用 ffmpeg 从视频中提取单声道 float32 音频"""
    try:
        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        return audio, sr
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg 解码失败: {e.stderr.decode(errors='ignore')}")


def sample_frames_uniform(video_path: str, target_fps: float) -> Tuple[List[np.ndarray], float, int, int, np.ndarray]:
    """按时间均匀采样帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV 无法打开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(fps, 1e-6)

    target_count = max(1, int(round(duration * max(target_fps, 1e-6))))
    idxs = np.linspace(0, max(0, total_frames - 1), target_count)
    idxs = np.clip(np.floor(idxs).astype(int), 0, max(0, total_frames - 1))

    frames: List[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    kept = len(frames)
    return frames, fps, total_frames, kept, idxs[:kept]


def pool_audio_to_frames(
    audio_emb: np.ndarray,
    audio_dur: float,
    frame_times: np.ndarray,
    frame_fps: float,
    pad_sec: float = 0.0
) -> np.ndarray:
    """音频特征对齐到视频帧"""
    T_a, D_a = audio_emb.shape
    if T_a <= 0:
        return np.zeros((len(frame_times), D_a), dtype=np.float32)

    hop = audio_dur / T_a
    audio_times = (np.arange(T_a) + 0.5) * hop
    frame_step = 1.0 / max(frame_fps, 1e-6)

    pooled = np.zeros((len(frame_times), D_a), dtype=np.float32)
    for i, t in enumerate(frame_times):
        left = t - 0.5 * frame_step - pad_sec
        right = t + 0.5 * frame_step + pad_sec
        idx = np.where((audio_times >= left) & (audio_times <= right))[0]
        if idx.size == 0:
            j = int(np.clip(round(t / hop - 0.5), 0, T_a - 1))
            pooled[i] = audio_emb[j]
        else:
            pooled[i] = audio_emb[idx].mean(axis=0)
    return pooled


def l2_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n


def encode_video_to_base64(video_path: str) -> str:
    """将视频编码为 base64"""
    import base64
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def merge_videos_for_mllm(video_paths: List[str], output_path: str, max_duration: float = 60.0) -> bool:
    """
    合并多个 utterance 视频为一个视频，用于 MLLM 分析
    
    Args:
        video_paths: 要合并的视频文件路径列表
        output_path: 输出的合并视频路径
        max_duration: 最大视频时长（秒），超过则跳过后续视频
        
    Returns:
        是否成功合并
    """
    import subprocess
    import tempfile
    
    if not video_paths:
        return False
    
    # 如果只有一个视频，直接复制
    if len(video_paths) == 1:
        import shutil
        try:
            shutil.copy2(video_paths[0], output_path)
            return True
        except Exception as e:
            print(f"    ⚠️  复制视频失败: {e}")
            return False
    
    # 过滤出存在的视频文件
    valid_paths = [p for p in video_paths if Path(p).exists()]
    if not valid_paths:
        return False
    
    # 创建临时文件列表（ffmpeg concat 格式）
    try:
        # 计算每个视频的时长，限制总时长
        selected_paths = []
        total_duration = 0.0
        
        for vp in valid_paths:
            try:
                # 获取视频时长
                probe = ffmpeg.probe(vp)
                duration = float(probe['format'].get('duration', 0))
                
                if total_duration + duration <= max_duration or len(selected_paths) == 0:
                    selected_paths.append(vp)
                    total_duration += duration
                else:
                    # 超过最大时长，停止添加
                    break
            except Exception:
                # 无法获取时长，假设较短
                selected_paths.append(vp)
                total_duration += 3.0  # 假设平均3秒
                if total_duration > max_duration:
                    break
        
        if not selected_paths:
            return False
        
        # 创建 ffmpeg concat 文件列表
        # 注意：必须使用绝对路径，因为 ffmpeg concat 会基于 concat 文件位置解析相对路径
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            for vp in selected_paths:
                # 转换为绝对路径并转义特殊字符
                abs_path = str(Path(vp).resolve())
                escaped_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # 使用 ffmpeg concat demuxer 合并视频
        # 使用 subprocess 而不是 ffmpeg-python，更可控
        cmd = [
            'ffmpeg', '-y',  # 覆盖输出
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # 直接复制流，不重新编码
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        
        # 清理临时文件
        try:
            os.unlink(concat_file)
        except:
            pass
        
        if result.returncode != 0:
            # 如果直接复制失败，尝试重新编码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for vp in selected_paths:
                    # 转换为绝对路径
                    abs_path = str(Path(vp).resolve())
                    escaped_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            try:
                os.unlink(concat_file)
            except:
                pass
            
            if result.returncode != 0:
                print(f"    ⚠️  视频合并失败: {result.stderr.decode(errors='ignore')[:200]}")
                return False
        
        return Path(output_path).exists()
        
    except Exception as e:
        print(f"    ⚠️  视频合并异常: {e}")
        return False


def call_mllm(video_path: str, prompt: str, client: OpenAI, model_name: str, desc: str = "MLLM", temperature: float = 0.7) -> str:
    """调用 MLLM 分析视频"""
    import time
    
    # 编码视频
    video_base64 = encode_video_to_base64(video_path)
    
    # 构造请求
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_base64}",
                        "mime_type": "video/mp4"
                    }
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        result = response.choices[0].message.content.strip()
        
        # 提取 <answer></answer> 标签中的内容
        match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
        if match:
            return match.group(1).strip()
        return result
    
    except Exception as e:
        print(f"    ⚠️  MLLM 调用失败: {e}")
        return ""


# ========== 并行 MLLM 调用 ==========

class ParallelMLLMProcessor:
    """并行MLLM处理器 - 支持多线程并发调用API"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        text_model,
        num_workers: int = 4,
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.text_model = text_model
        self.num_workers = num_workers
        self.temperature = temperature
        self.lock = threading.Lock()
        self._local = threading.local()
    
    def _get_client(self) -> OpenAI:
        """获取线程本地的OpenAI客户端"""
        if not hasattr(self._local, 'client'):
            self._local.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._local.client
    
    def call_mllm_single(self, video_path: str, prompt: str, desc: str = "MLLM") -> str:
        """单个MLLM调用"""
        try:
            client = self._get_client()
            video_base64 = encode_video_to_base64(video_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}",
                                "mime_type": "video/mp4"
                            }
                        }
                    ]
                }
            ]
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            result = response.choices[0].message.content.strip()
            
            # 提取 <answer></answer> 标签中的内容
            match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
            if match:
                return match.group(1).strip()
            return result
        
        except Exception as e:
            return ""
    
    def process_utterance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个utterance的MLLM任务"""
        result = {
            'video_id': task['video_id'],
            'utterance_id': task['utterance_id'],
            'social_text': '',
            'context_text': '',
            'social_emb': None,
            'context_emb': None,
        }
        
        video_path = task['video_path']
        if not Path(video_path).exists():
            return result
        
        D_s = self.text_model.get_sentence_embedding_dimension()
        
        try:
            # 社会关系
            if task.get('extract_social', True):
                social_text = self.call_mllm_single(video_path, task['social_prompt'], "社会关系")
                result['social_text'] = social_text
                if social_text:
                    social_emb = self.text_model.encode(social_text, convert_to_numpy=True, normalize_embeddings=True)
                    result['social_emb'] = social_emb.astype(np.float32)
                else:
                    result['social_emb'] = np.zeros(D_s, dtype=np.float32)
            
            # 情境
            if task.get('extract_context', True):
                context_text = self.call_mllm_single(video_path, task['context_prompt'], "情境分析")
                result['context_text'] = context_text
                if context_text:
                    context_emb = self.text_model.encode(context_text, convert_to_numpy=True, normalize_embeddings=True)
                    result['context_emb'] = context_emb.astype(np.float32)
                else:
                    result['context_emb'] = np.zeros(D_s, dtype=np.float32)
        
        except Exception as e:
            if result['social_emb'] is None:
                result['social_emb'] = np.zeros(D_s, dtype=np.float32)
            if result['context_emb'] is None:
                result['context_emb'] = np.zeros(D_s, dtype=np.float32)
        
        return result
    
    def process_batch(self, tasks: List[Dict[str, Any]], desc: str = "MLLM批处理") -> Dict[Tuple[str, str], Dict[str, Any]]:
        """并行处理一批任务，返回 {(video_id, utterance_id): result}"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_utterance, task): (task['video_id'], task['utterance_id'])
                for task in tasks
            }
            
            with tqdm(total=len(futures), desc=desc, leave=False) as pbar:
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        result = future.result()
                        results[key] = result
                    except Exception as e:
                        pass
                    pbar.update(1)
        
        return results


# ========== 数据集加载器 ==========

class DatasetLoader:
    """数据集加载器基类"""
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
    
    def load_metadata(self) -> pd.DataFrame:
        """加载元数据，返回统一格式的 DataFrame"""
        raise NotImplementedError
    
    def get_video_groups(self, meta_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """将元数据按视频分组"""
        raise NotImplementedError
    
    def get_video_path(self, video_id: str, utterance_id: str, mode: str = None) -> str:
        """获取视频文件路径"""
        raise NotImplementedError


class CHSIMSLoader(DatasetLoader):
    """CH-SIMSV2S / CH-SIMS 数据加载器"""
    
    def load_metadata(self) -> pd.DataFrame:
        meta_path = self.input_dir / 'meta.csv'
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.csv not found in {self.input_dir}")
        
        df = pd.read_csv(
            meta_path,
            dtype={
                'video_id': str,
                'clip_id': str,
            }
        )
        print(f"Loaded meta.csv: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # 统一格式
        df = df.rename(columns={
            'clip_id': 'utterance_id',
            'text': 'text',
            'label': 'label',
            'label_T': 'label_T',
            'label_A': 'label_A',
            'label_V': 'label_V',
            'mode': 'split',
        })
        # clip_id 可能被解析为整数，确保保持原始的零填充格式
        df['utterance_id'] = df['utterance_id'].astype(str).str.zfill(4)
        
        return df
    
    def get_video_groups(self, meta_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按 video_id 分组"""
        groups = {}
        for video_id, group in meta_df.groupby('video_id'):
            groups[video_id] = group.sort_values('utterance_id')
        return groups
    
    def get_video_path(self, video_id: str, utterance_id: str, mode: str = None) -> Path:
        """获取视频文件路径"""
        return self.input_dir / 'Raw' / video_id / f"{utterance_id}.mp4"


class MELDLoader(DatasetLoader):
    """MELD 数据加载器"""
    
    def __init__(self, input_dir: str):
        super().__init__(input_dir)
        self.split_mapping = {
            'train': 'train',
            'dev': 'valid',
            'test': 'test'
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """加载并合并三个 split 的 CSV"""
        dfs = []
        
        for split_name in ['train', 'dev', 'test']:
            csv_path = self.input_dir / split_name / f"{split_name}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
            
            # 正确解析CSV（使用csv.DictReader确保正确处理引号内的逗号）
            import csv
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # 验证必需字段存在且为数字
                        dia_id = int(row['Dialogue_ID'])
                        utt_id = int(row['Utterance_ID'])
                        rows.append(row)
                    except (ValueError, KeyError) as e:
                        # 跳过解析错误的行
                        pass
            
            df = pd.DataFrame(rows)
            
            if len(df) == 0:
                print(f"Error: No valid data in {csv_path}")
                continue
            
            print(f"  Loaded {split_name}: {len(df)} rows")
            
            # 立即验证：Dialogue_ID和Utterance_ID必须保持为字符串类型的数字
            # 确保所有值都是有效的数字字符串
            df['Dialogue_ID'] = df['Dialogue_ID'].astype(str).str.strip()
            df['Utterance_ID'] = df['Utterance_ID'].astype(str).str.strip()
            
            # 验证：dialogue 0 和 1 的行数
            if split_name == 'train':
                dia0_count = len(df[df['Dialogue_ID'] == '0'])
                dia1_count = len(df[df['Dialogue_ID'] == '1'])
                print(f"    验证: Dialogue_0={dia0_count}行, Dialogue_1={dia1_count}行")
            
            # 检查列名 - 使用Emotion列（7分类）而非Sentiment列（3分类）
            required_cols = ['Dialogue_ID', 'Utterance_ID', 'Utterance', 'Emotion']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Missing columns in {csv_path}: {missing_cols}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            df['split'] = self.split_mapping[split_name]
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded MELD metadata: {len(combined_df)} rows")
        print(f"Columns: {combined_df.columns.tolist()}")
        
        # 统一格式
        combined_df = combined_df.rename(columns={
            'Dialogue_ID': 'video_id',
            'Utterance_ID': 'utterance_id',
            'Utterance': 'text',
            'Emotion': 'label_text',  # 使用 Emotion 列（7分类）
        })
        
        # 将 Emotion 转换为数值标签（7分类：0-6）
        # MELD的7个情感类别
        emotion_map = {
            'neutral': 0,    # 中性
            'joy': 1,        # 喜悦
            'sadness': 2,    # 悲伤
            'anger': 3,      # 愤怒
            'surprise': 4,   # 惊讶
            'fear': 5,       # 恐惧
            'disgust': 6,    # 厌恶
        }
        combined_df['label'] = combined_df['label_text'].map(emotion_map)
        
        # 打印标签分布统计
        print(f"\nMELD 7-class emotion distribution:")
        for emotion, idx in sorted(emotion_map.items(), key=lambda x: x[1]):
            count = len(combined_df[combined_df['label'] == idx])
            print(f"  {idx}: {emotion} = {count}")
        
        # 删除label为空的行（如果有解析错误）
        before_len = len(combined_df)
        combined_df = combined_df.dropna(subset=['label', 'video_id', 'utterance_id'])
        after_len = len(combined_df)
        if before_len != after_len:
            print(f"Warning: Dropped {before_len - after_len} rows with invalid data")
        
        # video_id 和 utterance_id 已经是字符串了（从csv.DictReader来的）
        # MELD的Dialogue_ID在train/dev/test中是独立编号的，需要加上split前缀区分
        combined_df['video_id'] = combined_df['split'] + '_dia_' + combined_df['video_id']
        combined_df['utterance_id'] = 'utt_' + combined_df['utterance_id']
        
        # 显示统计信息
        print(f"Unique dialogues: {combined_df['video_id'].nunique()}")
        print(f"Total utterances: {len(combined_df)}")
        
        # 验证前几个dialogue的数据（每个split独立）
        print(f"\n数据验证:")
        for dia_id in ['train_dia_0', 'train_dia_1', 'dev_dia_0']:
            dia_data = combined_df[combined_df['video_id'] == dia_id]
            if len(dia_data) > 0:
                utt_ids = sorted(dia_data['utterance_id'].unique(), key=lambda x: int(x.replace('utt_', '')))
                print(f"  {dia_id}: {len(dia_data)} utterances = {utt_ids}")
        
        return combined_df
    
    def get_video_groups(self, meta_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按 video_id (Dialogue_ID) 分组"""
        groups = {}
        for video_id, group in meta_df.groupby('video_id'):
            # 验证分组后的数据确实只包含这个video_id
            unique_vids = group['video_id'].unique()
            if len(unique_vids) != 1 or unique_vids[0] != video_id:
                print(f"  ❌ 错误: 分组{video_id}包含了其他video的数据: {unique_vids}")
                continue
            
            # 按utterance_id排序（需要按数字排序，不是字符串排序）
            # 添加临时列用于数字排序
            group = group.copy()
            group['_utt_num'] = group['utterance_id'].str.replace('utt_', '').astype(int)
            group = group.sort_values('_utt_num').drop(columns=['_utt_num'])
            groups[video_id] = group
            
        
        return groups
    
    def get_video_path(self, video_id: str, utterance_id: str, mode: str) -> Path:
        """获取视频文件路径"""
        # video_id 格式: "train_dia_0" 或 "dev_dia_0"
        # utterance_id 格式: "utt_0"
        
        # 从video_id中提取split和dialogue编号
        if '_dia_' in video_id:
            split_prefix, dia_part = video_id.split('_dia_')
            dia_num = dia_part
        else:
            # 兼容旧格式（如果没有split前缀）
            dia_num = video_id.replace('dia_', '')
            split_prefix = mode
        
        utt_num = utterance_id.replace('utt_', '')
        
        # MELD 视频路径: {split}/dia{num}_utt{num}.mp4
        # 例如: train/dia0_utt0.mp4, dev/dia1_utt2.mp4, test/dia5_utt3.mp4
        split_dir_map = {'train': 'train', 'valid': 'dev', 'test': 'test'}
        split_dir_name = split_dir_map.get(split_prefix, split_prefix)
        
        video_file = f"dia{dia_num}_utt{utt_num}.mp4"
        return self.input_dir / split_dir_name / video_file


# ========== 特征提取 ==========

def process_utterance(
    utterance_path: str,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    frame_rate: float,
    target_sr: int,
    batch_size_img: int,
    audio_pad_sec: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个语句的视频，提取视觉和音频特征
    
    Returns:
        (vision_features, audio_features)
    """
    # 视觉特征
    frames, fps, total_frames, kept, idxs = sample_frames_uniform(utterance_path, frame_rate)
    duration = total_frames / max(fps, 1e-6)
    frame_times = (idxs[:kept] + 0.5) / max(fps, 1e-6)
    
    # 批量提取视觉特征
    img_emb_list = []
    num_batches = (kept + batch_size_img - 1) // batch_size_img
    with torch.no_grad():
        for i in range(0, kept, batch_size_img):
            batch_imgs = frames[i:i + batch_size_img]
            batch_tensors = [clip_preprocess(Image.fromarray(fr)) for fr in batch_imgs]
            batch = torch.stack(batch_tensors).to(device)
            feat = clip_model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_emb_list.append(feat.detach().cpu().numpy())
    img_emb = np.concatenate(img_emb_list, axis=0).astype(np.float32)  # [T, D_v]
    
    # 音频特征
    try:
        wav, sr = extract_audio_waveform(utterance_path, target_sr)
        adur = len(wav) / float(sr)
        
        wav_tensor = torch.from_numpy(wav.copy()).unsqueeze(0)
        a_in = wavlm_feat(
            wav_tensor.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            a_out = wavlm(a_in.input_values.to(device))
        a_emb = a_out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        # 对齐到帧
        a_pooled = pool_audio_to_frames(
            a_emb, audio_dur=adur, frame_times=frame_times,
            frame_fps=kept / max(duration, 1e-6), pad_sec=audio_pad_sec
        )
        a_pooled = l2_norm(a_pooled, axis=1)
    
    except Exception as e:
        print(f"      ⚠️  音频提取失败: {e}")
        D_a = 1024
        a_pooled = np.zeros((kept, D_a), dtype=np.float32)
    
    return img_emb, a_pooled


def update_prior_only(
    video_id: str,
    utterances_df: pd.DataFrame,
    dataset_loader: DatasetLoader,
    output_dir: Path,
    text_model,
    mllm_processor: Optional[ParallelMLLMProcessor],
    args,
) -> bool:
    """
    只更新先验特征（社会关系和/或情境）- 支持并行处理
    读取已有的 utterances.pkl，为每个 utterance 单独提取先验特征
    
    Returns:
        bool: 是否成功
    """
    video_pkl = output_dir / video_id / 'utterances.pkl'
    
    if not video_pkl.exists():
        print(f"  ⚠️  {video_id}: utterances.pkl 不存在，跳过")
        return False
    
    try:
        # 加载现有数据
        with open(video_pkl, 'rb') as f:
            video_data = pickle.load(f)
        
        # 获取 split 信息
        mode = utterances_df.iloc[0]['split']
        
        # 根据 extract_mode 决定提取哪些特征
        extract_social = args.extract_mode in ['social', 'prior']
        extract_context = args.extract_mode in ['context', 'prior']
        
        D_s = text_model.get_sentence_embedding_dimension()
        
        # 收集 MLLM 任务
        utterance_ids = video_data.get('utterance_ids', [])
        mllm_tasks = []
        
        for utt_id in utterance_ids:
            # 获取视频路径
            video_path = dataset_loader.get_video_path(video_id, utt_id, mode)
            if not video_path.exists():
                continue
            
            mllm_tasks.append({
                'video_id': video_id,
                'utterance_id': utt_id,
                'video_path': str(video_path),
                'social_prompt': args.social_prompt,
                'context_prompt': args.context_prompt,
                'extract_social': extract_social,
                'extract_context': extract_context,
            })
        
        if not mllm_tasks:
            print(f"  ⚠️  {video_id}: 没有找到有效的视频文件")
            return False
        
        # 并行调用 MLLM
        if mllm_processor:
            print(f"  并行MLLM调用 ({len(mllm_tasks)} 任务, {mllm_processor.num_workers} 并行)")
            mllm_results = mllm_processor.process_batch(mllm_tasks, desc=f"  更新先验特征")
            
            success_count = 0
            for (vid, utt_id), result in mllm_results.items():
                # 确保 utterance 数据存在
                if utt_id not in video_data['utterances']:
                    video_data['utterances'][utt_id] = {}
                
                utt_data = video_data['utterances'][utt_id]
                
                if extract_social:
                    utt_data['social'] = result.get('social_emb', np.zeros(D_s, dtype=np.float32))
                    utt_data['social_text'] = result.get('social_text', '')
                
                if extract_context:
                    utt_data['context'] = result.get('context_emb', np.zeros(D_s, dtype=np.float32))
                    utt_data['context_text'] = result.get('context_text', '')
                
                if result.get('social_text') or result.get('context_text'):
                    success_count += 1
            
            # 保存更新后的数据
            with open(video_pkl, 'wb') as f:
                pickle.dump(video_data, f)
            
            print(f"  ✅ 成功更新 {success_count}/{len(utterance_ids)} 个 utterance 的先验特征")
            return success_count > 0
        else:
            print(f"  ⚠️  MLLM处理器未初始化，跳过")
            return False
    
    except Exception as e:
        print(f"  ❌ {video_id}: 更新失败 - {e}")
        return False


def process_video(
    video_id: str,
    utterances_df: pd.DataFrame,
    dataset_loader: DatasetLoader,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    text_model,
    mllm_processor: Optional[ParallelMLLMProcessor],
    args,
) -> Dict:
    """
    处理单个视频的所有语句（支持并行MLLM调用）
    
    Returns:
        {
            'video_id': str,
            'utterance_ids': List[str],
            'utterances': Dict[utterance_id -> features],  # 每个 utterance 包含 social 和 context
        }
    """
    print(f"\n处理视频: {video_id}")
    
    video_data = {
        'video_id': video_id,
        'utterance_ids': [],
        'utterances': {},
    }
    
    if len(utterances_df) == 0:
        print(f"  ⚠️  没有找到 {video_id} 的语句信息")
        return None
    
    print(f"  - CSV中记录 {len(utterances_df)} 个语句")
    
    # 获取 split 信息（用于找视频路径）
    mode = utterances_df.iloc[0]['split']
    
    # 判断是否需要提取基本模态和先验模态
    extract_basic = args.extract_mode in ['all', 'basic']
    extract_prior = args.extract_mode in ['all', 'social', 'context', 'prior']
    extract_social = args.extract_mode in ['all', 'social', 'prior']
    extract_context = args.extract_mode in ['all', 'context', 'prior']
    
    D_s = text_model.get_sentence_embedding_dimension()
    
    # ========== 第一阶段：提取基本模态特征 ==========
    pbar_utterances = tqdm(utterances_df.iterrows(), total=len(utterances_df), 
                           desc=f"  处理语句", leave=False)
    
    skipped_files = []  # 记录跳过的文件
    mllm_tasks = []  # 收集 MLLM 任务
    
    for idx, row in pbar_utterances:
        utterance_id = row['utterance_id']
        text_content = row['text']
        label = float(row['label'])
        
        # 尝试获取单模态标签（如果存在）
        label_T = float(row.get('label_T', 0.0))
        label_A = float(row.get('label_A', 0.0))
        label_V = float(row.get('label_V', 0.0))
        
        pbar_utterances.set_postfix({'utt_id': utterance_id})
        
        # 找到对应的视频文件
        utterance_file = dataset_loader.get_video_path(video_id, utterance_id, mode)
        
        if not utterance_file.exists():
            # CSV中有这条记录，但视频文件不存在
            print(f"  ⚠️  未找到语句文件: {utterance_file}")
            skipped_files.append(utterance_id)
            continue
        
        try:
            utterance_data = {
                'label': label,
                'label_T': label_T,
                'label_A': label_A,
                'label_V': label_V
            }
            
            # ===== 提取基本模态（视觉、音频、文本）=====
            if extract_basic:
                # 提取视觉和音频特征
                utterance_vision, utterance_audio = process_utterance(
                    str(utterance_file),
                    clip_model,
                    clip_preprocess,
                    wavlm,
                    wavlm_feat,
                    args.frame_rate,
                    args.target_sr,
                    args.batch_size_img,
                    0.0,  # audio_pad_sec
                    args.device,
                )
                
                # 文本特征（从 meta 读取并编码）
                text_emb = text_model.encode(text_content, convert_to_numpy=True, normalize_embeddings=True)
                text_emb = text_emb.astype(np.float32)
                
                utterance_data['vision'] = utterance_vision
                utterance_data['audio'] = utterance_audio
                utterance_data['text'] = text_emb
            
            # 初始化先验特征（零向量），稍后并行填充
            if extract_prior:
                if extract_social:
                    utterance_data['social'] = np.zeros(D_s, dtype=np.float32)
                    utterance_data['social_text'] = ''
                if extract_context:
                    utterance_data['context'] = np.zeros(D_s, dtype=np.float32)
                    utterance_data['context_text'] = ''
                
                # 收集 MLLM 任务
                if mllm_processor and not args.skip_mllm:
                    mllm_tasks.append({
                        'video_id': video_id,
                        'utterance_id': utterance_id,
                        'video_path': str(utterance_file),
                        'social_prompt': args.social_prompt,
                        'context_prompt': args.context_prompt,
                        'extract_social': extract_social,
                        'extract_context': extract_context,
                    })
            
            video_data['utterance_ids'].append(utterance_id)
            video_data['utterances'][utterance_id] = utterance_data
        
        except Exception as e:
            print(f"  ⚠️  处理语句 {utterance_id} 失败: {e}")
            continue
    
    # ========== 第二阶段：并行调用 MLLM ==========
    if mllm_tasks and mllm_processor:
        print(f"  并行MLLM调用 ({len(mllm_tasks)} 任务, {mllm_processor.num_workers} 并行)")
        mllm_results = mllm_processor.process_batch(mllm_tasks, desc="  MLLM处理")
        
        # 将结果填充到 utterance 数据中
        for (vid, utt_id), result in mllm_results.items():
            if utt_id in video_data['utterances']:
                if result.get('social_emb') is not None:
                    video_data['utterances'][utt_id]['social'] = result['social_emb']
                    video_data['utterances'][utt_id]['social_text'] = result['social_text']
                if result.get('context_emb') is not None:
                    video_data['utterances'][utt_id]['context'] = result['context_emb']
                    video_data['utterances'][utt_id]['context_text'] = result['context_text']
    
    if len(video_data['utterances']) == 0:
        print(f"  ❌ 没有成功处理任何语句 (CSV中有{len(utterances_df)}条记录，但所有视频文件都不存在)")
        return None
    
    processed_count = len(video_data['utterances'])
    total_count = len(utterances_df)
    skipped_count = len(skipped_files)
    if skipped_count > 0:
        print(f"  ✅ 成功处理 {processed_count}/{total_count} 个语句 (跳过 {skipped_count} 个缺失文件)")
    else:
        print(f"  ✅ 成功处理 {processed_count}/{total_count} 个语句")
    
    return video_data


def main():
    parser = argparse.ArgumentParser(description='统一数据集预处理脚本')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['chsimsv2', 'chsims', 'meld'],
                        help='数据集名称')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    # 模式选择
    parser.add_argument('--extract_mode', type=str, default='all',
                        choices=['all', 'basic', 'social', 'context', 'prior'],
                        help='提取模式：all=完整提取5个模态, basic=仅基本模态(视觉/音频/文本), social=仅社会关系, context=仅情境, prior=社会关系+情境')
    parser.add_argument('--skip_mllm', action='store_true',
                        help='跳过 MLLM 特征提取（用于测试）')
    
    # 模型路径参数
    parser.add_argument('--clip_model', type=str, default='ViT-g-14',
                        help='OpenCLIP 模型名称')
    parser.add_argument('--clip_ckpt', type=str, 
                        default='./models/open_clip/vit_g14_laion2b/open_clip_pytorch_model.bin',
                        help='OpenCLIP 权重路径')
    parser.add_argument('--wavlm_path', type=str, default='./models/wavlm/wavlm-large',
                        help='WavLM 模型路径')
    parser.add_argument('--text_model_path', type=str, 
                        default='./models/sent/paraphrase-multilingual-mpnet-base-v2',
                        help='文本模型路径（支持中文）')
    
    # 处理参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备（cuda:0, cuda:1, cpu）')
    parser.add_argument('--frame_rate', type=float, default=3.0,
                        help='视频抽帧率（fps）')
    parser.add_argument('--batch_size_img', type=int, default=32,
                        help='图像批处理大小')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='音频采样率')
    
    # MLLM 参数
    parser.add_argument('--api_key', type=str, default='sk-CopXuPMUxmJY7UNSXrjyBA',
                        help='MLLM API Key')
    parser.add_argument('--base_url', type=str, default='https://llm.rekeymed.com/v1/',
                        help='MLLM API Base URL')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B',
                        help='MLLM 模型名称')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='MLLM 温度参数 (0.0-2.0, 越高越随机)')
    parser.add_argument('--max_merge_duration', type=float, default=60.0,
                        help='合并视频的最大时长（秒），超过则只取部分utterance')
    
    # 并行参数
    parser.add_argument('--parallel_workers', type=int, default=4,
                        help='MLLM并行调用数量（可根据API限制调整）')
    
    # 提示词
    parser.add_argument('--social_prompt', type=str, default=DEFAULT_SOCIAL_PROMPT,
                        help='社会关系提示词')
    parser.add_argument('--context_prompt', type=str, default=DEFAULT_CONTEXT_PROMPT,
                        help='情境提示词')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    safe_makedirs(str(output_dir))
    
    # 显示提取模式
    print("\n" + "="*60)
    print(f"加载数据集: {args.dataset}")
    print(f"提取模式: {args.extract_mode}")
    print(f"MLLM并行数: {args.parallel_workers}")
    print("="*60)
    
    mode_desc = {
        'all': '完整提取（视觉+音频+文本+社会关系+情境）',
        'basic': '仅基本模态（视觉+音频+文本）',
        'social': '仅社会关系（重提取）',
        'context': '仅情境（重提取）',
        'prior': '先验模态（社会关系+情境，重提取）',
    }
    print(f"说明: {mode_desc.get(args.extract_mode, '未知模式')}")
    print("="*60)
    
    if args.dataset in ['chsimsv2', 'chsims']:
        dataset_loader = CHSIMSLoader(str(input_dir))
    elif args.dataset == 'meld':
        dataset_loader = MELDLoader(str(input_dir))
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    meta_df = dataset_loader.load_metadata()
    video_groups = dataset_loader.get_video_groups(meta_df)
    
    print(f"找到 {len(video_groups)} 个视频")
    
    # ========== 加载模型 ==========
    
    print("\n" + "="*60)
    print("加载模型...")
    print("="*60)
    
    # 根据提取模式决定需要加载哪些模型
    need_basic_models = args.extract_mode in ['all', 'basic']
    need_text_model = True  # 总是需要文本模型（用于编码MLLM结果）
    need_mllm = args.extract_mode in ['all', 'social', 'context', 'prior'] and not args.skip_mllm
    
    clip_model = None
    clip_preprocess = None
    wavlm = None
    wavlm_feat = None
    
    if need_basic_models:
        # OpenCLIP
        print("[1/4] 加载 OpenCLIP...")
        try:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                args.clip_model, pretrained=None, device=args.device
            )
            sd = torch.load(args.clip_ckpt, map_location=args.device)
            clip_model.load_state_dict(sd, strict=True)
            clip_model.eval()
            print("  ✅ OpenCLIP 加载成功")
        except Exception as e:
            print(f"  ❌ OpenCLIP 加载失败: {e}")
            return
        
        # WavLM
        print("[2/4] 加载 WavLM...")
        try:
            wavlm_feat = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_path)
            wavlm = WavLMModel.from_pretrained(args.wavlm_path).to(args.device)
            wavlm.eval()
            print("  ✅ WavLM 加载成功")
        except Exception as e:
            print(f"  ❌ WavLM 加载失败: {e}")
            return
    else:
        print("[1-2/4] 跳过基本模态模型（OpenCLIP, WavLM）")
    
    # 文本模型
    if need_text_model:
        print("[3/4] 加载文本嵌入模型...")
        try:
            text_model = SentenceTransformer(args.text_model_path, device=args.device)
            print(f"  ✅ 文本模型加载成功，维度: {text_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"  ❌ 文本模型加载失败: {e}")
            return
    else:
        text_model = None
        print("[3/4] 跳过文本模型")
    
    # MLLM 并行处理器
    print("[4/4] 初始化 MLLM 并行处理器...")
    mllm_processor = None
    if need_mllm:
        try:
            mllm_processor = ParallelMLLMProcessor(
                api_key=args.api_key,
                base_url=args.base_url,
                model_name=args.model_name,
                text_model=text_model,
                num_workers=args.parallel_workers,
                temperature=args.temperature,
            )
            print(f"  ✅ MLLM 并行处理器初始化成功 (并行数: {args.parallel_workers})")
        except Exception as e:
            print(f"  ⚠️  MLLM 处理器初始化失败: {e}")
            print("  ⚠️  将跳过 MLLM 特征提取")
            args.skip_mllm = True
            mllm_processor = None
    else:
        mllm_processor = None
        print("  ⏭️  跳过 MLLM")
    
    # ========== 处理视频 ==========
    
    print("\n" + "="*60)
    print("开始处理视频...")
    print("="*60)
    print(f"总计: {len(video_groups)} 个视频")
    
    # 按顺序处理：先train，再test，最后valid
    def get_sort_key(video_id):
        if video_id.startswith('train_'):
            return (0, video_id)
        elif video_id.startswith('test_'):
            return (1, video_id)
        else:  # valid/dev
            return (2, video_id)
    
    sorted_video_items = sorted(video_groups.items(), key=lambda x: get_sort_key(x[0]))
    
    processed_videos = []
    failed_videos = []
    
    pbar_videos = tqdm(sorted_video_items, desc="总体进度", position=0)
    
    for video_id, video_utterances_df in pbar_videos:
        pbar_videos.set_description(f"总体进度 [{video_id}]")
        
        try:
            # 判断是否只更新先验特征
            if args.extract_mode in ['social', 'context', 'prior']:
                # 只更新先验特征
                success = update_prior_only(
                    video_id=video_id,
                    utterances_df=video_utterances_df,
                    dataset_loader=dataset_loader,
                    output_dir=output_dir,
                    text_model=text_model,
                    mllm_processor=mllm_processor,
                    args=args,
                )
                if success:
                    processed_videos.append(video_id)
                else:
                    failed_videos.append(video_id)
            else:
                # 完整处理或只处理基本模态
                video_data = process_video(
                    video_id=video_id,
                    utterances_df=video_utterances_df,
                    dataset_loader=dataset_loader,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    wavlm=wavlm,
                    wavlm_feat=wavlm_feat,
                    text_model=text_model,
                    mllm_processor=mllm_processor,
                    args=args,
                )
                
                if video_data is None:
                    failed_videos.append(video_id)
                    continue
                
                # 保存
                video_output_dir = output_dir / video_id
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = video_output_dir / 'utterances.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(video_data, f)
                
                processed_videos.append(video_id)
            
            # 更新进度条统计
            pbar_videos.set_postfix({
                'success': len(processed_videos), 
                'failed': len(failed_videos)
            })
        
        except Exception as e:
            print(f"\n❌ {video_id}: 处理失败")
            print(traceback.format_exc())
            failed_videos.append(video_id)
    
    pbar_videos.close()
    
    # ========== 生成数据划分 ==========
    
    print("\n" + "="*60)
    print("生成数据划分...")
    print("="*60)
    
    train_videos = []
    valid_videos = []
    test_videos = []
    
    for video_id in processed_videos:
        # 从video_id中提取split信息（MELD格式：train_dia_0）
        if video_id.startswith('train_'):
            train_videos.append(video_id)
        elif video_id.startswith('valid_') or video_id.startswith('dev_'):
            valid_videos.append(video_id)
        elif video_id.startswith('test_'):
            test_videos.append(video_id)
        else:
            # 对于其他数据集（CH-SIMS等），从meta_df查找
            video_meta = meta_df[meta_df['video_id'] == video_id]
            if len(video_meta) == 0:
                continue
            split = video_meta.iloc[0]['split']
            if split == 'train':
                train_videos.append(video_id)
            elif split == 'valid':
                valid_videos.append(video_id)
            elif split == 'test':
                test_videos.append(video_id)
    
    splits = {
        'train': sorted(train_videos),
        'valid': sorted(valid_videos),
        'test': sorted(test_videos),
    }
    
    split_file = output_dir / 'splits.pkl'
    with open(split_file, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"  Train: {len(splits['train'])} 个视频")
    print(f"  Valid: {len(splits['valid'])} 个视频")
    print(f"  Test:  {len(splits['test'])} 个视频")
    print(f"  保存到: {split_file}")
    
    # ========== 生成先验模态文本CSV ==========
    # 当提取了社会关系或情境模态时，生成CSV记录原始文本
    if args.extract_mode in ['all', 'social', 'context', 'prior']:
        print("\n" + "="*60)
        print("生成先验模态文本CSV...")
        print("="*60)
        
        # 按split收集数据
        prior_records = {'train': [], 'valid': [], 'test': []}
        
        for split_name, video_ids in splits.items():
            for video_id in video_ids:
                video_pkl = output_dir / video_id / 'utterances.pkl'
                if not video_pkl.exists():
                    continue
                
                try:
                    with open(video_pkl, 'rb') as f:
                        video_data = pickle.load(f)
                    
                    # 每个utterance有自己的social_text和context_text
                    utterance_ids = video_data.get('utterance_ids', [])
                    utterances = video_data.get('utterances', {})
                    
                    for utt_id in utterance_ids:
                        utt_data = utterances.get(utt_id, {})
                        prior_records[split_name].append({
                            'video_id': video_id,
                            'utterance_id': utt_id,
                            'social_text': utt_data.get('social_text', ''),
                            'context_text': utt_data.get('context_text', ''),
                        })
                except Exception as e:
                    print(f"  ⚠️  读取 {video_id} 失败: {e}")
                    continue
        
        # 保存CSV
        import csv
        csv_dir = output_dir / 'prior_texts'
        csv_dir.mkdir(exist_ok=True)
        
        for split_name, records in prior_records.items():
            if len(records) == 0:
                continue
            
            csv_path = csv_dir / f'{split_name}_prior_texts.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['video_id', 'utterance_id', 'social_text', 'context_text'])
                writer.writeheader()
                writer.writerows(records)
            
            print(f"  {split_name}: {len(records)} 条记录 -> {csv_path}")
        
        print(f"  先验模态文本已保存到: {csv_dir}")
    
    # ========== 总结 ==========
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"  成功: {len(processed_videos)} 个视频")
    print(f"  失败: {len(failed_videos)} 个视频")
    
    # 统计实际处理的utterance数量
    if args.extract_mode in ['all', 'basic'] and len(processed_videos) > 0:
        total_utterances_in_csv = len(meta_df)
        total_utterances_processed = 0
        for video_id in processed_videos[:min(10, len(processed_videos))]:
            video_pkl = output_dir / video_id / 'utterances.pkl'
            if video_pkl.exists():
                with open(video_pkl, 'rb') as f:
                    video_data = pickle.load(f)
                    total_utterances_processed += len(video_data['utterances'])
        
        avg_per_video = total_utterances_processed / min(10, len(processed_videos))
        estimated_total = int(avg_per_video * len(processed_videos))
        
        print(f"\nUtterance统计:")
        print(f"  CSV中记录: {total_utterances_in_csv} 条")
        print(f"  实际处理: ~{estimated_total} 个 (基于抽样估计)")
        print(f"  说明: MELD数据集CSV包含所有utterance，但部分视频文件缺失是正常的")
    
    if failed_videos:
        print(f"\n失败的视频:")
        for vid in failed_videos[:10]:
            print(f"  - {vid}")
        if len(failed_videos) > 10:
            print(f"  ... 还有 {len(failed_videos) - 10} 个")
    
    print(f"\n输出目录: {output_dir}")
    print("✅ 完成！")


if __name__ == '__main__':
    main()

