# -*- coding: utf-8 -*-
"""
统一数据集预处理脚本 - Utterance 级别特征版本
支持 CH-SIMSV2S, CH-SIMS, MELD 数据集
每个 utterance 提取一个特征向量（而非帧级别特征）
只提取三种基本模态：视觉、音频、文本

数据集结构：
1. CH-SIMSV2S / CH-SIMS:
   - Raw/video_xxxx/xxxx.mp4
   - meta.csv (包含所有标注和 train/valid/test 划分)

2. MELD:
   - train/train/dia{x}_utt{y}.mp4
   - dev/dev/dia{x}_utt{y}.mp4
   - test/test/dia{x}_utt{y}.mp4
   - 各自的 CSV 文件

目标格式（Utterance 级别）：
  output_dir/
    video_xxxx/ (或 dia_xxxx/)
      - utterances.pkl
        {
          'video_id': str,
          'utterance_ids': [str, ...],
          'utterances': {
            'utt_id': {
              'vision': [D_v],      # utterance 级别的单个向量
              'audio': [D_a],       # utterance 级别的单个向量
              'text': [D_t],        # utterance 级别的单个向量
              'label': float,
            },
            ...
          }
        }
    splits.pkl  # {'train': [...], 'valid': [...], 'test': [...]}
"""

import os
import sys
import pickle
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import cv2
import ffmpeg
from tqdm import tqdm
from PIL import Image

# ========== 导入依赖 ==========
import open_clip
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sentence_transformers import SentenceTransformer

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


def sample_frames_uniform(video_path: str, target_fps: float) -> Tuple[List[np.ndarray], float, int, int]:
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
    return frames, fps, total_frames, kept


def l2_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n


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
            
            # 检查列名
            required_cols = ['Dialogue_ID', 'Utterance_ID', 'Utterance', 'Sentiment']
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
            'Sentiment': 'label_text',  # 使用 Sentiment 作为标签
        })
        
        # 将 Sentiment 转换为数值
        # neutral=0, positive=1, negative=-1
        sentiment_map = {
            'neutral': 0.0,
            'positive': 1.0,
            'negative': -1.0
        }
        combined_df['label'] = combined_df['label_text'].map(sentiment_map)
        
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
        
        # MELD 路径: train/train/dia0_utt0.mp4
        split_dir_map = {'train': 'train', 'valid': 'dev', 'test': 'test'}
        split_dir_name = split_dir_map.get(split_prefix, split_prefix)
        
        video_file = f"dia{dia_num}_utt{utt_num}.mp4"
        return self.input_dir / split_dir_name / split_dir_name / video_file


# ========== Utterance 级别特征提取 ==========

def process_utterance_level(
    utterance_path: str,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    frame_rate: float,
    target_sr: int,
    batch_size_img: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个语句的视频，提取 utterance 级别的视觉和音频特征
    （对所有帧/音频片段进行平均池化，得到单个特征向量）
    
    Returns:
        (vision_feature, audio_feature)
        vision_feature: [D_v] - 单个视觉向量
        audio_feature: [D_a] - 单个音频向量
    """
    # ========== 视觉特征 ==========
    frames, fps, total_frames, kept = sample_frames_uniform(utterance_path, frame_rate)
    
    # 批量提取视觉特征
    img_emb_list = []
    with torch.no_grad():
        for i in range(0, kept, batch_size_img):
            batch_imgs = frames[i:i + batch_size_img]
            batch_tensors = [clip_preprocess(Image.fromarray(fr)) for fr in batch_imgs]
            batch = torch.stack(batch_tensors).to(device)
            feat = clip_model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_emb_list.append(feat.detach().cpu().numpy())
    
    img_emb = np.concatenate(img_emb_list, axis=0).astype(np.float32)  # [T, D_v]
    
    # 对所有帧进行平均池化，得到 utterance 级别特征
    vision_utterance = img_emb.mean(axis=0)  # [D_v]
    vision_utterance = l2_norm(vision_utterance.reshape(1, -1), axis=1).squeeze(0)  # L2 归一化
    
    # ========== 音频特征 ==========
    try:
        wav, sr = extract_audio_waveform(utterance_path, target_sr)
        
        wav_tensor = torch.from_numpy(wav.copy()).unsqueeze(0)
        a_in = wavlm_feat(
            wav_tensor.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            a_out = wavlm(a_in.input_values.to(device))
        a_emb = a_out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T_a, D_a]
        
        # 对所有音频帧进行平均池化，得到 utterance 级别特征
        audio_utterance = a_emb.mean(axis=0)  # [D_a]
        audio_utterance = l2_norm(audio_utterance.reshape(1, -1), axis=1).squeeze(0)  # L2 归一化
    
    except Exception as e:
        print(f"      ⚠️  音频提取失败: {e}")
        D_a = 1024  # WavLM-large 的特征维度
        audio_utterance = np.zeros(D_a, dtype=np.float32)
    
    return vision_utterance, audio_utterance


def process_video(
    video_id: str,
    utterances_df: pd.DataFrame,
    dataset_loader: DatasetLoader,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    text_model,
    args,
) -> Dict:
    """
    处理单个视频的所有语句（Utterance 级别特征）
    
    Returns:
        {
            'video_id': str,
            'utterance_ids': List[str],
            'utterances': Dict[utterance_id -> features],
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
    
    # ========== 为每个语句提取特征 ==========
    pbar_utterances = tqdm(utterances_df.iterrows(), total=len(utterances_df), 
                           desc=f"  处理语句", leave=False)
    
    skipped_files = []  # 记录跳过的文件
    
    for idx, row in pbar_utterances:
        utterance_id = row['utterance_id']
        text_content = row['text']
        label = float(row['label'])
        
        pbar_utterances.set_postfix({'utt_id': utterance_id})
        
        # 找到对应的视频文件
        utterance_file = dataset_loader.get_video_path(video_id, utterance_id, mode)
        
        if not utterance_file.exists():
            # CSV中有这条记录，但视频文件不存在
            print(f"  ⚠️  未找到语句文件: {utterance_file}")
            skipped_files.append(utterance_id)
            continue
        
        try:
            # 提取 utterance 级别的视觉和音频特征
            utterance_vision, utterance_audio = process_utterance_level(
                str(utterance_file),
                clip_model,
                clip_preprocess,
                wavlm,
                wavlm_feat,
                args.frame_rate,
                args.target_sr,
                args.batch_size_img,
                args.device,
            )
            
            # 文本特征（从 meta 读取并编码）
            text_emb = text_model.encode(text_content, convert_to_numpy=True, normalize_embeddings=True)
            text_emb = text_emb.astype(np.float32)
            
            utterance_data = {
                'vision': utterance_vision,  # [D_v] - utterance 级别
                'audio': utterance_audio,    # [D_a] - utterance 级别
                'text': text_emb,            # [D_t] - utterance 级别
                'label': label,
            }
            
            video_data['utterance_ids'].append(utterance_id)
            video_data['utterances'][utterance_id] = utterance_data
        
        except Exception as e:
            print(f"  ⚠️  处理语句 {utterance_id} 失败: {e}")
            traceback.print_exc()
            continue
    
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
    parser = argparse.ArgumentParser(description='统一数据集预处理脚本 - Utterance 级别特征')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['chsimsv2', 'chsims', 'meld'],
                        help='数据集名称')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
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
                        help='视频抽帧率（fps），用于采样帧数')
    parser.add_argument('--batch_size_img', type=int, default=32,
                        help='图像批处理大小')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='音频采样率')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    safe_makedirs(str(output_dir))
    
    # 显示提取模式
    print("\n" + "="*60)
    print(f"加载数据集: {args.dataset}")
    print(f"特征级别: Utterance 级别（单个向量）")
    print(f"提取模态: 视觉 + 音频 + 文本（三种基本模态）")
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
    
    # OpenCLIP
    print("[1/3] 加载 OpenCLIP...")
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
    print("[2/3] 加载 WavLM...")
    try:
        wavlm_feat = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_path)
        wavlm = WavLMModel.from_pretrained(args.wavlm_path).to(args.device)
        wavlm.eval()
        print("  ✅ WavLM 加载成功")
    except Exception as e:
        print(f"  ❌ WavLM 加载失败: {e}")
        return
    
    # 文本模型
    print("[3/3] 加载文本嵌入模型...")
    try:
        text_model = SentenceTransformer(args.text_model_path, device=args.device)
        print(f"  ✅ 文本模型加载成功，维度: {text_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"  ❌ 文本模型加载失败: {e}")
        return
    
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
            video_data = process_video(
                video_id=video_id,
                utterances_df=video_utterances_df,
                dataset_loader=dataset_loader,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                wavlm=wavlm,
                wavlm_feat=wavlm_feat,
                text_model=text_model,
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
    
    # ========== 总结 ==========
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"  成功: {len(processed_videos)} 个视频")
    print(f"  失败: {len(failed_videos)} 个视频")
    
    # 统计实际处理的utterance数量
    if len(processed_videos) > 0:
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
    
    if failed_videos:
        print(f"\n失败的视频:")
        for vid in failed_videos[:10]:
            print(f"  - {vid}")
        if len(failed_videos) > 10:
            print(f"  ... 还有 {len(failed_videos) - 10} 个")
    
    print(f"\n输出目录: {output_dir}")
    print("✅ 完成！")
    print("\n特征格式说明:")
    print("  - vision: [D_v] - utterance 级别的单个向量")
    print("  - audio:  [D_a] - utterance 级别的单个向量")
    print("  - text:   [D_t] - utterance 级别的单个向量")
    print("  (注: 不包含帧维度，每个 utterance 一个特征向量)")


if __name__ == '__main__':
    main()



