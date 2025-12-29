# -*- coding: utf-8 -*-
"""
IEMOCAP 数据集预处理脚本（支持并行MLLM调用）

特点：
1. 支持并行调用MLLM API提取社会关系和情境特征
2. 使用Leave-One-Session-Out (LOSO) 进行数据划分
3. 4分类情感标签：neutral, happy(+excited), sad, angry
4. 输出格式与MELD/CHSIMSV2一致

数据结构：
输入: IEMOCAP_processed/
    - meta.csv
    - videos/Session{1-5}/*.mp4
    - audios/Session{1-5}/dialogue_id/*.wav

输出: iemocap_complete/iemocap/
    - dialogue_id/utterances.pkl
    - splits.pkl (LOSO 5折)
    - prior_texts/
"""

import os
import sys
import pickle
import argparse
import traceback
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

DEFAULT_SOCIAL_PROMPT = """Please describe the social relationships in the scene using concise natural language, including explicit relationship types (e.g., family members, friends, colleagues, etc.) as well as implicit relationship characteristics (e.g., power dynamics between characters, emotional intimacy, frequency of interaction, etc.). Output in a natural paragraph without bullet points, approximately 100 words. Please answer in English."""

DEFAULT_CONTEXT_PROMPT = """Please describe the background and atmosphere of the current situation using concise natural language, including: the time, place, type of event or activity, and the atmosphere of the communication (e.g., emotional tone of the conversation, degree of psychological tension, whether it is relaxed and casual or tense and formal, whether it has a ceremonial nature, etc.). Output in a natural paragraph without bullet points, approximately 100 words. Please answer in English."""

# IEMOCAP 情感标签映射（4分类）
EMOTION_MAP = {
    'neu': 0,  # neutral
    'hap': 1,  # happy
    'exc': 1,  # excited -> happy
    'sad': 2,  # sad
    'ang': 3,  # angry
    # 以下类别会被过滤掉
    'fru': -1,  # frustration (过滤)
    'xxx': -1,  # no agreement (过滤)
    'oth': -1,  # other (过滤)
    'sur': -1,  # surprise (过滤)
    'fea': -1,  # fear (过滤)
    'dis': -1,  # disgust (过滤)
}

EMOTION_NAMES = ['neutral', 'happy', 'sad', 'angry']

# ========== 工具函数 ==========

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def extract_audio_waveform(audio_path: str, sr: int) -> Tuple[np.ndarray, int]:
    """从音频文件提取波形"""
    try:
        out, _ = (
            ffmpeg.input(audio_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        return audio, sr
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg 解码失败: {e.stderr.decode(errors='ignore')}")


def extract_video_segment(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """从完整视频中提取指定时间段的片段"""
    try:
        duration = end_time - start_time
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac',
            '-loglevel', 'error',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        return False


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


# ========== 并行 MLLM 调用 ==========

class ParallelMLLMProcessor:
    """并行MLLM处理器 - 支持多线程并发调用API"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        text_model: SentenceTransformer,
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
    
    def process_batch(self, tasks: List[Dict[str, Any]], desc: str = "MLLM批处理") -> Dict[str, Dict[str, Any]]:
        """并行处理一批任务"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_utterance, task): task['utterance_id']
                for task in tasks
            }
            
            with tqdm(total=len(futures), desc=desc, leave=False) as pbar:
                for future in as_completed(futures):
                    utt_id = futures[future]
                    try:
                        result = future.result()
                        results[utt_id] = result
                    except Exception as e:
                        pass
                    pbar.update(1)
        
        return results


# ========== IEMOCAP 数据加载器 ==========

class IEMOCAPLoader:
    """IEMOCAP 数据加载器"""
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.meta_df = None
    
    def load_metadata(self) -> pd.DataFrame:
        """加载 meta.csv"""
        meta_path = self.input_dir / 'meta.csv'
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.csv not found in {self.input_dir}")
        
        df = pd.read_csv(meta_path)
        print(f"Loaded meta.csv: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # 映射情感标签
        df['label'] = df['emotion'].map(EMOTION_MAP)
        
        # 过滤掉无效标签
        before_len = len(df)
        df = df[df['label'] >= 0].copy()
        after_len = len(df)
        print(f"过滤后保留 {after_len}/{before_len} 个样本（4分类：neu/hap/exc/sad/ang）")
        
        # 统计标签分布
        print("\n情感标签分布（4分类）:")
        for label_idx, label_name in enumerate(EMOTION_NAMES):
            count = len(df[df['label'] == label_idx])
            print(f"  {label_idx}: {label_name} = {count}")
        
        self.meta_df = df
        return df
    
    def get_dialogue_groups(self, meta_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按 dialogue_id 分组"""
        groups = {}
        for dialogue_id, group in meta_df.groupby('dialogue_id'):
            group = group.sort_values('start_time')
            groups[dialogue_id] = group
        return groups
    
    def get_video_path(self, row: pd.Series) -> Path:
        """获取视频文件路径"""
        return self.input_dir / row['video_path']
    
    def get_audio_path(self, row: pd.Series) -> Path:
        """获取音频文件路径"""
        return self.input_dir / row['audio_path']
    
    def get_loso_splits(self, meta_df: pd.DataFrame, test_session: int) -> Dict[str, List[str]]:
        """生成Leave-One-Session-Out划分"""
        train_dialogues = []
        test_dialogues = []
        
        for dialogue_id in meta_df['dialogue_id'].unique():
            session = int(dialogue_id[3])  # Ses01F_impro01 -> 1
            if session == test_session:
                test_dialogues.append(dialogue_id)
            else:
                train_dialogues.append(dialogue_id)
        
        # 从训练集抽取10%作为验证集
        np.random.seed(42)
        np.random.shuffle(train_dialogues)
        valid_size = max(1, len(train_dialogues) // 10)
        valid_dialogues = train_dialogues[:valid_size]
        train_dialogues = train_dialogues[valid_size:]
        
        return {
            'train': sorted(train_dialogues),
            'valid': sorted(valid_dialogues),
            'test': sorted(test_dialogues),
        }


# ========== 特征提取 ==========

def process_utterance_features(
    audio_path: str,
    video_path: str,
    start_time: float,
    end_time: float,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    frame_rate: float,
    target_sr: int,
    batch_size_img: int,
    device: str,
    temp_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """提取单个utterance的视觉和音频特征"""
    
    # 音频特征
    D_a = 1024  # WavLM-large 维度
    try:
        wav, sr = extract_audio_waveform(audio_path, target_sr)
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
        
    except Exception as e:
        a_emb = np.zeros((1, D_a), dtype=np.float32)
        adur = end_time - start_time
    
    # 视频片段提取
    temp_video_path = os.path.join(temp_dir, f"temp_{Path(audio_path).stem}.mp4")
    video_extracted = extract_video_segment(video_path, start_time, end_time, temp_video_path)
    
    # ViT-g-14 维度
    D_v = 1024  # 与MELD保持一致（实际可能是1408，但项目中使用1024）
    
    if video_extracted and Path(temp_video_path).exists():
        try:
            frames, fps, total_frames, kept, idxs = sample_frames_uniform(temp_video_path, frame_rate)
            duration = total_frames / max(fps, 1e-6)
            frame_times = (idxs[:kept] + 0.5) / max(fps, 1e-6)
            
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
            img_emb = np.concatenate(img_emb_list, axis=0).astype(np.float32)
            
            # 对齐音频到帧
            a_pooled = pool_audio_to_frames(
                a_emb, audio_dur=adur, frame_times=frame_times,
                frame_fps=kept / max(duration, 1e-6), pad_sec=0.0
            )
            a_pooled = l2_norm(a_pooled, axis=1)
            
            return img_emb, a_pooled, temp_video_path
            
        except Exception as e:
            pass
    
    # 回退：返回零向量
    target_frames = max(1, int(round(adur * frame_rate)))
    img_emb = np.zeros((target_frames, D_v), dtype=np.float32)
    
    if len(a_emb) >= target_frames:
        a_pooled = l2_norm(a_emb[:target_frames], axis=1)
    else:
        a_pooled = np.zeros((target_frames, D_a), dtype=np.float32)
        a_pooled[:len(a_emb)] = l2_norm(a_emb, axis=1)
    
    return img_emb, a_pooled, temp_video_path if video_extracted else None


def process_dialogue(
    dialogue_id: str,
    utterances_df: pd.DataFrame,
    loader: IEMOCAPLoader,
    clip_model,
    clip_preprocess,
    wavlm,
    wavlm_feat,
    text_model,
    mllm_processor: Optional[ParallelMLLMProcessor],
    args,
) -> Optional[Dict]:
    """处理单个对话"""
    print(f"\n处理对话: {dialogue_id}")
    
    dialogue_data = {
        'video_id': dialogue_id,
        'utterance_ids': [],
        'utterances': {},
    }
    
    if len(utterances_df) == 0:
        return None
    
    print(f"  - {len(utterances_df)} 个语句")
    
    extract_basic = args.extract_mode in ['all', 'basic']
    extract_prior = args.extract_mode in ['all', 'social', 'context', 'prior']
    extract_social = args.extract_mode in ['all', 'social', 'prior']
    extract_context = args.extract_mode in ['all', 'context', 'prior']
    
    D_s = text_model.get_sentence_embedding_dimension()
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix=f"iemocap_{dialogue_id}_")
    mllm_tasks = []
    
    try:
        # ========== 第一阶段：提取基本模态特征 ==========
        pbar = tqdm(utterances_df.iterrows(), total=len(utterances_df), 
                    desc="  提取特征", leave=False)
        
        for idx, row in pbar:
            utterance_id = row['utterance_id']
            text_content = row['text']
            label = float(row['label'])
            start_time = row['start_time']
            end_time = row['end_time']
            
            audio_path = loader.get_audio_path(row)
            video_path = loader.get_video_path(row)
            
            # 生成utterance_id格式与MELD一致 (utt_0, utt_1, ...)
            utt_idx = len(dialogue_data['utterance_ids'])
            utt_key = f"utt_{utt_idx}"
            
            pbar.set_postfix({'utt': utt_key})
            
            if not audio_path.exists():
                continue
            
            try:
                utterance_data = {'label': label}
                temp_video_path = None
                
                if extract_basic:
                    vision, audio, temp_video_path = process_utterance_features(
                        str(audio_path),
                        str(video_path),
                        start_time,
                        end_time,
                        clip_model,
                        clip_preprocess,
                        wavlm,
                        wavlm_feat,
                        args.frame_rate,
                        args.target_sr,
                        args.batch_size_img,
                        args.device,
                        temp_dir,
                    )
                    
                    # 文本特征
                    text_emb = text_model.encode(str(text_content), convert_to_numpy=True, normalize_embeddings=True)
                    
                    utterance_data['vision'] = vision
                    utterance_data['audio'] = audio
                    utterance_data['text'] = text_emb.astype(np.float32)
                
                # 准备MLLM任务
                if extract_prior and mllm_processor:
                    if temp_video_path is None:
                        temp_video_path = os.path.join(temp_dir, f"temp_{utterance_id}.mp4")
                        extract_video_segment(str(video_path), start_time, end_time, temp_video_path)
                    
                    mllm_tasks.append({
                        'utterance_id': utt_key,
                        'video_path': temp_video_path,
                        'social_prompt': args.social_prompt,
                        'context_prompt': args.context_prompt,
                        'extract_social': extract_social,
                        'extract_context': extract_context,
                    })
                
                # 初始化先验特征
                if extract_prior:
                    utterance_data['social'] = np.zeros(D_s, dtype=np.float32)
                    utterance_data['social_text'] = ''
                    utterance_data['context'] = np.zeros(D_s, dtype=np.float32)
                    utterance_data['context_text'] = ''
                
                dialogue_data['utterance_ids'].append(utt_key)
                dialogue_data['utterances'][utt_key] = utterance_data
            
            except Exception as e:
                traceback.print_exc()
                continue
        
        # ========== 第二阶段：并行调用MLLM ==========
        if mllm_tasks and mllm_processor:
            print(f"  并行MLLM调用 ({len(mllm_tasks)} 任务, {mllm_processor.num_workers} 并行)")
            mllm_results = mllm_processor.process_batch(mllm_tasks, desc="  MLLM处理")
            
            for utt_id, result in mllm_results.items():
                if utt_id in dialogue_data['utterances']:
                    dialogue_data['utterances'][utt_id]['social'] = result['social_emb']
                    dialogue_data['utterances'][utt_id]['social_text'] = result['social_text']
                    dialogue_data['utterances'][utt_id]['context'] = result['context_emb']
                    dialogue_data['utterances'][utt_id]['context_text'] = result['context_text']
        
        if len(dialogue_data['utterances']) == 0:
            return None
        
        print(f"  ✅ 成功处理 {len(dialogue_data['utterances'])}/{len(utterances_df)} 个语句")
        return dialogue_data
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP 数据集预处理脚本（支持并行）')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='iemocap')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # 模式选择
    parser.add_argument('--extract_mode', type=str, default='all',
                        choices=['all', 'basic', 'social', 'context', 'prior'])
    parser.add_argument('--skip_mllm', action='store_true')
    
    # 模型路径
    parser.add_argument('--clip_model', type=str, default='ViT-g-14')
    parser.add_argument('--clip_ckpt', type=str, 
                        default='./models/open_clip/vit_g14_laion2b/open_clip_pytorch_model.bin')
    parser.add_argument('--wavlm_path', type=str, default='./models/wavlm/wavlm-large')
    parser.add_argument('--text_model_path', type=str, 
                        default='./models/sent/paraphrase-multilingual-mpnet-base-v2')
    
    # 处理参数
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--frame_rate', type=float, default=10.0)
    parser.add_argument('--batch_size_img', type=int, default=32)
    parser.add_argument('--target_sr', type=int, default=16000)
    
    # MLLM 参数
    parser.add_argument('--api_key', type=str, default='sk-CopXuPMUxmJY7UNSXrjyBA')
    parser.add_argument('--base_url', type=str, default='https://llm.rekeymed.com/v1/')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B')
    parser.add_argument('--temperature', type=float, default=0.7)
    
    # 并行参数
    parser.add_argument('--parallel_workers', type=int, default=4)
    
    # 提示词
    parser.add_argument('--social_prompt', type=str, default=DEFAULT_SOCIAL_PROMPT)
    parser.add_argument('--context_prompt', type=str, default=DEFAULT_CONTEXT_PROMPT)
    
    # LOSO 参数
    parser.add_argument('--test_session', type=int, default=5)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    # 输出目录结构与MELD一致: output_dir/iemocap/
    output_dir = Path(args.output_dir) / args.dataset
    safe_makedirs(str(output_dir))
    
    # ========== 显示配置 ==========
    print("\n" + "="*60)
    print("IEMOCAP 数据集预处理（支持并行MLLM）")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"提取模式: {args.extract_mode}")
    print(f"MLLM并行数: {args.parallel_workers}")
    print(f"测试Session: {args.test_session}")
    print("="*60)
    
    # ========== 加载数据 ==========
    loader = IEMOCAPLoader(str(input_dir))
    meta_df = loader.load_metadata()
    dialogue_groups = loader.get_dialogue_groups(meta_df)
    
    print(f"\n找到 {len(dialogue_groups)} 个对话")
    
    # ========== 加载模型 ==========
    print("\n" + "="*60)
    print("加载模型...")
    print("="*60)
    
    need_basic_models = args.extract_mode in ['all', 'basic']
    need_mllm = args.extract_mode in ['all', 'social', 'context', 'prior'] and not args.skip_mllm
    
    clip_model = None
    clip_preprocess = None
    wavlm = None
    wavlm_feat = None
    
    if need_basic_models:
        print("[1/4] 加载 OpenCLIP...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=None, device=args.device
        )
        sd = torch.load(args.clip_ckpt, map_location=args.device)
        clip_model.load_state_dict(sd, strict=True)
        clip_model.eval()
        print("  ✅ OpenCLIP 加载成功")
        
        print("[2/4] 加载 WavLM...")
        wavlm_feat = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_path)
        wavlm = WavLMModel.from_pretrained(args.wavlm_path).to(args.device)
        wavlm.eval()
        print("  ✅ WavLM 加载成功")
    else:
        print("[1-2/4] 跳过基本模态模型")
    
    print("[3/4] 加载文本嵌入模型...")
    text_model = SentenceTransformer(args.text_model_path, device=args.device)
    print(f"  ✅ 文本模型加载成功，维度: {text_model.get_sentence_embedding_dimension()}")
    
    print("[4/4] 初始化 MLLM 处理器...")
    mllm_processor = None
    if need_mllm:
        mllm_processor = ParallelMLLMProcessor(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            text_model=text_model,
            num_workers=args.parallel_workers,
            temperature=args.temperature,
        )
        print(f"  ✅ MLLM处理器初始化成功 (并行数: {args.parallel_workers})")
    else:
        print("  ⏭️  跳过 MLLM")
    
    # ========== 处理对话 ==========
    print("\n" + "="*60)
    print("开始处理对话...")
    print("="*60)
    
    processed_dialogues = []
    failed_dialogues = []
    
    pbar = tqdm(list(dialogue_groups.items()), desc="总体进度")
    
    for dialogue_id, utterances_df in pbar:
        pbar.set_description(f"处理 {dialogue_id}")
        
        try:
            dialogue_data = process_dialogue(
                dialogue_id=dialogue_id,
                utterances_df=utterances_df,
                loader=loader,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                wavlm=wavlm,
                wavlm_feat=wavlm_feat,
                text_model=text_model,
                mllm_processor=mllm_processor,
                args=args,
            )
            
            if dialogue_data is None:
                failed_dialogues.append(dialogue_id)
                continue
            
            # 保存
            dialogue_output_dir = output_dir / dialogue_id
            dialogue_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(dialogue_output_dir / 'utterances.pkl', 'wb') as f:
                pickle.dump(dialogue_data, f)
            
            processed_dialogues.append(dialogue_id)
            
            pbar.set_postfix({
                'success': len(processed_dialogues),
                'failed': len(failed_dialogues)
            })
        
        except Exception as e:
            print(f"\n❌ {dialogue_id}: 处理失败")
            traceback.print_exc()
            failed_dialogues.append(dialogue_id)
    
    # ========== 生成LOSO划分 ==========
    print("\n" + "="*60)
    print("生成LOSO数据划分...")
    print("="*60)
    
    processed_df = meta_df[meta_df['dialogue_id'].isin(processed_dialogues)]
    splits = loader.get_loso_splits(processed_df, args.test_session)
    
    for split_name in splits:
        splits[split_name] = [d for d in splits[split_name] if d in processed_dialogues]
    
    with open(output_dir / 'splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"  Train: {len(splits['train'])} 个对话")
    print(f"  Valid: {len(splits['valid'])} 个对话")
    print(f"  Test:  {len(splits['test'])} 个对话 (Session {args.test_session})")
    
    # ========== 生成先验文本CSV ==========
    if args.extract_mode in ['all', 'social', 'context', 'prior']:
        print("\n" + "="*60)
        print("生成先验模态文本CSV...")
        print("="*60)
        
        csv_dir = output_dir / 'prior_texts'
        csv_dir.mkdir(exist_ok=True)
        
        import csv
        prior_records = {'train': [], 'valid': [], 'test': []}
        
        for split_name, dialogue_ids in splits.items():
            for dialogue_id in dialogue_ids:
                pkl_path = output_dir / dialogue_id / 'utterances.pkl'
                if not pkl_path.exists():
                    continue
                
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                for utt_id in data.get('utterance_ids', []):
                    utt_data = data['utterances'].get(utt_id, {})
                    prior_records[split_name].append({
                        'video_id': dialogue_id,
                        'utterance_id': utt_id,
                        'social_text': utt_data.get('social_text', ''),
                        'context_text': utt_data.get('context_text', ''),
                    })
        
        for split_name, records in prior_records.items():
            if not records:
                continue
            csv_path = csv_dir / f'{split_name}_prior_texts.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['video_id', 'utterance_id', 'social_text', 'context_text'])
                writer.writeheader()
                writer.writerows(records)
            print(f"  {split_name}: {len(records)} 条 -> {csv_path}")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"  成功: {len(processed_dialogues)} 个对话")
    print(f"  失败: {len(failed_dialogues)} 个对话")
    print(f"\n输出目录: {output_dir}")
    print("✅ 完成！")


if __name__ == '__main__':
    main()
