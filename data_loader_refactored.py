# -*- coding: utf-8 -*-
"""
数据加载模块 - 重构版
支持分离的social和context特征（utterance级别）
"""

import numpy as np
import pickle
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader


def sampling(feature, length, d):
    """对特征进行时间维度的重采样"""
    new_feature = np.zeros((length, d))
    feature_len = feature.shape[0]
    sample_index = np.linspace(0, feature_len, length + 1, dtype=np.uint16)
    
    for i in range(len(sample_index) - 1):
        if sample_index[i] == sample_index[i + 1]:
            new_feature[i, :] = feature[sample_index[i], :]
        else:
            new_feature[i, :] = feature[sample_index[i]:sample_index[i + 1], :].mean(0)
    
    return new_feature


class VideoLevelDataset_Refactored(Dataset):
    """
    视频级数据集 - 重构版
    支持分离的social和context特征
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_length: int = 50,
        augment: bool = True,
        noise_scale: float = 0.0,
        cache_size: int = 20,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_length = seq_length
        self.augment = augment
        self.noise_scale = noise_scale
        self.cache_size = cache_size
        
        # 加载数据划分
        self._load_splits()
        
        # 建立索引
        self._build_index()
        
        # LRU缓存
        self.cache = OrderedDict()
        
        print(f"[{split}] Loaded {len(self.index_map)} utterances from {len(self.video_ids)} videos")
    
    def _load_splits(self):
        """加载数据划分"""
        splits_file = self.data_dir / 'splits.pkl'
        
        if not splits_file.exists():
            raise FileNotFoundError(f"splits.pkl not found in {self.data_dir}")
        
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        
        self.video_ids = splits[self.split]
        print(f"Loaded {len(self.video_ids)} videos for {self.split} split")
    
    def _build_index(self):
        """建立全局索引：idx -> (video_id, utterance_id)"""
        self.index_map = []
        
        for video_id in self.video_ids:
            video_pkl = self.data_dir / video_id / 'utterances.pkl'
            
            if not video_pkl.exists():
                print(f"Warning: {video_pkl} not found, skipping")
                continue
            
            with open(video_pkl, 'rb') as f:
                video_data = pickle.load(f)
                utterance_ids = video_data['utterance_ids']
            
            for utt_id in utterance_ids:
                self.index_map.append((video_id, utt_id))
    
    def _load_video_data(self, video_id: str) -> Dict:
        """加载视频数据（带缓存）"""
        if video_id in self.cache:
            self.cache.move_to_end(video_id)
            return self.cache[video_id]
        
        video_pkl = self.data_dir / video_id / 'utterances.pkl'
        with open(video_pkl, 'rb') as f:
            video_data = pickle.load(f)
        
        self.cache[video_id] = video_data
        
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return video_data
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        video_id, utterance_id = self.index_map[idx]
        
        video_data = self._load_video_data(video_id)
        utterance = video_data['utterances'][utterance_id]
        
        # 采样到固定长度
        vision = sampling(utterance['vision'], self.seq_length, utterance['vision'].shape[1])
        audio = sampling(utterance['audio'], self.seq_length, utterance['audio'].shape[1])
        
        # 文本：扩展到序列长度
        text_global = utterance['text']
        text_seq = np.tile(text_global, (self.seq_length, 1))
        
        # 数据增强
        if self.augment and self.noise_scale > 0:
            vision = vision + np.random.normal(0, self.noise_scale, vision.shape)
            audio = audio + np.random.normal(0, self.noise_scale, audio.shape)
        
        # 转换为tensor
        vision = torch.FloatTensor(vision)
        audio = torch.FloatTensor(audio)
        text_seq = torch.FloatTensor(text_seq)
        text_global = torch.FloatTensor(text_global)
        label = torch.FloatTensor([utterance['label']])
        
        # 分离的social和context（现在是utterance级别）
        social = torch.FloatTensor(utterance.get('social', np.zeros(768, dtype=np.float32)))
        context = torch.FloatTensor(utterance.get('context', np.zeros(768, dtype=np.float32)))
        
        sample = {
            'audio': audio,
            'text': text_seq,
            'vision': vision,
            'label': label,
            'text_global': text_global,
            'social': social,
            'context': context,
            'id': f"{video_id}_{utterance_id}",
            'video_id': video_id,
            'utterance_id': utterance_id,
        }
        
        return sample


def create_dataloader_refactored(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    seq_length: int = 50,
    augment: bool = True,
    noise_scale: float = 0.0,
    cache_size: int = 20,
    shuffle: bool = None,
):
    """创建数据加载器"""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = VideoLevelDataset_Refactored(
        data_dir=data_dir,
        split=split,
        seq_length=seq_length,
        augment=augment,
        noise_scale=noise_scale,
        cache_size=cache_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def create_dataloaders_refactored(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    seq_length: int = 50,
    augment_train: bool = True,
    noise_scale: float = 0.0,
    cache_size: int = 20,
):
    """创建训练/验证/测试数据加载器"""
    train_loader = create_dataloader_refactored(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        seq_length=seq_length,
        augment=augment_train,
        noise_scale=noise_scale,
        cache_size=cache_size,
        shuffle=True,
    )
    
    valid_loader = create_dataloader_refactored(
        data_dir=data_dir,
        split='valid',
        batch_size=batch_size,
        num_workers=num_workers,
        seq_length=seq_length,
        augment=False,
        noise_scale=0.0,
        cache_size=cache_size,
        shuffle=False,
    )
    
    test_loader = create_dataloader_refactored(
        data_dir=data_dir,
        split='test',
        batch_size=batch_size,
        num_workers=num_workers,
        seq_length=seq_length,
        augment=False,
        noise_scale=0.0,
        cache_size=cache_size,
        shuffle=False,
    )
    
    return train_loader, valid_loader, test_loader

