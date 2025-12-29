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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def worker_init_fn(worker_id):
    """
    为每个 DataLoader worker 设置独立但可重复的随机种子
    确保多进程数据加载的可重复性
    """
    # 使用 worker_id 和全局种子生成每个 worker 的唯一种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UtteranceLimitedBatchSampler:
    """
    动态 BatchSampler：限制每批的最大 utterance 数
    避免因对话长度差异导致的 OOM
    
    Args:
        dialogue_lengths: 每个对话的 utterance 数量列表
        max_utterances_per_batch: 每批最大 utterance 数
        shuffle: 是否打乱顺序
        drop_last: 是否丢弃不完整的最后一批
    """
    def __init__(
        self,
        dialogue_lengths: List[int],
        max_utterances_per_batch: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        weights: Optional[List[float]] = None,  # 新增：采样权重
    ):
        self.dialogue_lengths = dialogue_lengths
        self.max_utterances = max_utterances_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.weights = weights  # 存储权重
        self.epoch = 0
        
        # 预计算 batches
        self._create_batches()
    
    def _create_batches(self):
        """根据 utterance 数量限制创建 batches"""
        num_samples = len(self.dialogue_lengths)
        
        if self.shuffle:
            if self.weights is not None:
                # ====== 关键修改：基于权重的重采样 ======
                # 使用带放回采样，根据权重选择对话索引
                # 包含稀有样本的对话将被多次选中
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                
                # 转换为tensor
                weights_tensor = torch.tensor(self.weights, dtype=torch.float)
                # 采样数量保持与原数据集大小一致（或者可以根据需要调整）
                indices_tensor = torch.multinomial(
                    weights_tensor, 
                    num_samples, 
                    replacement=True, 
                    generator=g
                )
                indices = indices_tensor.tolist()
            else:
                # 原有的随机打乱
                indices = list(range(num_samples))
                rng = random.Random(self.seed + self.epoch)
                rng.shuffle(indices)
        else:
            indices = list(range(num_samples))
        
        self.batches = []
        current_batch = []
        current_utterances = 0
        
        for idx in indices:
            dialogue_len = self.dialogue_lengths[idx]
            
            # 如果单个对话就超过限制，单独成一批（避免被跳过）
            if dialogue_len > self.max_utterances:
                if current_batch:
                    self.batches.append(current_batch)
                self.batches.append([idx])
                current_batch = []
                current_utterances = 0
                continue
            
            # 如果加入这个对话会超过限制，先保存当前批
            if current_utterances + dialogue_len > self.max_utterances and current_batch:
                self.batches.append(current_batch)
                current_batch = []
                current_utterances = 0
            
            current_batch.append(idx)
            current_utterances += dialogue_len
        
        # 处理最后一批
        if current_batch:
            if not self.drop_last:
                self.batches.append(current_batch)
    
    def __iter__(self):
        # 每个 epoch 重新创建 batches（如果 shuffle）
        if self.shuffle:
            self._create_batches()
        
        for batch in self.batches:
            yield batch
        
        self.epoch += 1
    
    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """设置 epoch（用于分布式训练的同步）"""
        self.epoch = epoch


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
        self.labels = []  # 存储所有样本的标签，用于 WeightedRandomSampler
        
        for video_id in self.video_ids:
            video_pkl = self.data_dir / video_id / 'utterances.pkl'
            
            if not video_pkl.exists():
                print(f"Warning: {video_pkl} not found, skipping")
                continue
            
            with open(video_pkl, 'rb') as f:
                video_data = pickle.load(f)
                utterance_ids = video_data['utterance_ids']
                utterances = video_data['utterances']
            
            for utt_id in utterance_ids:
                self.index_map.append((video_id, utt_id))
                # 尝试获取标签
                if 'label' in utterances[utt_id]:
                    self.labels.append(utterances[utt_id]['label'])
                else:
                    self.labels.append(0) # 默认为0，防止出错
    
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
        
        # 单模态标签（CH-SIMSv2 MTL用）
        label_T = torch.FloatTensor([utterance.get('label_T', 0.0)])
        label_A = torch.FloatTensor([utterance.get('label_A', 0.0)])
        label_V = torch.FloatTensor([utterance.get('label_V', 0.0)])
        
        # 分离的social和context（现在是utterance级别）
        social = torch.FloatTensor(utterance.get('social', np.zeros(768, dtype=np.float32)))
        context = torch.FloatTensor(utterance.get('context', np.zeros(768, dtype=np.float32)))
        
        sample = {
            'audio': audio,
            'text': text_seq,
            'vision': vision,
            'label': label,
            'label_T': label_T,
            'label_A': label_A,
            'label_V': label_V,
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
    use_weighted_sampler: bool = False,  # 新增：是否使用加权随机采样
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
    
    sampler = None
    # ⭐ Weighted Random Sampler 逻辑
    if split == 'train' and use_weighted_sampler:
        # 获取所有标签
        try:
            labels = np.array(dataset.labels).astype(int)
            class_counts = np.bincount(labels)
            
            # 计算类别权重（数量越少，权重越大）
            # 添加平滑项防止除零，虽然通常不应该有0样本的类
            weights = 1.0 / (class_counts + 1e-6)
            
            # 为每个样本分配权重
            samples_weights = weights[labels]
            
            # 创建 Sampler
            # replacement=True 表示放回采样，这是过采样的关键
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(samples_weights),
                num_samples=len(samples_weights),
                replacement=True
            )
            
            # 使用 Sampler 时必须设置 shuffle=False
            shuffle = False
            print(f"[{split}] Enabled WeightedRandomSampler for class balancing")
            print(f"  Class counts: {class_counts}")
            print(f"  Class weights: {weights}")
            
        except Exception as e:
            print(f"Warning: Failed to create WeightedRandomSampler: {e}")
            print("Falling back to standard shuffling")
            sampler = None
    
    # 创建生成器确保 shuffle 可重复
    g = torch.Generator()
    g.manual_seed(42)  # 使用固定种子
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,  # 传入 sampler
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g if shuffle else None,
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
    use_weighted_sampler: bool = False,  # 新增
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
        use_weighted_sampler=use_weighted_sampler,  # 传递
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


# ==================== 对话级数据加载器（用于超图建模）====================

class DialogueLevelDataset(Dataset):
    """
    对话级数据集 - 用于超图建模
    每个样本是一个完整的对话（包含该对话的所有 utterance）
    
    支持：
    - CH-SIMSv2: 每个视频是一个对话
    - MELD: 每个 Dialogue_ID 是一个对话
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_length: int = 50,
        augment: bool = True,
        noise_scale: float = 0.0,
        max_dialogue_len: int = 50,  # 单个对话最大 utterance 数量
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_length = seq_length
        self.augment = augment
        self.noise_scale = noise_scale
        self.max_dialogue_len = max_dialogue_len
        
        # 加载数据划分
        self._load_splits()
        
        # 预加载对话信息（用于过滤过长对话）并计算采样权重
        self._build_dialogue_index()
        
        print(f"[{split}] DialogueLevel: {len(self.dialogue_ids)} dialogues, "
              f"max_len={max_dialogue_len}")
    
    def _load_splits(self):
        """加载数据划分"""
        splits_file = self.data_dir / 'splits.pkl'
        
        if not splits_file.exists():
            raise FileNotFoundError(f"splits.pkl not found in {self.data_dir}")
        
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        
        self.all_video_ids = splits[self.split]
        print(f"Loaded {len(self.all_video_ids)} videos for {self.split} split")
    
    def _build_dialogue_index(self):
        """建立对话索引，过滤过长的对话，并计算采样权重"""
        self.dialogue_ids = []
        self.dialogue_lengths = {}
        self.dialogue_weights = []  # 存储每个对话的采样权重
        
        # 统计全局正负样本分布（用于计算平衡因子）
        total_pos_utts = 0
        total_neg_utts = 0
        
        # 临时存储每个对话的统计信息
        dialogue_stats = []
        
        skipped = 0
        
        print("Building dialogue index and calculating weights...")
        for video_id in self.all_video_ids:
            video_pkl = self.data_dir / video_id / 'utterances.pkl'
            
            if not video_pkl.exists():
                skipped += 1
                continue
            
            with open(video_pkl, 'rb') as f:
                video_data = pickle.load(f)
                utterance_ids = video_data['utterance_ids']
                # 获取标签以计算分布
                utterances = video_data['utterances']
                
                # 统计该对话中的正负样本
                pos_count = 0
                neg_count = 0
                for uid in utterance_ids:
                    label = utterances[uid]['label']
                    if label > 0:
                        pos_count += 1
                    else:
                        neg_count += 1
            
            num_utterances = len(utterance_ids)
            
            # 过滤过长的对话
            if num_utterances > self.max_dialogue_len:
                skipped += 1
                continue
            
            # 跳过空对话
            if num_utterances == 0:
                skipped += 1
                continue
            
            self.dialogue_ids.append(video_id)
            self.dialogue_lengths[video_id] = num_utterances
            
            # 累加全局统计
            total_pos_utts += pos_count
            total_neg_utts += neg_count
            
            dialogue_stats.append({
                'pos': pos_count, 
                'neg': neg_count, 
                'total': num_utterances
            })
        
        # ====== 计算平衡权重 ======
        if self.split == 'train' and total_pos_utts > 0:
            # 计算正样本的提升系数 (Alpha)
            # 目标是让正样本的总权重接近负样本的总权重
            # Alpha = Neg_Total / Pos_Total
            pos_weight_factor = total_neg_utts / total_pos_utts
            print(f"  [Balance] Total Pos: {total_pos_utts}, Total Neg: {total_neg_utts}")
            print(f"  [Balance] Positive Weight Factor (Alpha): {pos_weight_factor:.2f}")
            
            # 为每个对话计算权重
            # Weight(Dialogue) = Sum(Weight(Utterance))
            # 其中 Weight(Pos_Utt) = Alpha, Weight(Neg_Utt) = 1.0
            for stats in dialogue_stats:
                w = stats['neg'] * 1.0 + stats['pos'] * pos_weight_factor
                self.dialogue_weights.append(w)
        else:
            # 非训练集或无正样本，权重均匀
            self.dialogue_weights = [1.0] * len(self.dialogue_ids)
            
        if skipped > 0:
            print(f"DialogueLevel: Skipped {skipped} dialogues (too long/empty/missing)")
    
    def __len__(self):
        return len(self.dialogue_ids)
    
    def __getitem__(self, idx):
        """返回一个完整对话的所有 utterance"""
        video_id = self.dialogue_ids[idx]
        
        video_pkl = self.data_dir / video_id / 'utterances.pkl'
        with open(video_pkl, 'rb') as f:
            video_data = pickle.load(f)
        
        utterance_ids = video_data['utterance_ids']
        utterances_data = video_data['utterances']
        
        # 收集该对话的所有 utterance
        audios, texts, visions, labels = [], [], [], []
        labels_T, labels_A, labels_V = [], [], []  # 单模态标签
        text_globals, socials, contexts = [], [], []
        utt_ids = []
        
        for utt_id in utterance_ids:
            utterance = utterances_data[utt_id]
            
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
            
            # 转换为 tensor
            audios.append(torch.FloatTensor(audio))
            texts.append(torch.FloatTensor(text_seq))
            visions.append(torch.FloatTensor(vision))
            text_globals.append(torch.FloatTensor(text_global))
            labels.append(torch.FloatTensor([utterance['label']]))
            
            # 单模态标签（CH-SIMSv2 MTL用）
            labels_T.append(torch.FloatTensor([utterance.get('label_T', 0.0)]))
            labels_A.append(torch.FloatTensor([utterance.get('label_A', 0.0)]))
            labels_V.append(torch.FloatTensor([utterance.get('label_V', 0.0)]))
            
            # social 和 context
            socials.append(torch.FloatTensor(
                utterance.get('social', np.zeros(768, dtype=np.float32))
            ))
            contexts.append(torch.FloatTensor(
                utterance.get('context', np.zeros(768, dtype=np.float32))
            ))
            utt_ids.append(utt_id)
        
        # Stack 成 [num_utterances, ...] 形状
        sample = {
            'audio': torch.stack(audios, dim=0),        # [N, T, D_audio]
            'text': torch.stack(texts, dim=0),          # [N, T, D_text]
            'vision': torch.stack(visions, dim=0),      # [N, T, D_vision]
            'label': torch.stack(labels, dim=0),        # [N, 1]
            'label_T': torch.stack(labels_T, dim=0),    # [N, 1]
            'label_A': torch.stack(labels_A, dim=0),    # [N, 1]
            'label_V': torch.stack(labels_V, dim=0),    # [N, 1]
            'text_global': torch.stack(text_globals, dim=0),  # [N, D_text]
            'social': torch.stack(socials, dim=0),      # [N, D_social]
            'context': torch.stack(contexts, dim=0),    # [N, D_context]
            'dialogue_len': len(utterance_ids),         # 标量：该对话的 utterance 数量
            'video_id': video_id,
            'utterance_ids': utt_ids,
        }
        
        return sample


def dialogue_collate_fn(batch: List[Dict]) -> Dict:
    """
    对话级 collate 函数
    将多个对话合并成一个 batch，并记录每个对话的长度
    
    Args:
        batch: list of dialogue samples, 每个 sample 是 __getitem__ 返回的 dict
    
    Returns:
        合并后的 batch dict，包含：
        - audio: [total_utterances, T, D_audio]
        - text: [total_utterances, T, D_text]
        - vision: [total_utterances, T, D_vision]
        - label: [total_utterances, 1]
        - text_global: [total_utterances, D_text]
        - social: [total_utterances, D_social]
        - context: [total_utterances, D_context]
        - batch_dia_len: List[int]，每个对话的 utterance 数量
        - video_ids: List[str]，每个对话的 ID
    """
    # 收集各个对话的数据
    audios, texts, visions, labels = [], [], [], []
    labels_T, labels_A, labels_V = [], [], []  # 单模态标签
    text_globals, socials, contexts = [], [], []
    batch_dia_len = []
    video_ids = []
    
    for sample in batch:
        audios.append(sample['audio'])          # [N_i, T, D]
        texts.append(sample['text'])
        visions.append(sample['vision'])
        labels.append(sample['label'])
        labels_T.append(sample.get('label_T', sample['label']))
        labels_A.append(sample.get('label_A', sample['label']))
        labels_V.append(sample.get('label_V', sample['label']))
        text_globals.append(sample['text_global'])
        socials.append(sample['social'])
        contexts.append(sample['context'])
        batch_dia_len.append(sample['dialogue_len'])
        video_ids.append(sample['video_id'])
    
    # Concat 所有对话的 utterance
    collated = {
        'audio': torch.cat(audios, dim=0),          # [sum(N_i), T, D]
        'text': torch.cat(texts, dim=0),
        'vision': torch.cat(visions, dim=0),
        'label': torch.cat(labels, dim=0),
        'label_T': torch.cat(labels_T, dim=0),
        'label_A': torch.cat(labels_A, dim=0),
        'label_V': torch.cat(labels_V, dim=0),
        'text_global': torch.cat(text_globals, dim=0),
        'social': torch.cat(socials, dim=0),
        'context': torch.cat(contexts, dim=0),
        'batch_dia_len': batch_dia_len,             # List[int]
        'video_ids': video_ids,
    }
    
    return collated


def create_dialogue_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = None,  # 已弃用，使用 max_utterances_per_batch
    num_workers: int = 4,
    seq_length: int = 50,
    augment: bool = True,
    noise_scale: float = 0.0,
    max_dialogue_len: int = 50,
    max_utterances_per_batch: int = 128,  # 新参数：每批最大 utterance 数
    shuffle: bool = None,
):
    """
    创建对话级数据加载器（用于超图建模）
    
    使用动态 BatchSampler 确保每批的总 utterance 数不超过 max_utterances_per_batch，
    避免因对话长度差异导致的 OOM。
    
    Args:
        max_utterances_per_batch: 每批最大 utterance 数（控制显存使用）
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = DialogueLevelDataset(
        data_dir=data_dir,
        split=split,
        seq_length=seq_length,
        augment=augment,
        noise_scale=noise_scale,
        max_dialogue_len=max_dialogue_len,
    )
    
    # 获取每个对话的长度，用于动态 batching
    dialogue_lengths = [dataset.dialogue_lengths[vid] for vid in dataset.dialogue_ids]
    
    # 仅在训练集且开启 shuffle 时使用权重
    weights = dataset.dialogue_weights if shuffle else None
    
    # 使用动态 BatchSampler 限制每批 utterance 数
    batch_sampler = UtteranceLimitedBatchSampler(
        dialogue_lengths=dialogue_lengths,
        max_utterances_per_batch=max_utterances_per_batch,
        shuffle=shuffle,
        drop_last=False,
        seed=42,
        weights=weights,  # 传入权重
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,  # 使用自定义 batch sampler
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dialogue_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    
    # 打印 batching 信息
    total_utterances = sum(dialogue_lengths)
    avg_batch_size = total_utterances / len(batch_sampler) if len(batch_sampler) > 0 else 0
    print(f"[{split}] UtteranceLimitedBatching: {len(batch_sampler)} batches, "
          f"max_utts={max_utterances_per_batch}, avg_utts={avg_batch_size:.1f}")
    
    return dataloader


def create_dialogue_dataloaders(
    data_dir: str,
    batch_size: int = None,  # 已弃用
    num_workers: int = 4,
    seq_length: int = 50,
    augment_train: bool = True,
    noise_scale: float = 0.0,
    max_dialogue_len: int = 50,
    max_utterances_per_batch: int = 128,  # 新参数：每批最大 utterance 数
):
    """
    创建对话级训练/验证/测试数据加载器
    
    Args:
        max_utterances_per_batch: 每批最大 utterance 数（控制显存使用，默认 128）
    """
    train_loader = create_dialogue_dataloader(
        data_dir=data_dir,
        split='train',
        num_workers=num_workers,
        seq_length=seq_length,
        augment=augment_train,
        noise_scale=noise_scale,
        max_dialogue_len=max_dialogue_len,
        max_utterances_per_batch=max_utterances_per_batch,
        shuffle=True,
    )
    
    valid_loader = create_dialogue_dataloader(
        data_dir=data_dir,
        split='valid',
        num_workers=num_workers,
        seq_length=seq_length,
        augment=False,
        noise_scale=0.0,
        max_dialogue_len=max_dialogue_len,
        max_utterances_per_batch=max_utterances_per_batch,
        shuffle=False,
    )
    
    test_loader = create_dialogue_dataloader(
        data_dir=data_dir,
        split='test',
        num_workers=num_workers,
        seq_length=seq_length,
        augment=False,
        noise_scale=0.0,
        max_dialogue_len=max_dialogue_len,
        max_utterances_per_batch=max_utterances_per_batch,
        shuffle=False,
    )
    
    return train_loader, valid_loader, test_loader
