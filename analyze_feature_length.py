# -*- coding: utf-8 -*-
"""
分析实际抽取好的特征文件中的序列长度
直接从 pkl 文件读取，获取真实的特征帧数
"""

import os
import pickle
import numpy as np
from collections import defaultdict

def analyze_dataset(data_dir, dataset_name):
    """分析单个数据集的特征长度"""
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"路径: {data_dir}")
    print('='*60)
    
    lengths = {
        'video': [],
        'audio': [],
    }
    
    sample_count = 0
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == 'utterances.pkl':
                pkl_path = os.path.join(root, f)
                try:
                    with open(pkl_path, 'rb') as pf:
                        data = pickle.load(pf)
                    
                    # 根据实际数据结构解析
                    if isinstance(data, dict) and 'utterances' in data:
                        utterances_dict = data['utterances']
                        if isinstance(utterances_dict, dict):
                            for utt_id, utt in utterances_dict.items():
                                if isinstance(utt, dict):
                                    sample_count += 1
                                    
                                    # vision特征
                                    if 'vision' in utt and utt['vision'] is not None:
                                        feat = utt['vision']
                                        if isinstance(feat, np.ndarray) and len(feat.shape) >= 1:
                                            lengths['video'].append(feat.shape[0])
                                    
                                    # audio特征
                                    if 'audio' in utt and utt['audio'] is not None:
                                        feat = utt['audio']
                                        if isinstance(feat, np.ndarray) and len(feat.shape) >= 1:
                                            lengths['audio'].append(feat.shape[0])
                    
                except Exception as e:
                    print(f"Error processing {pkl_path}: {e}")
    
    print(f"\n总样本数: {sample_count}")
    return lengths

def print_statistics(lengths, modality_name, seq_length=50):
    """打印统计信息并分析与当前seq_length的关系"""
    if not lengths:
        print(f"  {modality_name}: 无数据")
        return None
    
    arr = np.array(lengths)
    
    # 基础统计
    stats = {
        'count': len(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'p25': np.percentile(arr, 25),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
        'raw': arr,  # 保存原始数据
    }
    
    # 与seq_length=50的对比
    over_50 = np.sum(arr > seq_length)
    over_50_pct = over_50 / len(arr) * 100
    under_50 = np.sum(arr < seq_length)
    under_50_pct = under_50 / len(arr) * 100
    equal_50 = np.sum(arr == seq_length)
    
    # 计算padding和截断的浪费
    padding_waste = np.sum(np.maximum(0, seq_length - arr))  # 总padding帧数
    truncation_loss = np.sum(np.maximum(0, arr - seq_length))  # 总截断帧数
    avg_truncation = truncation_loss / over_50 if over_50 > 0 else 0
    
    print(f"\n  {modality_name}特征长度统计:")
    print(f"    样本数量: {stats['count']}")
    print(f"    最小值: {stats['min']:.0f}")
    print(f"    最大值: {stats['max']:.0f}")
    print(f"    平均值: {stats['mean']:.2f}")
    print(f"    中位数: {stats['median']:.0f}")
    print(f"    标准差: {stats['std']:.2f}")
    print(f"    25%分位: {stats['p25']:.0f}")
    print(f"    75%分位: {stats['p75']:.0f}")
    print(f"    90%分位: {stats['p90']:.0f}")
    print(f"    95%分位: {stats['p95']:.0f}")
    print(f"    99%分位: {stats['p99']:.0f}")
    
    print(f"\n  与seq_length={seq_length}的对比:")
    print(f"    帧数 > {seq_length} 的样本: {over_50} ({over_50_pct:.1f}%) → 被截断")
    print(f"    帧数 < {seq_length} 的样本: {under_50} ({under_50_pct:.1f}%) → 需padding")
    print(f"    帧数 = {seq_length} 的样本: {equal_50} ({equal_50/len(arr)*100:.1f}%)")
    print(f"    总截断帧数: {truncation_loss:.0f} (平均每个被截断样本丢失 {avg_truncation:.1f} 帧)")
    print(f"    总padding帧数: {padding_waste:.0f}")
    
    return stats

def main():
    base_dir = '/home/kemove/wsy2'
    current_seq_length = 50
    
    all_results = {}
    
    # 1. 分析 meld_10
    meld_10_dir = os.path.join(base_dir, 'meld_10/meld')
    if os.path.exists(meld_10_dir):
        lengths = analyze_dataset(meld_10_dir, 'MELD (10fps)')
        all_results['meld'] = {}
        for modality, lens in lengths.items():
            stats = print_statistics(lens, modality, current_seq_length)
            if stats:
                all_results['meld'][modality] = stats
    
    # 2. 分析 chsimsv2_10
    chsimsv2_10_dir = os.path.join(base_dir, 'chsimsv2_10/chsimsv2_processed')
    if os.path.exists(chsimsv2_10_dir):
        lengths = analyze_dataset(chsimsv2_10_dir, 'CH-SIMSv2 (10fps)')
        all_results['chsimsv2'] = {}
        for modality, lens in lengths.items():
            stats = print_statistics(lens, modality, current_seq_length)
            if stats:
                all_results['chsimsv2'][modality] = stats
    
    # 3. 总结与建议
    print("\n" + "="*60)
    print("📊 总结与seq_length建议")
    print("="*60)
    
    print("\n" + "-"*60)
    print(f"当前设置: seq_length = {current_seq_length}")
    print("-"*60)
    
    for dataset, modalities in all_results.items():
        if 'video' in modalities:
            stats = modalities['video']
            raw = stats['raw']
            
            print(f"\n【{dataset.upper()}】视频特征:")
            print(f"  平均帧数: {stats['mean']:.0f}")
            print(f"  95%分位: {stats['p95']:.0f}")
            print(f"  99%分位: {stats['p99']:.0f}")
            
            # 计算最佳seq_length建议
            rec_95 = int(np.ceil(stats['p95'] / 10) * 10)  # 向上取整到10的倍数
            rec_99 = int(np.ceil(stats['p99'] / 10) * 10)
            
            print(f"  建议seq_length: {rec_95} (覆盖95%) 或 {rec_99} (覆盖99%)")
    
    print("\n" + "="*60)
    print("🔍 结论：原来seq_length=50是否导致问题？")
    print("="*60)
    
    for dataset, modalities in all_results.items():
        if 'video' in modalities:
            stats = modalities['video']
            raw = stats['raw']
            
            over_50_pct = np.sum(raw > 50) / len(raw) * 100
            truncation_loss = np.sum(np.maximum(0, raw - 50))
            total_frames = np.sum(raw)
            loss_pct = truncation_loss / total_frames * 100
            
            print(f"\n【{dataset.upper()}】:")
            if over_50_pct < 5:
                print(f"  ✅ seq_length=50 基本足够")
                print(f"     仅 {over_50_pct:.1f}% 的样本被截断")
            elif over_50_pct < 20:
                print(f"  ⚠️  seq_length=50 会导致部分信息丢失")
                print(f"     {over_50_pct:.1f}% 的样本被截断")
                print(f"     总帧数的 {loss_pct:.1f}% 被丢弃")
                print(f"     建议提高到 {int(np.ceil(stats['p95']/10)*10)}")
            else:
                print(f"  ❌ seq_length=50 严重不足！")
                print(f"     {over_50_pct:.1f}% 的样本被截断")
                print(f"     总帧数的 {loss_pct:.1f}% 被丢弃")
                print(f"     建议提高到 {int(np.ceil(stats['p95']/10)*10)}")
            
            # 分析是否过度padding
            under_50_pct = np.sum(raw < 50) / len(raw) * 100
            padding_waste = np.sum(np.maximum(0, 50 - raw))
            if stats['mean'] < 40:
                print(f"  ⚠️  同时存在过度padding：平均帧数仅{stats['mean']:.0f}，但padding到50")

if __name__ == '__main__':
    main()
