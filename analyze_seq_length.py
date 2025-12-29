# -*- coding: utf-8 -*-
"""
分析各数据集的视频特征长度，以确定合适的seq_length参数
"""

import os
import pickle
import numpy as np
from collections import defaultdict
import subprocess
import glob

def get_video_duration(video_path):
    """使用ffprobe获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return None

def analyze_processed_data(data_dir, dataset_name):
    """分析处理后的数据集特征长度"""
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"路径: {data_dir}")
    print('='*60)
    
    lengths = {
        'text': [],
        'audio': [],
        'video': []
    }
    
    # 遍历所有子目录
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        pkl_path = os.path.join(subdir_path, 'utterances.pkl')
        if not os.path.exists(pkl_path):
            continue
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # data可能是列表或字典
            if isinstance(data, list):
                utterances = data
            elif isinstance(data, dict):
                utterances = list(data.values()) if not isinstance(list(data.values())[0], (list, dict)) else data.get('utterances', [data])
            else:
                continue
            
            for utt in utterances:
                if isinstance(utt, dict):
                    for key in ['text', 'audio', 'video', 'text_feat', 'audio_feat', 'video_feat']:
                        feat = utt.get(key)
                        if feat is not None:
                            if isinstance(feat, np.ndarray):
                                if len(feat.shape) >= 1:
                                    base_key = key.replace('_feat', '')
                                    if base_key in lengths:
                                        lengths[base_key].append(feat.shape[0])
                            elif isinstance(feat, (list, tuple)):
                                base_key = key.replace('_feat', '')
                                if base_key in lengths:
                                    lengths[base_key].append(len(feat))
        except Exception as e:
            pass  # 静默处理错误
    
    return lengths

def analyze_raw_videos(video_dirs, dataset_name):
    """分析原始视频时长"""
    print(f"\n{'='*60}")
    print(f"分析原始视频: {dataset_name}")
    print('='*60)
    
    durations = []
    video_count = 0
    
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            print(f"目录不存在: {video_dir}")
            continue
            
        for ext in ['*.mp4', '*.avi', '*.mkv']:
            for video_path in glob.glob(os.path.join(video_dir, '**', ext), recursive=True):
                duration = get_video_duration(video_path)
                if duration is not None:
                    durations.append(duration)
                    video_count += 1
                    if video_count % 100 == 0:
                        print(f"  已处理 {video_count} 个视频...")
                
                # 限制采样数量以加快分析
                if video_count >= 500:
                    print(f"  达到采样上限 (500)，停止采样")
                    break
            if video_count >= 500:
                break
        if video_count >= 500:
            break
    
    return durations

def print_statistics(lengths, modality_name):
    """打印统计信息"""
    if not lengths:
        print(f"  {modality_name}: 无数据")
        return
    
    arr = np.array(lengths)
    print(f"\n  {modality_name}特征长度统计:")
    print(f"    样本数量: {len(arr)}")
    print(f"    最小值: {np.min(arr):.2f}")
    print(f"    最大值: {np.max(arr):.2f}")
    print(f"    平均值: {np.mean(arr):.2f}")
    print(f"    中位数: {np.median(arr):.2f}")
    print(f"    标准差: {np.std(arr):.2f}")
    print(f"    25%分位: {np.percentile(arr, 25):.2f}")
    print(f"    75%分位: {np.percentile(arr, 75):.2f}")
    print(f"    90%分位: {np.percentile(arr, 90):.2f}")
    print(f"    95%分位: {np.percentile(arr, 95):.2f}")
    print(f"    99%分位: {np.percentile(arr, 99):.2f}")

def analyze_iemocap_from_meta(meta_path):
    """从IEMOCAP元数据分析utterance时长"""
    print(f"\n{'='*60}")
    print(f"分析IEMOCAP元数据")
    print('='*60)
    
    import csv
    
    durations = []
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_time = float(row['start_time'])
                end_time = float(row['end_time'])
                durations.append(end_time - start_time)
            except:
                pass
    
    durations = np.array(durations)
    
    print(f"\n  Utterance时长统计（秒）:")
    print(f"    样本数量: {len(durations)}")
    print(f"    最小值: {np.min(durations):.2f}")
    print(f"    最大值: {np.max(durations):.2f}")
    print(f"    平均值: {np.mean(durations):.2f}")
    print(f"    中位数: {np.median(durations):.2f}")
    print(f"    标准差: {np.std(durations):.2f}")
    print(f"    25%分位: {np.percentile(durations, 25):.2f}")
    print(f"    75%分位: {np.percentile(durations, 75):.2f}")
    print(f"    90%分位: {np.percentile(durations, 90):.2f}")
    print(f"    95%分位: {np.percentile(durations, 95):.2f}")
    print(f"    99%分位: {np.percentile(durations, 99):.2f}")
    
    # 按不同采样率计算帧数
    for fps in [3, 10]:
        frame_counts = durations * fps
        print(f"\n  按{fps}fps采样的帧数统计:")
        print(f"    最小值: {np.min(frame_counts):.2f}")
        print(f"    最大值: {np.max(frame_counts):.2f}")
        print(f"    平均值: {np.mean(frame_counts):.2f}")
        print(f"    中位数: {np.median(frame_counts):.2f}")
        print(f"    90%分位: {np.percentile(frame_counts, 90):.2f}")
        print(f"    95%分位: {np.percentile(frame_counts, 95):.2f}")
        print(f"    99%分位: {np.percentile(frame_counts, 99):.2f}")
    
    return durations

def main():
    base_dir = '/home/kemove/wsy2'
    
    # 统计结果汇总
    all_stats = {}
    
    # 1. 分析CH-SIMSv2处理后的数据
    chsimsv2_data_dir = os.path.join(base_dir, 'data/ch-simsv2s_processed')
    if os.path.exists(chsimsv2_data_dir):
        lengths = analyze_processed_data(chsimsv2_data_dir, 'CH-SIMSv2')
        for modality, lens in lengths.items():
            print_statistics(lens, modality)
        all_stats['CH-SIMSv2'] = lengths
    
    # 2. 分析MELD处理后的数据
    meld_data_dir = os.path.join(base_dir, 'data/meld')
    if os.path.exists(meld_data_dir):
        lengths = analyze_processed_data(meld_data_dir, 'MELD')
        for modality, lens in lengths.items():
            print_statistics(lens, modality)
        all_stats['MELD'] = lengths
    
    # 3. 分析MELD原始视频时长
    print("\n" + "="*60)
    print("分析MELD原始视频时长")
    print("="*60)
    meld_video_dirs = [
        os.path.join(base_dir, 'MELD/train'),
        os.path.join(base_dir, 'MELD/dev'),
        os.path.join(base_dir, 'MELD/test')
    ]
    meld_durations = analyze_raw_videos(meld_video_dirs, 'MELD')
    if meld_durations:
        print_statistics(meld_durations, "视频时长（秒）")
        # 按不同fps采样
        for fps in [3, 10]:
            frame_counts = [d * fps for d in meld_durations]
            print_statistics(frame_counts, f"按{fps}fps采样的帧数")
    
    # 4. 分析IEMOCAP
    iemocap_meta = os.path.join(base_dir, 'IEMOCAP_processed/meta.csv')
    if os.path.exists(iemocap_meta):
        iemocap_durations = analyze_iemocap_from_meta(iemocap_meta)
    
    # 5. 分析CH-SIMSv2原始视频
    print("\n" + "="*60)
    print("分析CH-SIMSv2原始视频时长")
    print("="*60)
    chsimsv2_raw_dir = os.path.join(base_dir, 'chsimsv2/ch-simsv2s/Raw')
    if os.path.exists(chsimsv2_raw_dir):
        chsimsv2_durations = analyze_raw_videos([chsimsv2_raw_dir], 'CH-SIMSv2')
        if chsimsv2_durations:
            print_statistics(chsimsv2_durations, "视频时长（秒）")
            for fps in [3, 10]:
                frame_counts = [d * fps for d in chsimsv2_durations]
                print_statistics(frame_counts, f"按{fps}fps采样的帧数")
    
    # 6. 总结与建议
    print("\n" + "="*60)
    print("总结与seq_length建议")
    print("="*60)
    
    # 计算各数据集的建议值
    print("\n" + "-"*60)
    print("按10fps采样时的建议seq_length（覆盖95%样本）:")
    print("-"*60)
    
    # MELD
    if meld_durations:
        meld_95 = np.percentile(meld_durations, 95) * 10
        meld_99 = np.percentile(meld_durations, 99) * 10
        meld_avg = np.mean(meld_durations) * 10
        print(f"  MELD:     平均={meld_avg:.0f}帧, 95%={meld_95:.0f}帧, 99%={meld_99:.0f}帧")
        print(f"            建议seq_length: {int(np.ceil(meld_95/10)*10)} (覆盖95%) 或 {int(np.ceil(meld_99/10)*10)} (覆盖99%)")
    
    # IEMOCAP
    if 'iemocap_durations' in dir() and len(iemocap_durations) > 0:
        iemocap_95 = np.percentile(iemocap_durations, 95) * 10
        iemocap_99 = np.percentile(iemocap_durations, 99) * 10
        iemocap_avg = np.mean(iemocap_durations) * 10
        print(f"  IEMOCAP:  平均={iemocap_avg:.0f}帧, 95%={iemocap_95:.0f}帧, 99%={iemocap_99:.0f}帧")
        print(f"            建议seq_length: {int(np.ceil(iemocap_95/10)*10)} (覆盖95%) 或 {int(np.ceil(iemocap_99/10)*10)} (覆盖99%)")
    
    # CH-SIMSv2
    if 'chsimsv2_durations' in dir() and len(chsimsv2_durations) > 0:
        chsimsv2_95 = np.percentile(chsimsv2_durations, 95) * 10
        chsimsv2_99 = np.percentile(chsimsv2_durations, 99) * 10
        chsimsv2_avg = np.mean(chsimsv2_durations) * 10
        print(f"  CH-SIMSv2: 平均={chsimsv2_avg:.0f}帧, 95%={chsimsv2_95:.0f}帧, 99%={chsimsv2_99:.0f}帧")
        print(f"            建议seq_length: {int(np.ceil(chsimsv2_95/10)*10)} (覆盖95%) 或 {int(np.ceil(chsimsv2_99/10)*10)} (覆盖99%)")
    
    print("\n" + "-"*60)
    print("与当前seq_length=50的对比分析 (10fps):")
    print("-"*60)
    
    if meld_durations:
        over_50_pct = np.mean(np.array(meld_durations) * 10 > 50) * 100
        print(f"  MELD:     {over_50_pct:.1f}% 的样本帧数>50，会被截断丢失信息")
    
    if 'iemocap_durations' in dir() and len(iemocap_durations) > 0:
        over_50_pct = np.mean(np.array(iemocap_durations) * 10 > 50) * 100
        print(f"  IEMOCAP:  {over_50_pct:.1f}% 的样本帧数>50，会被截断丢失信息")
    
    if 'chsimsv2_durations' in dir() and len(chsimsv2_durations) > 0:
        over_50_pct = np.mean(np.array(chsimsv2_durations) * 10 > 50) * 100
        print(f"  CH-SIMSv2: {over_50_pct:.1f}% 的样本帧数>50，会被截断丢失信息")
    
    print("""
结论：
1. 当前seq_length=50在10fps采样下会导致大量样本被截断
2. 建议根据数据集特点单独设置seq_length
3. 权衡：增大seq_length会增加计算量和显存占用
""")

if __name__ == '__main__':
    main()

