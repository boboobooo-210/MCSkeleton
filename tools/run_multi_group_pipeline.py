#!/usr/bin/env python3
"""
多分组骨架流水线 - GIF动画生成器
完全借鉴skeleton_extraction_reconstruction_pipeline.py的可视化方法
生成连续帧序列的GIF动画

骨架提取器: OptimizedMARSModel (极简优化版)
- 模型路径: models/skeleton_extractor_final.py
- 默认权重: mars_optimized_best.pth
- 架构特点: SpatialPreservingBackbone + SimplifiedRegressionHead
- 损失函数: MSE(0.7) + L1(0.3), 无复杂约束
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.multi_group_skeleton_pipeline import create_pipeline, GROUPING_CONFIGS

# MARS数据集映射关节常量
FINGER_JOINTS = {7, 21, 22, 11, 23, 24}  # 左手指: 7,21,22  右手指: 11,23,24

# NTU骨架连接关系
SKELETON_EDGES = [
    (3, 2), (2, 20), (20, 1), (1, 0),  # 头部和脊柱
    (20, 4), (4, 5), (5, 6),           # 左臂
    (6, 22), (6, 7), (7, 21),          # 左手指(映射)
    (20, 8), (8, 9), (9, 10),          # 右臂
    (10, 24), (10, 11), (11, 23),      # 右手指(映射)
    (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
    (0, 16), (16, 17), (17, 18), (18, 19)   # 右腿
]


def calculate_mse_without_fingers(skeleton1, skeleton2):
    """计算MSE，排除手指6关节的影响"""
    if isinstance(skeleton1, torch.Tensor):
        skeleton1 = skeleton1.cpu().numpy()
    if isinstance(skeleton2, torch.Tensor):
        skeleton2 = skeleton2.cpu().numpy()
    
    # 创建掩码：仅包含非手指关节
    mask = np.ones(skeleton1.shape[0], dtype=bool)
    for joint_idx in FINGER_JOINTS:
        mask[joint_idx] = False
    
    # 仅计算19个真实关节的MSE
    return np.mean((skeleton1[mask] - skeleton2[mask])**2)


def plot_skeleton_3d(ax, skeleton, title, color='blue'):
    """绘制3D骨架(不显示手指关节)"""
    if isinstance(skeleton, torch.Tensor):
        skeleton = skeleton.cpu().numpy()
    
    skeleton = skeleton.copy()
    skeleton[:, 2] = -skeleton[:, 2]  # 反转Z轴改善视角
    
    # 过滤手指关节的边
    edges = [e for e in SKELETON_EDGES if e[0] not in FINGER_JOINTS and e[1] not in FINGER_JOINTS]
    
    # 绘制骨骼连接
    for start_idx, end_idx in edges:
        start, end = skeleton[start_idx], skeleton[end_idx]
        if not (np.all(start == 0) or np.all(end == 0)):
            ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                     color=color, alpha=0.8, linewidth=2.0)
    
    # 绘制关节点(排除手指)
    for i in range(len(skeleton)):
        if i in FINGER_JOINTS:
            continue
        joint = skeleton[i]
        if not np.all(joint == 0):
            ax.scatter(joint[0], joint[1], joint[2],
                      c=color, s=25, alpha=0.9, edgecolors='white', linewidth=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置等比例坐标范围
    valid_joints = skeleton[~np.all(skeleton == 0, axis=1)]
    if len(valid_joints) > 0:
        center = np.mean(valid_joints, axis=0)
        max_range = max(np.max(np.max(valid_joints, axis=0) - np.min(valid_joints, axis=0)) / 2, 0.3)
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
        ax.set_box_aspect([1, 1, 1])
    
    ax.view_init(elev=15, azim=45)


def apply_temporal_smoothing(data, window_size=5):
    """
    对数据进行时序平滑 (移动平均)
    data: [T, ...] numpy array
    """
    if window_size <= 1:
        return data
    
    T = data.shape[0]
    if T < window_size:
        return data
        
    # print(f"   🔄 应用时序平滑 (Window={window_size})...")
    
    # Reshape to [T, -1] for easy processing
    original_shape = data.shape
    flattened = data.reshape(T, -1)
    smoothed_flat = np.zeros_like(flattened)
    pad_size = window_size // 2
    
    for i in range(flattened.shape[1]):
        padded = np.pad(flattened[:, i], (pad_size, pad_size), mode='edge')
        kernel = np.ones(window_size) / window_size
        convolved = np.convolve(padded, kernel, mode='valid')
        
        if len(convolved) > T:
            convolved = convolved[:T]
        smoothed_flat[:, i] = convolved
        
    return smoothed_flat.reshape(original_shape)


def generate_gif_animation(pipeline, radar_data_path, output_dir, grouping_type, 
                           num_sequences=10, frames_per_sequence=8, fps=3):
    """生成GIF动画"""
    print(f"\n🎬 生成 {grouping_type.upper()} GIF...")
    
    gif_output_dir = os.path.join(output_dir, f"gif_10p_adaptive_576_balance" if grouping_type == '10p' else f"gif_{grouping_type}")
    os.makedirs(gif_output_dir, exist_ok=True)
    
    # 加载数据
    labels_path = '/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy'
    if not os.path.exists(radar_data_path) or not os.path.exists(labels_path):
        print(f"❌ 数据文件不存在")
        return []
    
    full_data = np.load(radar_data_path)
    label_data = np.load(labels_path)
    print(f"✅ 数据: radar {full_data.shape}, label {label_data.shape}")
    
    gif_info_list = []
    
    # 生成多个序列的GIF
    for seq_idx in range(num_sequences):
        # 为每个序列选择不同的起始位置
        start_idx = seq_idx * (len(full_data) // (num_sequences + 1))
        end_idx = min(start_idx + frames_per_sequence, len(full_data))
        
        if end_idx - start_idx < frames_per_sequence:
            # 如果数据不够，从末尾向前取
            end_idx = len(full_data) - 1
            start_idx = max(0, end_idx - frames_per_sequence + 1)
        
        print(f"📹 生成序列 {seq_idx+1}/{num_sequences}: 帧 {start_idx}-{end_idx-1}")
        
        # 提取序列数据
        sequence_data = full_data[start_idx:end_idx]
        sequence_labels = label_data[start_idx:end_idx]
        
        # 1. 批量提取骨架
        extracted_skeletons = []
        for radar_frame in sequence_data:
            radar_tensor = torch.from_numpy(radar_frame.transpose(2, 0, 1)).unsqueeze(0).float().to(pipeline.device)
            extracted = pipeline.extract_skeleton(radar_tensor)
            extracted_skeletons.append(extracted.cpu().numpy())
            
        extracted_skeletons = np.concatenate(extracted_skeletons, axis=0) # [T, 25, 3]
        
        # 2. 应用时序平滑
        smoothed_skeletons = apply_temporal_smoothing(extracted_skeletons, window_size=5)
        
        # 3. 重构与结果收集
        frame_results = []
        for frame_idx, (label_frame, smoothed_skel) in enumerate(zip(sequence_labels, smoothed_skeletons)):
            # 使用平滑后的骨架进行重构
            smoothed_skel_tensor = torch.from_numpy(smoothed_skel).unsqueeze(0).to(pipeline.device)
            
            # 重构
            recon_result = pipeline.reconstruct_skeleton(smoothed_skel_tensor)
            
            # 计算MSE(排除手指6关节)
            mse = calculate_mse_without_fingers(
                smoothed_skel,
                recon_result['reconstructed'][0].cpu().numpy()
            )
            
            frame_results.append({
                'frame_idx': frame_idx,
                'label': label_frame,
                'extracted': smoothed_skel,
                'reconstructed': recon_result['reconstructed'][0].cpu().numpy(),
                'mse': mse
            })
        
        # 生成GIF
        gif_path = os.path.join(gif_output_dir, f'skeleton_reconstruction_sequence_{seq_idx+1:02d}.gif')
        gif_info = create_skeleton_gif(frame_results, gif_path, grouping_type, pipeline, fps)
        gif_info.update({'sequence_id': seq_idx + 1, 'start_frame': start_idx, 'end_frame': end_idx - 1})
        gif_info_list.append(gif_info)
    
    return gif_info_list


def create_skeleton_gif(frame_results, gif_path, grouping_type, pipeline, fps=3):
    """创建骨架重构GIF动画"""
    num_frames = len(frame_results)
    if num_frames == 0:
        return {'success': False, 'path': gif_path}
    
    fig = plt.figure(figsize=(24, 8))
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 10
    })
    
    def animate(frame_idx):
        fig.clear()
        current = frame_results[frame_idx]
        
        # 转换MARS标签为NTU格式
        mars_tensor = torch.tensor(current['label']).unsqueeze(0).float().to(pipeline.device)
        label_ntu = pipeline.joint_mapper(mars_tensor)[0].detach().cpu().numpy()
        
        # 创建3个子图
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        # 绘制三种骨架(均不显示手指)
        plot_skeleton_3d(ax1, label_ntu, f'Frame {frame_idx+1}/{num_frames}: Ground Truth', 'blue')
        plot_skeleton_3d(ax2, current['extracted'], f'Frame {frame_idx+1}/{num_frames}: Extracted', 'green')
        plot_skeleton_3d(ax3, current['reconstructed'], f'Frame {frame_idx+1}/{num_frames}: Reconstructed', 'red')
        
        # 标题(MSE已排除手指关节)
        fig.suptitle(f'Skeleton Reconstruction | Frame {frame_idx+1}/{num_frames} | MSE (19 joints): {current["mse"]:.6f}',
                    fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
    
    try:
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=False, repeat=True)
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=150)
        plt.close(fig)
        
        print(f"✅ GIF保存: {os.path.basename(gif_path)}")
        
        mse_errors = [fr['mse'] for fr in frame_results]
        return {
            'success': True,
            'path': gif_path,
            'num_frames': num_frames,
            'avg_mse': float(np.mean(mse_errors)),
            'min_mse': float(np.min(mse_errors)),
            'max_mse': float(np.max(mse_errors)),
            'frame_range': (0, num_frames-1)
        }
    except Exception as e:
        print(f"❌ GIF生成失败: {e}")
        plt.close(fig)
        return {'success': False, 'path': gif_path, 'error': str(e)}


def create_random_png_snapshots(pipeline, radar_data_path, output_dir, grouping_type, num_snapshots=20):
    """
    生成随机20帧PNG快照，展示GT、提取骨架和重构骨架的对比
    """
    print(f"\n📸 开始生成 {num_snapshots} 个随机PNG快照...")
    
    # 创建PNG输出目录
    png_dir = os.path.join(output_dir, f"png_snapshots_10p_adaptive_576_balance" if grouping_type == '10p' else f"png_snapshots_{grouping_type}")
    os.makedirs(png_dir, exist_ok=True)
    
    # 加载数据
    labels_path = '/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy'
    if not os.path.exists(radar_data_path) or not os.path.exists(labels_path):
        print(f"❌ 数据文件不存在")
        return None
    
    full_data = np.load(radar_data_path)
    label_data = np.load(labels_path)
    total_samples = len(full_data)
    
    # 随机选择不重复的帧索引
    np.random.seed(42)
    random_indices = np.random.choice(total_samples, size=min(num_snapshots, total_samples), replace=False)
    random_indices = sorted(random_indices.tolist())
    
    print(f"✓ 随机选择的帧索引: {random_indices[:10]}..." if len(random_indices) > 10 else f"✓ 随机选择的帧索引: {random_indices}")
    
    snapshot_results = []
    
    for idx, sample_idx in enumerate(random_indices, 1):
        # 加载数据
        radar_frame = full_data[sample_idx]
        label_frame = label_data[sample_idx]
        
        # 处理
        radar_tensor = torch.from_numpy(radar_frame.transpose(2, 0, 1)).unsqueeze(0).float().to(pipeline.device)
        result = pipeline.process_full_pipeline(radar_tensor)
        
        # 转换MARS标签为NTU格式
        mars_tensor = torch.tensor(label_frame).unsqueeze(0).float().to(pipeline.device)
        label_ntu = pipeline.joint_mapper(mars_tensor)[0].detach().cpu().numpy()
        
        extracted = result['extracted'][0].cpu().numpy()
        reconstructed = result['reconstructed'][0].cpu().numpy()
        
        # 计算MSE(排除手指)
        mse_extracted = calculate_mse_without_fingers(label_ntu, extracted)
        mse_reconstructed = calculate_mse_without_fingers(label_ntu, reconstructed)
        
        # 创建图形
        fig = plt.figure(figsize=(24, 8))
        plt.rcParams.update({
            'font.sans-serif': ['DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10
        })
        
        # 3个子图
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        # 绘制三种骨架(均不显示手指)
        plot_skeleton_3d(ax1, label_ntu, f'Frame {sample_idx+1}: Ground Truth', 'blue')
        plot_skeleton_3d(ax2, extracted, f'Frame {sample_idx+1}: Extracted', 'green')
        plot_skeleton_3d(ax3, reconstructed, f'Frame {sample_idx+1}: Reconstructed', 'red')
        
        # 标题
        title_text = (f'Skeleton Pipeline ({grouping_type.upper()}) | Frame {sample_idx+1:04d} | '
                     f'MSE Extracted: {mse_extracted:.6f} | MSE Reconstructed: {mse_reconstructed:.6f} (19 joints)')
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)
        
        # 保存PNG
        png_path = os.path.join(png_dir, f'skeleton_frame_{sample_idx+1:04d}.png')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        snapshot_results.append({
            'frame_idx': sample_idx,
            'mse_extracted': float(mse_extracted),
            'mse_reconstructed': float(mse_reconstructed)
        })
        
        if idx % 5 == 0 or idx == len(random_indices):
            print(f"  进度: {idx}/{len(random_indices)} 张PNG已生成")
    
    print(f"✅ PNG快照生成完成!")
    print(f"输出目录: {png_dir}/")
    
    # 统计信息
    mse_extracted_list = [r['mse_extracted'] for r in snapshot_results]
    mse_reconstructed_list = [r['mse_reconstructed'] for r in snapshot_results]
    
    stats = {
        'num_snapshots': len(snapshot_results),
        'sample_indices': random_indices,
        'avg_mse_extracted': float(np.mean(mse_extracted_list)),
        'avg_mse_reconstructed': float(np.mean(mse_reconstructed_list)),
        'min_mse_extracted': float(np.min(mse_extracted_list)),
        'min_mse_reconstructed': float(np.min(mse_reconstructed_list)),
        'max_mse_extracted': float(np.max(mse_extracted_list)),
        'max_mse_reconstructed': float(np.max(mse_reconstructed_list)),
        'output_dir': png_dir
    }
    
    # 保存统计
    stats_path = os.path.join(png_dir, 'png_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ PNG统计: 提取MSE {stats['avg_mse_extracted']:.6f}, 重构MSE {stats['avg_mse_reconstructed']:.6f}")
    
    return stats


def process_with_grouping_gif(grouping_type, radar_data_path, output_base_dir, generate_png=True):
    """使用指定分组类型处理数据并生成GIF"""
    print("\n" + "=" * 80)
    print(f"处理 {grouping_type.upper()} 分组")
    print("=" * 80)
    
    pipeline = create_pipeline(grouping_type=grouping_type)
    pipeline.print_info()
    
    num_sequences = 10 if grouping_type == '10p' else 5

    gif_info_list = generate_gif_animation(
        pipeline=pipeline,
        radar_data_path=radar_data_path,
        output_dir=output_base_dir,
        grouping_type=grouping_type,
        num_sequences=num_sequences,
        frames_per_sequence=8,
        fps=2
    )
    
    output_dir = os.path.join(output_base_dir, f"gif_10p_adaptive_576_balance" if grouping_type == '10p' else f"gif_{grouping_type}")
    
    # 生成PNG快照
    png_stats = None
    if generate_png:
        print("\n" + "=" * 80)
        png_stats = create_random_png_snapshots(
            pipeline=pipeline,
            radar_data_path=radar_data_path,
            output_dir=output_base_dir,
            grouping_type=grouping_type,
            num_snapshots=20
        )
    
    stats = {
        'grouping_type': grouping_type,
        'grouping_name': GROUPING_CONFIGS[grouping_type].name,
        'num_sequences': len(gif_info_list),
        'note': 'MSE计算已排除6个手指映射关节，仅基于19个真实关节',
        'png_snapshots': png_stats if png_stats else None,
        'sequences': [{
            'sequence_id': g['sequence_id'],
            'start_frame': g['start_frame'],
            'end_frame': g['end_frame'],
            'num_frames': g['num_frames'],
            'avg_mse': g['avg_mse'],
            'min_mse': g['min_mse'],
            'max_mse': g['max_mse'],
            'file': os.path.basename(g['path'])
        } for g in gif_info_list if g['success']]
    }
    
    with open(os.path.join(output_dir, 'gif_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计
    successful = sum(1 for g in gif_info_list if g['success'])
    print(f"\n📊 统计: {successful}/{len(gif_info_list)} 个GIF成功")
    
    if successful > 0:
        all_mse = [g['avg_mse'] for g in gif_info_list if g['success']]
        print(f"  GIF MSE (19关节): {np.mean(all_mse):.6f} [{np.min(all_mse):.6f} - {np.max(all_mse):.6f}]")
    
    if png_stats:
        print(f"  PNG MSE 提取: {png_stats['avg_mse_extracted']:.6f} [{png_stats['min_mse_extracted']:.6f} - {png_stats['max_mse_extracted']:.6f}]")
        print(f"  PNG MSE 重构: {png_stats['avg_mse_reconstructed']:.6f} [{png_stats['min_mse_reconstructed']:.6f} - {png_stats['max_mse_reconstructed']:.6f}]")
    
    print(f"\n📁 输出: {output_dir}/")
    for g in gif_info_list:
        if g['success']:
            print(f"  ✓ 序列{g['sequence_id']:02d}: {os.path.basename(g['path'])} (MSE: {g['avg_mse']:.6f})")
        else:
            print(f"  ✗ 序列{g['sequence_id']:02d}: {g.get('error', '失败')}")


def compare_groupings_gif(radar_data_path, output_base_dir, generate_png=True):
    """对比不同分组配置的GIF效果"""
    print("\n" + "=" * 80)
    print("对比模式 - 生成所有分组GIF")
    print("=" * 80)
    
    for grouping_type in GROUPING_CONFIGS.keys():
        process_with_grouping_gif(grouping_type, radar_data_path, output_base_dir, generate_png=generate_png)
    
    comparison_stats = {
        'groupings': list(GROUPING_CONFIGS.keys()),
        'note': 'MSE基于19个真实关节(排除6个手指映射关节)'
    }
    
    with open(os.path.join(output_base_dir, 'grouping_comparison_gif.json'), 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    print("\n🎉 所有分组GIF生成完成！")


def main():
    parser = argparse.ArgumentParser(description='多分组骨架流水线 - GIF动画生成器')
    parser.add_argument('--mode', type=str, default='5p', 
                       choices=['5p', '8p', '10p', 'all', 'compare'],
                       help='处理模式: 5p=5分组, 8p=8分组, 10p=10分组, all=所有分组, compare=对比')
    parser.add_argument('--data', type=str, 
                       default='/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy',
                       help='雷达数据路径')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='输出目录')
    parser.add_argument('--no-png', action='store_true',
                       help='禁用PNG快照生成')
    
    args = parser.parse_args()
    
    generate_png = not args.no_png
    
    print("=" * 80)
    print("🚀 多分组骨架流水线 - GIF动画生成器")
    print("=" * 80)
    print(f"📂 数据路径: {args.data}")
    print(f"📂 输出目录: {args.output}")
    print(f"🎯 处理模式: {args.mode}")
    print(f"📸 PNG快照: {'启用' if generate_png else '禁用'}")
    
    if args.mode == 'compare' or args.mode == 'all':
        compare_groupings_gif(args.data, args.output, generate_png=generate_png)
    else:
        process_with_grouping_gif(args.mode, args.data, args.output, generate_png=generate_png)


if __name__ == "__main__":
    main()
