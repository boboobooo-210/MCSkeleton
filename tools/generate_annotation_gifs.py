#!/usr/bin/env python3
"""
Generate Annotation GIFs for NTU Dataset (10-part Model)
========================================================

Function:
1. Loads the 10-part GCNSkeletonTokenizer model.
2. Scans the NTU dataset to find samples for each token in the codebooks.
3. Generates GIFs for these samples to facilitate manual annotation.
4. Organizes the output into 6 Limb Groups (Head, Spine, Left/Right Upper, Left/Right Lower)
   to help the annotator understand the context (e.g. seeing the whole arm when annotating the forearm).

Usage:
python tools/generate_annotation_gifs.py \
    --config cfgs/NTU_models/gcn_skeleton_memory_optimized_10p.yaml \
    --checkpoint experiments/gcn_skeleton_memory_optimized_10p/NTU_models/adaptive_gcnskeleton_576tokens_balanced/ckpt-best.pth \
    --output_dir annotation_materials \
    --samples_per_token 3
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import importlib.util

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.config import cfg_from_yaml_file
from models.GCNSkeletonTokenizer_10p import GCNSkeletonTokenizer_10p
from datasets.build import build_dataset_from_cfg as build_dataset

# Import Visualizer
vis_path = project_root / 'visualizations' / '0_gcn' / 'gcn_skeleton_gif_visualizer.py'
spec = importlib.util.spec_from_file_location('gcn_skeleton_gif_visualizer', vis_path)
vis_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis_mod)
GCNSkeletonGifVisualizer = vis_mod.GCNSkeletonGifVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Annotation GIFs')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='annotation_materials')
    parser.add_argument('--samples_per_token', type=int, default=3)
    parser.add_argument('--max_batches', type=int, default=200, help='Limit scanning to save time')
    return parser.parse_args()

def get_limb_group(semantic_group):
    """Map 10 semantic groups to 6 annotation limb groups"""
    mapping = {
        'head_neck': 'Head',
        'spine': 'Spine',
        'left_arm': 'Left_Upper_Limb',
        'left_forearm': 'Left_Upper_Limb',
        'right_arm': 'Right_Upper_Limb',
        'right_forearm': 'Right_Upper_Limb',
        'left_leg': 'Left_Lower_Limb',
        'left_foot': 'Left_Lower_Limb',
        'right_leg': 'Right_Lower_Limb',
        'right_foot': 'Right_Lower_Limb'
    }
    return mapping.get(semantic_group, 'Other')

def create_simple_gif(data, save_path, title="Skeleton"):
    """
    Create a simple skeleton GIF using matplotlib.
    data: (T, 25, 3) numpy array
    """
    # NTU Skeleton Connections
    connections = [
        (3, 2), (2, 20), (20, 1), (1, 0), # Spine
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22), # Left Arm
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10, 24), # Right Arm
        (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
        (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine bounds
    min_vals = np.min(data, axis=(0, 1))
    max_vals = np.max(data, axis=(0, 1))
    mid_vals = (max_vals + min_vals) / 2
    max_range = np.max(max_vals - min_vals) / 2
    
    def update(frame):
        ax.clear()
        ax.set_title(title)
        
        # Swap axes for visualization: Y-up (NTU) -> Z-up (Plot)
        # Data: (X, Y, Z)
        # Plot: (X, Z, Y)
        
        # X axis
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        # Y axis (in plot) corresponds to Z in data
        ax.set_ylim(mid_vals[2] - max_range, mid_vals[2] + max_range)
        # Z axis (in plot) corresponds to Y in data
        ax.set_zlim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z') 
        ax.set_zlabel('Y')
        
        # Plot joints
        current_pose = data[frame]
        # Swap Y and Z: x->x, y->z, z->y
        x = current_pose[:, 0]
        y = current_pose[:, 2]
        z = current_pose[:, 1]
        
        ax.scatter(x, y, z, c='b', marker='o', s=20)
        
        # Plot connections
        for i, j in connections:
            ax.plot([current_pose[i, 0], current_pose[j, 0]],
                    [current_pose[i, 2], current_pose[j, 2]], # Z -> Y
                    [current_pose[i, 1], current_pose[j, 1]], # Y -> Z
                    c='r')

    anim = FuncAnimation(fig, update, frames=len(data), interval=200)
    anim.save(save_path, writer='pillow', fps=5)
    plt.close(fig)

def main():
    args = parse_args()
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = cfg_from_yaml_file(args.config)
    
    # 2. Load Model
    print("Loading model...")
    # cfg.model is the config dict for the model
    model = GCNSkeletonTokenizer_10p(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict keys
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 3. Load Dataset
    print("Loading dataset...")
    
    # Manual merge for dataset config
    dataset_cfg = cfg.dataset.train
    if '_base_' in dataset_cfg:
        base_cfg = dataset_cfg['_base_']
        # Merge base into dataset_cfg
        for k, v in base_cfg.items():
            if k not in dataset_cfg:
                dataset_cfg[k] = v
        # Merge 'others' to override
        if 'others' in dataset_cfg:
            for k, v in dataset_cfg['others'].items():
                dataset_cfg[k] = v
                
    dataset = build_dataset(dataset_cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.dataset.train.others.bs, 
        shuffle=True, # Shuffle to get random samples
        num_workers=4
    )
    
    # 4. Scan for Samples
    print("Scanning dataset for token samples...")
    
    # Structure: group_name -> token_id -> list of (data_sample)
    token_samples = {} 
    group_names = ['head_neck', 'spine', 'left_arm', 'left_forearm', 'right_arm', 'right_forearm', 'left_leg', 'left_foot', 'right_leg', 'right_foot']
    
    # Initialize storage
    for gn in group_names:
        token_samples[gn] = {}
        # Pre-fill keys
        codebook_size = model.codebook_configs[gn]['num_embeddings']
        for i in range(codebook_size):
            token_samples[gn][i] = []

    batches_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batches_processed >= args.max_batches:
                break
                
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3 and isinstance(batch[2], torch.Tensor):
                    data = batch[2]
                elif isinstance(batch[0], torch.Tensor):
                    data = batch[0]
                else:
                    continue
            else:
                continue
                
            data = data.float().to(device)
            
            # Encode
            # tokens: (B, num_groups)
            tokens = model.encode(data)
            
            # Store samples
            B = data.shape[0]
            data_cpu = data.cpu().numpy() # (B, 25, 3)
            tokens_cpu = tokens.cpu().numpy() # (B, 10)
            
            for b in range(B):
                sample_data = data_cpu[b]
                sample_tokens = tokens_cpu[b]
                
                for i, group_name in enumerate(group_names):
                    token_id = sample_tokens[i]
                    
                    # Store if we need more samples for this token
                    if len(token_samples[group_name][token_id]) < args.samples_per_token:
                        token_samples[group_name][token_id].append(sample_data)
            
            # Check if we have enough samples for all tokens (optimization)
            # For now just run for max_batches
            batches_processed += 1

    # 5. Generate GIFs
    print("Generating GIFs...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_gifs = 0
    
    for group_name in group_names:
        limb_group = get_limb_group(group_name)
        group_dir = os.path.join(args.output_dir, limb_group)
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"Processing {group_name} -> {limb_group}")
        
        for token_id, samples in token_samples[group_name].items():
            if not samples:
                continue
                
            # Create a folder for this token? Or just files?
            # User wants to annotate. Files might be easier to browse if named well.
            # Format: {SubGroup}_Token_{ID}_Sample_{k}.gif
            
            for k, sample_data in enumerate(samples):
                # sample_data is (25, 3) - single frame?
                # Wait, NTU data is usually (C, T, V) or (T, V, C)?
                # The model input was (B, 25, 3) or (B, T, 25, 3).
                # If the input to model.encode was (B, 25, 3), then it's single frame.
                # If it was (B, T, 25, 3), then sample_data is (T, 25, 3).
                
                # Let's check data shape
                if sample_data.ndim == 2: # (25, 3)
                    # Make it a short sequence for visualization (repeat)
                    # Or just visualize static pose
                    # But static pose GIF is boring.
                    # Maybe rotate it?
                    
                    # Create a rotation animation
                    frames = []
                    for angle in np.linspace(0, 360, 20):
                        # Rotate around Y axis (up)
                        theta = np.radians(angle)
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                        rotated = np.dot(sample_data, R.T)
                        frames.append(rotated)
                    frames = np.array(frames)
                    
                    filename = f"{group_name}_Token_{token_id:02d}_Sample_{k}.gif"
                    save_path = os.path.join(group_dir, filename)
                    
                    create_simple_gif(frames, save_path, title=f"{group_name} Token {token_id}")
                    total_gifs += 1
                    
                elif sample_data.ndim == 3: # (T, 25, 3)
                    filename = f"{group_name}_Token_{token_id:02d}_Sample_{k}.gif"
                    save_path = os.path.join(group_dir, filename)
                    create_simple_gif(sample_data, save_path, title=f"{group_name} Token {token_id}")
                    total_gifs += 1

    print(f"Done! Generated {total_gifs} GIFs in {args.output_dir}")

if __name__ == '__main__':
    main()
