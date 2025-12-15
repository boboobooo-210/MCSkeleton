#!/usr/bin/env python3
"""
去除可视化的 MARS -> NTU 提取 + GCN 重构批处理脚本

功能：
- 批量读取 MARS 的雷达特征图 npy 文件（train/test/validate）
- 使用 MARSTransformerModel 提取 MARS (19 joints)
- 使用 SkeletonJointMapper/EnhancedSkeletonMapper 转换为 NTU 25 关节
- 使用已训练的 GCN 重构器得到重构骨架和 token_sequence
- 将重构骨架、token_sequence、vq_loss、（可选）原始提取骨架按索引绑定并保存为压缩 npz

输出格式（单文件）：
  <out_dir>/<split>_recon.npz 包含：
    - reconstructed: float32 array (N, 25, 3)  # 已经是 [x,y,z] 格式（标准化或模型输出，见下）
    - token_sequences: int32 array (N, T) 或 (N,)
    - vq_losses: float32 array (N,)
    - extracted: float32 array (N, 25, 3)  # 可选：提取器的输出（NTU格式）
    - metadata: JSON 字符串（N 条元信息）

此外可选 per-sample 压缩文件（每样本一个 .npz），如需下游按样本加载更方便。

使用示例：
  python tools/skeleton_extraction_reconstruction_saver.py \
    --extractor mars_optimized_best.pth \
    --gcn_ckpt experiments/gcn_skeleton_memory_optimized_10p/NTU_models/adaptive_gcnskeleton_576tokens_balanced/ckpt-best.pth \
    --gcn_cfg cfgs/NTU_models/gcn_skeleton_memory_optimized_10p.yaml \
    --out_dir data/MARS_recon_tokens_10p \
    --batch_size 32

"""

import os
import sys
import json
import copy
from pathlib import Path
import argparse
import numpy as np
from datetime import datetime

# 把项目根加入路径，便于导入 models 等
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# 延迟导入 torch 并给出友好信息
try:
    import torch
except Exception as e:
    print("❌ 需要 PyTorch 才能运行此脚本，请先安装 torch：", e)
    sys.exit(1)

# 导入模型与映射器
try:
    from models.skeleton_extractor import MARSTransformerModel
    from models.skeleton_extractor_final import OptimizedMARSModel
    from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
    from utils.config import cfg_from_yaml_file
    from models.skeleton_joint_mapper import SkeletonJointMapper, EnhancedSkeletonMapper
except Exception as e:
    print("❌ 导入项目内模块失败：", e)
    raise

# 可选导入10部位Tokenizer（新模型）
try:
    from models.GCNSkeletonTokenizer_10p import GCNSkeletonTokenizer_10p
except Exception:
    GCNSkeletonTokenizer_10p = None

# 进度条
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x


DEFAULT_GROUP_DISPLAY_NAMES_ZH = {
    'head_spine': '头部脊柱',
    'head_neck': '头颈',
    'spine': '脊柱',
    'left_arm': '左上臂',
    'left_forearm': '左前臂与手',
    'right_arm': '右上臂',
    'right_forearm': '右前臂与手',
    'left_leg': '左大腿',
    'left_foot': '左小腿与脚',
    'right_leg': '右大腿',
    'right_foot': '右小腿与脚'
}

DEFAULT_GROUP_DISPLAY_NAMES_EN = {
    'head_spine': 'Head & Spine',
    'head_neck': 'Head & Neck',
    'spine': 'Spine',
    'left_arm': 'Left Upper Arm',
    'left_forearm': 'Left Forearm & Hand',
    'right_arm': 'Right Upper Arm',
    'right_forearm': 'Right Forearm & Hand',
    'left_leg': 'Left Thigh',
    'left_foot': 'Left Calf & Foot',
    'right_leg': 'Right Thigh',
    'right_foot': 'Right Calf & Foot'
}


class SkeletonReconstructionSaver:
    def __init__(self, extractor_ckpt, gcn_ckpt, gcn_cfg, device=None, use_enhanced_mapper=True):
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        # 映射器
        if use_enhanced_mapper:
            self.joint_mapper = EnhancedSkeletonMapper().to(self.device)
        else:
            self.joint_mapper = SkeletonJointMapper().to(self.device)

        # 加载模型
        self.skeleton_extractor = self._load_skeleton_extractor(extractor_ckpt)
        self.gcn_reconstructor = self._load_gcn_reconstructor(gcn_ckpt, gcn_cfg)

        if self.gcn_reconstructor is None:
            raise RuntimeError(
                f"Failed to load GCN reconstructor. Check checkpoint and config.\n"
                f"  ckpt: {gcn_ckpt}\n  cfg: {gcn_cfg}"
            )

        # 语义组与码本信息（用于多部位Tokenizer）
        self.group_names = []
        self.group_token_sizes = []
        self.group_offsets = []
        self.group_display_names = []
        self.group_joint_indices = []
        self.group_tokens_config = {}
        self.total_token_vocab = None
        self.default_tokens_per_group = 128
        self.tokenizer_model_name = None
        self.num_token_groups = None

        if self.gcn_reconstructor is not None:
            self.tokenizer_model_name = self.gcn_reconstructor.__class__.__name__
            semantic_groups = getattr(self.gcn_reconstructor.skeleton_graph, 'semantic_groups', {}) or {}
            tokens_config_raw = getattr(getattr(self.gcn_reconstructor, 'semantic_codebooks', None), 'tokens_config', {}) or {}
            self.default_tokens_per_group = getattr(getattr(self.gcn_reconstructor, 'semantic_codebooks', None), 'tokens_per_group', 128)

            # 规范化 tokens_config，确保返回普通字典
            tokens_config = {}
            if hasattr(tokens_config_raw, 'items'):
                tokens_config = {str(k): int(v) for k, v in tokens_config_raw.items()}
            elif isinstance(tokens_config_raw, dict):
                tokens_config = {str(k): int(v) for k, v in tokens_config_raw.items()}

            if semantic_groups:
                self.group_names = list(semantic_groups.keys())
                offset = 0
                for group_name in self.group_names:
                    group_size = int(tokens_config.get(group_name, self.default_tokens_per_group))
                    self.group_token_sizes.append(group_size)
                    self.group_offsets.append(offset)
                    self.group_joint_indices.append([int(idx) for idx in semantic_groups[group_name]])
                    zh_name, en_name = self._resolve_display_names(group_name)
                    self.group_display_names.append({'zh': zh_name, 'en': en_name})
                    self.group_tokens_config[group_name] = group_size
                    offset += group_size

                # 如果tokens_config缺失但semantic_groups存在，仍需回填默认值
                if not tokens_config:
                    for group_name in self.group_names:
                        self.group_tokens_config[group_name] = self.default_tokens_per_group

                self.total_token_vocab = offset if offset > 0 else None

        if not self.group_names:
            semantic_groups = getattr(getattr(self.gcn_reconstructor, 'skeleton_graph', None), 'semantic_groups', {}) or {}
            if semantic_groups:
                self.group_names = list(semantic_groups.keys())
                self.num_token_groups = len(self.group_names)
            else:
                self.num_token_groups = 5
        else:
            self.num_token_groups = len(self.group_names)

        if self.num_token_groups is None:
            codebooks = getattr(getattr(self.gcn_reconstructor, 'semantic_codebooks', None), 'group_codebooks', None)
            if isinstance(codebooks, dict) and codebooks:
                self.num_token_groups = len(codebooks)
        if self.num_token_groups is None:
            self.num_token_groups = 5

    @staticmethod
    def _title_case(name):
        return name.replace('_', ' ').title()

    def _resolve_display_names(self, group_name):
        zh_name = DEFAULT_GROUP_DISPLAY_NAMES_ZH.get(group_name)
        en_name = DEFAULT_GROUP_DISPLAY_NAMES_EN.get(group_name)

        if zh_name is None:
            zh_name = self._title_case(group_name)
        if en_name is None:
            en_name = self._title_case(group_name)

        return zh_name, en_name

    def _load_skeleton_extractor(self, model_path):
        """加载MARS骨架提取器 - 支持 OptimizedMARSModel 和 MARSTransformerModel"""
        print(f"Loading skeleton extractor: {model_path}")
        
        # 尝试加载权重以检测模型类型
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 1. 检测是否为 OptimizedMARSModel (v2.0)
        # 特征: 包含 'backbone.stage1.0.weight' (SpatialPreservingBackbone)
        if 'backbone.stage1.0.weight' in state_dict:
            print("🔍 检测到 OptimizedMARSModel (v2.0)")
            model = OptimizedMARSModel(input_channels=5, output_dim=57)
        
        # 2. 检测是否为 MARSTransformerModel (v1.0)
        else:
            # 检测是否为多尺度模型（通过第一层Linear的输入维度判断）
            first_linear_key = 'regression_head.feature_projection.0.weight'
            
            if first_linear_key in state_dict:
                input_dim = state_dict[first_linear_key].shape[1]  # (out_features, in_features)
                is_multi_scale = (input_dim == 448)
                
                if is_multi_scale:
                    print("🔍 检测到多尺度模型 (448维输入)")
                    model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True)
                else:
                    print("🔍 检测到单尺度模型 (256维输入)")
                    model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=False)
            else:
                # 如果找不到关键层，默认尝试多尺度（新版本）
                print("⚠️ 无法检测模型类型，默认使用多尺度模型")
                model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True)
        
        # 加载权重
        try:
            model.load_state_dict(state_dict)
            print("✅ 权重加载成功！")
        except RuntimeError as e:
            print(f"❌ 权重加载失败: {e}")
            print("⚠️ 可能是模型版本不匹配，请检查权重文件")
            raise
        
        model.to(self.device)
        model.eval()
        
        # 输出模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数: {total_params:,}")
        if hasattr(model, 'backbone'):
            # OptimizedMARSModel 和 MARSTransformerModel 都有 backbone，但属性可能不同
            if hasattr(model.backbone, 'output_dim'):
                print(f"📊 Backbone输出维度: {model.backbone.output_dim}")
        
        return model

    def _load_gcn_reconstructor(self, model_path, config_path):
        print(f"Loading GCN reconstructor: {model_path}")
        try:
            cfg = cfg_from_yaml_file(config_path)
            model_name = getattr(cfg.model, 'NAME', 'GCNSkeletonTokenizer')

            tokenizer_cls = None
            if model_name == 'GCNSkeletonTokenizer_10p' and GCNSkeletonTokenizer_10p is not None:
                tokenizer_cls = GCNSkeletonTokenizer_10p
            elif model_name == 'GCNSkeletonTokenizer':
                tokenizer_cls = GCNSkeletonTokenizer
            else:
                tokenizer_cls = GCNSkeletonTokenizer_10p if GCNSkeletonTokenizer_10p is not None else GCNSkeletonTokenizer

            model = tokenizer_cls(cfg.model)

            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('base_model', checkpoint)
            # 去掉 possible 'module.' 前缀
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            model.load_state_dict(new_state)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print("❌ 加载 GCN 重构器失败：", e)
            return None

    def _extract_batch(self, radar_batch):
        """ radar_batch: np.array (B, H, W, C) or (B, C, H, W) """
        # 转为 tensor (B, C, H, W)
        if len(radar_batch.shape) == 4 and radar_batch.shape[-1] in (1,3,5):
            tensor = torch.from_numpy(radar_batch.transpose(0, 3, 1, 2)).float().to(self.device)
        elif len(radar_batch.shape) == 4 and radar_batch.shape[1] in (1,3,5):
            tensor = torch.from_numpy(radar_batch).float().to(self.device)
        else:
            # 尝试处理 (H,W,C) 单样本
            if len(radar_batch.shape) == 3:
                tensor = torch.from_numpy(radar_batch.transpose(2,0,1)).unsqueeze(0).float().to(self.device)
            else:
                raise ValueError(f"Unexpected radar batch shape: {radar_batch.shape}")

        with torch.no_grad():
            out = self.skeleton_extractor(tensor)
            # 输出可能是 (B,57) 或 (B,19,3)
            if len(out.shape) == 2 and out.shape[1] == 57:
                # 保留原始 flat 57 输出供映射器使用
                flat = out
                # 但也构造 (B,19,3)
                x = out[:, 0:19]
                y = out[:, 19:38]
                z = out[:, 38:57]
                mars_skel = torch.stack([x, y, z], dim=-1)  # (B,19,3)
            else:
                # 假设是 (B,19,3)
                mars_skel = out
                flat = out.view(out.shape[0], -1)

            # 转换为 NTU 25 joints via mapper (joint_mapper 接受 flat 57 或 tensor)
            try:
                ntu = self.joint_mapper(flat)
            except Exception:
                # 有些映射器可能需要 CPU tensor
                ntu = self.joint_mapper(flat.to(self.device))

            return {
                'mars_flat': flat.cpu(),            # (B,57)
                'mars_skel': mars_skel.cpu(),      # (B,19,3)
                'ntu_skel': ntu.cpu()              # (B,25,3)
            }

    def _reconstruct_batch(self, ntu_batch):
        """ntubatch: np.array or torch tensor shape (B,25,3) - the method will normalize and feed GCN"""
        if self.gcn_reconstructor is None:
            raise RuntimeError("GCN reconstructor is not loaded")

        # normalize according to training expectation: convert (x,y,z)->(x,z,y) then normalize
        if isinstance(ntu_batch, torch.Tensor):
            ntu_np = ntu_batch.cpu().numpy()
        else:
            ntu_np = np.array(ntu_batch)

        # apply same normalization as pipeline: expects (B,25,3) in x,y,z -> convert to x,z,y then standardize per sample
        xzy = ntu_np[:, :, [0, 2, 1]]  # [x,y,z] -> [x,z,y]

        # normalization per-sample
        normalized = []
        for i in range(xzy.shape[0]):
            s = xzy[i]
            centroid = np.mean(s, axis=0)
            centered = s - centroid
            dists = np.sqrt(np.sum(centered**2, axis=1))
            maxd = np.max(dists)
            if maxd > 0:
                normalized.append(centered / maxd)
            else:
                normalized.append(centered)
        normalized = np.array(normalized).astype(np.float32)

        tensor_in = torch.from_numpy(normalized).to(self.device)
        with torch.no_grad():
            out = self.gcn_reconstructor(tensor_in, return_recon=True)

        # out expects keys: 'reconstructed' (B,25,3?) (in x,z,y or x,y,z depending model) and 'token_sequence' etc.
        recon = out['reconstructed'].cpu().numpy()
        tokens = out['token_sequence'].cpu().numpy()
        vq_loss = float(out.get('vq_loss', 0.0)) if not isinstance(out.get('vq_loss', 0.0), torch.Tensor) else out['vq_loss'].item()
        
        # 新增：获取 base_reconstructed 和 residual_scale（用于诊断）
        base_recon = out.get('base_reconstructed', None)
        if base_recon is not None:
            base_recon = base_recon.cpu().numpy()
        residual_scale = out.get('residual_scale', None)
        if residual_scale is not None and isinstance(residual_scale, torch.Tensor):
            residual_scale = float(residual_scale.cpu().numpy())

        # pipeline code used reconstructed_xzy then converted to x,y,z via [:,:, [0,2,1]]
        # if model returned xzy we convert to xyz
        try:
            # assume recon is in xzy -> convert
            recon_xyz = recon[:, :, [0, 2, 1]]
        except Exception:
            recon_xyz = recon
        
        # 转换 base_recon（如果存在）
        if base_recon is not None:
            try:
                base_recon_xyz = base_recon[:, :, [0, 2, 1]]
            except Exception:
                base_recon_xyz = base_recon
        else:
            base_recon_xyz = None

        return {
            'reconstructed': recon_xyz,
            'tokens': tokens,
            'vq_loss': vq_loss,
            'base_reconstructed': base_recon_xyz,
            'residual_scale': residual_scale
        }

    def analyze_token_diversity(self, tokens_array):
        """分析 token 序列的多样性（自适应语义组与码本大小）"""

        if tokens_array is None or tokens_array.size == 0:
            print("⚠️ Token array 为空，跳过多样性分析")
            return {}

        num_groups_detected = tokens_array.shape[1]
        group_names = self.group_names if self.group_names else [f'group_{i}' for i in range(num_groups_detected)]

        # 对齐语义组数量
        if len(group_names) < num_groups_detected:
            group_names = group_names + [f'group_{i}' for i in range(len(group_names), num_groups_detected)]

        group_sizes = self.group_token_sizes if self.group_token_sizes else [self.default_tokens_per_group] * len(group_names)
        group_offsets = self.group_offsets if self.group_offsets else [i * self.default_tokens_per_group for i in range(len(group_names))]

        group_stats = {}

        print("\n" + "="*80)
        print("TOKEN DIVERSITY ANALYSIS")
        print("="*80)

        total_possible_tokens = 0

        for group_idx in range(num_groups_detected):
            group_tokens = tokens_array[:, group_idx]
            group_name = group_names[group_idx]
            group_size = group_sizes[group_idx] if group_idx < len(group_sizes) else self.default_tokens_per_group
            group_offset = group_offsets[group_idx] if group_idx < len(group_offsets) else group_idx * self.default_tokens_per_group

            # 还原局部token ID
            original_tokens = group_tokens - group_offset
            unique_tokens, counts = np.unique(original_tokens, return_counts=True)

            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(group_size) if group_size > 0 else 1.0

            # 获取 Top-100 高频 Token (用于优先标注)
            top_k = 100
            top_indices = np.argsort(-counts)[:top_k]
            top_tokens = unique_tokens[top_indices]
            top_counts = counts[top_indices]

            # 构建完整的 Token 使用率字典 (Token ID -> Count)
            # 转换为 Python int 以便 JSON 序列化
            usage_dict = {int(t): int(c) for t, c in zip(unique_tokens, counts)}

            group_stats[group_name] = {
                'unique_tokens': int(len(unique_tokens)),
                'total_tokens': int(len(original_tokens)),
                'coverage': float(len(unique_tokens) / group_size * 100 if group_size > 0 else 0),
                'entropy': float(entropy),
                'normalized_entropy': float(entropy / max_entropy if max_entropy > 0 else 0),
                'top_100_tokens': top_tokens.tolist(),
                'top_100_counts': top_counts.tolist(),
                'top_100_freq': (top_counts / counts.sum() * 100).tolist(),
                'all_token_counts': usage_dict  # 新增：完整的使用率统计
            }

            total_possible_tokens += group_size

            print(f"\n{group_name.upper()} (Group {group_idx}):")
            print(f"  Unique tokens: {len(unique_tokens)}/{group_size} ({len(unique_tokens)/group_size*100 if group_size else 0:.1f}% coverage)")
            print(f"  Top 5 tokens: {top_tokens[:5]} (Counts: {top_counts[:5]})")
            print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} (normalized: {group_stats[group_name]['normalized_entropy']:.3f})")
            print("  Top-10 most used tokens:")
            for i, (tok, cnt, freq) in enumerate(zip(top_tokens, top_counts, top_counts/counts.sum()*100)):
                print(f"    #{i+1}: Token {tok:3d} - {cnt:5d} times ({freq:5.2f}%)")

        # 全局统计
        print(f"\n{'='*80}")
        print("GLOBAL SUMMARY:")
        print(f"{'='*80}")

        total_unique = sum(a['unique_tokens'] for a in group_stats.values())
        avg_coverage = np.mean([a['coverage'] for a in group_stats.values()]) if group_stats else 0
        avg_entropy = np.mean([a['normalized_entropy'] for a in group_stats.values()]) if group_stats else 0

        print(f"Total unique tokens used: {total_unique}/{total_possible_tokens} ({total_unique/total_possible_tokens*100 if total_possible_tokens else 0:.1f}%)")
        print(f"Average coverage per group: {avg_coverage:.1f}%")
        print(f"Average normalized entropy: {avg_entropy:.3f}")

        print(f"\n{'='*80}")
        print("DIAGNOSTIC:")
        print(f"{'='*80}")

        if avg_coverage < 10:
            print("❌ SEVERE COLLAPSE: Less than 10% of codebook used!")
            print("   → Model is NOT learning meaningful discrete representations")
        elif avg_coverage < 30:
            print("⚠️  MODERATE COLLAPSE: 10-30% of codebook used")
            print("   → Codebook learning is weak, consider increasing VQ loss weight")
        elif avg_coverage < 60:
            print("✓  FAIR: 30-60% of codebook used")
            print("   → Codebook is learning, but still has room for improvement")
        else:
            print("✅ GOOD: >60% of codebook used")
            print("   → Codebook is learning diverse representations")

        if avg_entropy < 0.3:
            print("❌ HIGHLY SKEWED: Token distribution is extremely imbalanced")
        elif avg_entropy < 0.6:
            print("⚠️  SKEWED: Token distribution is somewhat imbalanced")
        else:
            print("✅ BALANCED: Token distribution is relatively uniform")

        summary = {
            'total_unique_tokens': int(total_unique),
            'total_possible_tokens': int(total_possible_tokens),
            'unique_usage_rate': float(total_unique / total_possible_tokens * 100 if total_possible_tokens else 0),
            'average_coverage': float(avg_coverage),
            'average_normalized_entropy': float(avg_entropy)
        }

        return {
            'groups': group_stats,
            'summary': summary
        }
    
    def build_token_metadata(self):
        """构建语义组与码本的元数据描述"""

        if self.group_names:
            group_names = list(self.group_names)
        else:
            group_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

        group_sizes = list(self.group_token_sizes) if self.group_token_sizes else [self.default_tokens_per_group] * len(group_names)

        if self.group_offsets:
            group_offsets = list(self.group_offsets)
        else:
            group_offsets = []
            offset = 0
            for size in group_sizes:
                group_offsets.append(offset)
                offset += size

        display_map = {}
        for idx, name in enumerate(group_names):
            if self.group_display_names and idx < len(self.group_display_names):
                display_map[name] = self.group_display_names[idx]
            else:
                zh_name, en_name = self._resolve_display_names(name)
                display_map[name] = {'zh': zh_name, 'en': en_name}

        joint_indices = {}
        if self.group_joint_indices:
            for idx, name in enumerate(group_names):
                if idx < len(self.group_joint_indices):
                    joint_indices[name] = self.group_joint_indices[idx]
        elif self.gcn_reconstructor is not None:
            semantic_groups = getattr(self.gcn_reconstructor.skeleton_graph, 'semantic_groups', {}) or {}
            for name, joints in semantic_groups.items():
                joint_indices[name] = [int(j) for j in joints]

        tokens_config = self.group_tokens_config if self.group_tokens_config else {name: self.default_tokens_per_group for name in group_names}

        total_vocab = self.total_token_vocab
        if total_vocab is None:
            total_vocab = int(sum(group_sizes))

        metadata = {
            'schema_version': '1.1.0',
            'generated_at': datetime.now().isoformat(),
            'tokenizer_model': self.tokenizer_model_name or 'UnknownTokenizer',
            'num_groups': len(group_names),
            'group_order': group_names,
            'group_offsets': group_offsets,
            'group_token_sizes': group_sizes,
            'group_display_names': display_map,
            'group_joint_indices': joint_indices,
            'tokens_config': tokens_config,
            'default_tokens_per_group': int(self.default_tokens_per_group),
            'total_token_vocab': int(total_vocab)
        }

        return metadata

    def process_and_save(self, input_npy, out_dir, split_name='split', batch_size=32, per_sample=False, analyze_tokens=True):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Processing {input_npy} -> {out_dir} (batch_size={batch_size}, per_sample={per_sample})")

        data = np.load(input_npy)
        N = len(data)
        print(f"Loaded {N} samples from {input_npy}")

        # Prepare containers
        recon_list = []
        tokens_list = []
        vq_list = []
        extracted_list = []
        base_recon_list = []  # 新增：纯码本重建
        residual_scales = []  # 新增：残差系数
        metadata = []

        # iterate in batches
        for start in tqdm(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            batch = data[start:end]
            # ensure shape (B,H,W,C)
            if batch.ndim == 3:
                # (H,W,C) single sample -> expand
                batch = np.expand_dims(batch, 0)
            if batch.ndim == 4 and batch.shape[-1] not in (1,3,5):
                # maybe (B,C,H,W), convert to (B,H,W,C)
                batch = batch.transpose(0, 2, 3, 1)

            # extraction
            extracted = self._extract_batch(batch)
            ntu = extracted['ntu_skel'].numpy()  # (B,25,3)

            # reconstruction（返回值增强）
            try:
                recon_result = self._reconstruct_batch(ntu)
                recon_xyz = recon_result['reconstructed']
                tokens = recon_result['tokens']
                vq_loss = recon_result['vq_loss']
                base_recon_xyz = recon_result['base_reconstructed']
                residual_scale = recon_result['residual_scale']
            except Exception as e:
                print(f"❌ Reconstruction failed for batch {start}-{end}: {e}")
                # fill with zeros to keep alignment
                B = ntu.shape[0]
                recon_xyz = np.zeros_like(ntu)
                group_count = self.num_token_groups if isinstance(self.num_token_groups, int) and self.num_token_groups > 0 else 5
                tokens = np.zeros((B, group_count), dtype=np.int32)
                vq_loss = 0.0
                base_recon_xyz = None
                residual_scale = None

            # accumulate
            recon_list.append(recon_xyz.astype(np.float32))
            tokens_list.append(np.array(tokens))
            
            if base_recon_xyz is not None:
                base_recon_list.append(base_recon_xyz.astype(np.float32))
            
            if residual_scale is not None:
                residual_scales.append(residual_scale)
            
            # vq_loss may be scalar; replicate per sample if so
            if np.isscalar(vq_loss):
                vq_list.append(np.full((ntu.shape[0],), float(vq_loss), dtype=np.float32))
            else:
                vq_list.append(np.array(vq_loss, dtype=np.float32))

            extracted_list.append(ntu.astype(np.float32))

            # metadata per sample
            for i in range(ntu.shape[0]):
                metadata.append({'index': start + i, 'src_file': str(input_npy)})

            # optional per-sample saving
            if per_sample:
                for i in range(ntu.shape[0]):
                    idx = start + i
                    per_path = os.path.join(out_dir, f'{split_name}_sample_{idx:06d}.npz')
                    save_dict = {
                        'reconstructed': recon_xyz[i].astype(np.float32),
                        'tokens': np.array(tokens[i]).astype(np.int32) if np.ndim(tokens) > 1 else np.array([int(tokens[i])], dtype=np.int32),
                        'vq_loss': float(vq_loss) if np.isscalar(vq_loss) else float(vq_loss[i]),
                        'extracted': ntu[i].astype(np.float32)
                    }
                    if base_recon_xyz is not None:
                        save_dict['base_reconstructed'] = base_recon_xyz[i].astype(np.float32)
                    np.savez_compressed(per_path, **save_dict)

        # concat
        reconstructed = np.concatenate(recon_list, axis=0)
        try:
            token_sequences = np.concatenate(tokens_list, axis=0)
        except Exception:
            # tokens may be ragged; fallback to object array (save as list)
            token_sequences = np.array(tokens_list, dtype=object)
        vq_losses = np.concatenate(vq_list, axis=0)
        extracted_all = np.concatenate(extracted_list, axis=0)
        
        # 合并 base_recon 和 residual_scale
        if base_recon_list:
            base_reconstructed = np.concatenate(base_recon_list, axis=0)
        else:
            base_reconstructed = None
        
        avg_residual_scale = np.mean(residual_scales) if residual_scales else None

        token_metadata_base = self.build_token_metadata()
        split_metadata = copy.deepcopy(token_metadata_base)
        if isinstance(token_sequences, np.ndarray):
            split_metadata['token_sequences_shape'] = list(token_sequences.shape)
        else:
            split_metadata['token_sequences_shape'] = None
        split_metadata['split_name'] = split_name
        split_metadata['num_samples'] = int(reconstructed.shape[0])

        analysis_payload = None
        analysis_path = None

        # 新增：Token 多样性分析
        if analyze_tokens:
            if isinstance(token_sequences, np.ndarray) and token_sequences.dtype != object:
                try:
                    token_analysis = self.analyze_token_diversity(token_sequences)
                    split_metadata['analysis_summary'] = token_analysis.get('summary', {})
                    split_metadata['analysis_generated_at'] = datetime.now().isoformat()

                    analysis_payload = {
                        'metadata': split_metadata,
                        'groups': token_analysis.get('groups', {}),
                        'summary': token_analysis.get('summary', {})
                    }

                    analysis_path = os.path.join(out_dir, f'{split_name}_token_analysis.json')
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_payload, f, ensure_ascii=False, indent=2)
                    print(f"✅ Token analysis saved to: {analysis_path}")
                except Exception as e:
                    print(f"⚠️ Token diversity analysis failed: {e}")
            else:
                print("⚠️ Token diversity analysis skipped due to non-uniform token array")
        
        # 打印重构质量统计
        print(f"\n{'='*80}")
        print("RECONSTRUCTION QUALITY:")
        print(f"{'='*80}")
        print(f"Average VQ Loss: {vq_losses.mean():.6f}")
        if avg_residual_scale is not None:
            print(f"Average Residual Scale: {avg_residual_scale:.6f}")
            if avg_residual_scale > 0.5:
                print("⚠️  High residual contribution - model may be bypassing codebook!")
            else:
                print("✓  Reasonable residual contribution")
        
        if base_reconstructed is not None:
            # 计算纯码本重建的误差 vs 最终重建的误差
            base_error = np.mean((extracted_all - base_reconstructed) ** 2)
            final_error = np.mean((extracted_all - reconstructed) ** 2)
            improvement = (base_error - final_error) / base_error * 100 if base_error > 0 else 0
            
            print(f"Codebook-only MSE: {base_error:.6f}")
            print(f"Final MSE (w/ residual): {final_error:.6f}")
            print(f"Residual improvement: {improvement:.1f}%")
            
            if improvement > 50:
                print("❌ Residual provides >50% improvement - codebook is weak!")
            elif improvement > 20:
                print("⚠️  Residual provides 20-50% improvement - codebook needs strengthening")
            else:
                print("✅ Residual provides <20% improvement - codebook is doing most of the work")

        # save single compressed file
        out_path = os.path.join(out_dir, f'{split_name}_recon.npz')
        print(f"\nSaving aggregated results to {out_path}")

        # metadata to json string
        meta_json = json.dumps(metadata)

        # 如果 token_sequences 是 object（长度不一），我们无法直接保存为数组，改为保存为 np.savez 的 list
        save_dict = {
            'reconstructed': reconstructed,
            'vq_losses': vq_losses,
            'extracted': extracted_all,
            'metadata': meta_json,
            'token_metadata': json.dumps(split_metadata, ensure_ascii=False)
        }
        
        if base_reconstructed is not None:
            save_dict['base_reconstructed'] = base_reconstructed
        
        if token_sequences.dtype == object:
            # 保存 tokens 单独为 .npy via pickle-friendly method
            tokens_path = os.path.join(out_dir, f'{split_name}_tokens.npy')
            print(f"Tokens are ragged; saving tokens as list to {tokens_path}")
            np.save(tokens_path, token_sequences, allow_pickle=True)
        else:
            save_dict['token_sequences'] = token_sequences.astype(np.int32)
        
        np.savez_compressed(out_path, **save_dict)

        print(f"✅ Saved: {out_path} (N={reconstructed.shape[0]})")

        # 更新全局token schema
        token_schema_path = os.path.join(out_dir, 'token_schema.json')
        schema_base = copy.deepcopy(token_metadata_base)

        if os.path.exists(token_schema_path):
            try:
                with open(token_schema_path, 'r', encoding='utf-8') as f:
                    schema_doc = json.load(f)
            except Exception:
                schema_doc = copy.deepcopy(schema_base)
                schema_doc['splits'] = {}
        else:
            schema_doc = copy.deepcopy(schema_base)
            schema_doc['splits'] = {}

        # 确保基础元数据与当前模型保持一致
        for key, value in schema_base.items():
            if key == 'generated_at':
                schema_doc.setdefault('generated_at', value)
            else:
                schema_doc[key] = value

        schema_doc.setdefault('splits', {})

        schema_doc['splits'][split_name] = {
            'num_samples': split_metadata.get('num_samples'),
            'token_sequences_shape': split_metadata.get('token_sequences_shape'),
            'analysis_summary': split_metadata.get('analysis_summary'),
            'analysis_file': os.path.relpath(analysis_path, out_dir) if analysis_path else None,
            'recon_file': os.path.relpath(out_path, out_dir),
            'updated_at': datetime.now().isoformat()
        }

        with open(token_schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema_doc, f, ensure_ascii=False, indent=2)
        
        # 生成CSV索引文件
        if per_sample and token_sequences.dtype != object:
            csv_path = os.path.join(out_dir, f'{split_name}_index.csv')
            print(f"Generating CSV index: {csv_path}")
            
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'split', 'file_path', 'tokens_str', 'token_first', 'vq_loss'])
                
                for i in range(len(token_sequences)):
                    sample_path = os.path.join(out_dir, f'{split_name}_sample_{i:06d}.npz')
                    tokens_str = str(token_sequences[i].tolist())
                    token_first = int(token_sequences[i][0])
                    vq_loss_val = float(vq_losses[i])
                    
                    writer.writerow([i, split_name, sample_path, tokens_str, token_first, vq_loss_val])
            
            print(f"✅ CSV index saved: {csv_path} ({len(token_sequences)} rows)")
        
        return out_path


def find_model_files():
    """自动查找可用的模型文件"""
    import glob
    
    # 查找提取器模型
    extractor_patterns = [
        'mars_optimized_best.pth',
        'mars_transformer_best*.pth',
        'mars_transformer*.pth',
        'models/mars_transformer*.pth',
        'checkpoints/mars_transformer*.pth'
    ]
    
    extractor_files = []
    for pattern in extractor_patterns:
        extractor_files.extend(glob.glob(pattern))
    
    # 查找GCN模型
    gcn_patterns = [
        'experiments/gcn_skeleton_memory_optimized_10p/NTU_models/adaptive_gcnskeleton_576tokens_balanced/ckpt-best.pth',
        'experiments/gcn_skeleton_memory_optimized/NTU_models/*/ckpt-best.pth',
        'experiments/*/ckpt-best.pth',
        'checkpoints/gcn*.pth'
    ]
    
    gcn_files = []
    for pattern in gcn_patterns:
        gcn_files.extend(glob.glob(pattern))
    
    # 查找GCN配置
    cfg_patterns = [
        'cfgs/NTU_models/gcn_skeleton_memory_optimized_10p.yaml',
        'cfgs/NTU_models/*.yaml',
        'configs/*.yaml'
    ]
    
    cfg_files = []
    for pattern in cfg_patterns:
        cfg_files.extend(glob.glob(pattern))
    
    return extractor_files, gcn_files, cfg_files


def main():
    parser = argparse.ArgumentParser(
        description='Batch reconstruct MARS datasets and save recon+tokens (no visualization).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动查找模型（交互式）
  python %(prog)s
  
  # 指定所有参数
  python %(prog)s --extractor mars_transformer_best_150.pth \\
                  --gcn_ckpt experiments/.../ckpt-best.pth \\
                  --gcn_cfg cfgs/NTU_models/gcn_skeleton_memory_optimized_10p.yaml
                  
  # 生成单样本文件（用于精细批注）
  python %(prog)s --per_sample
        """
    )
    
    # 默认路径
    default_extractor = 'mars_optimized_best.pth'
    default_gcn_ckpt = 'experiments/gcn_skeleton_memory_optimized_10p/NTU_models/adaptive_gcnskeleton_576tokens_balanced/ckpt-best.pth'
    default_gcn_cfg = 'cfgs/NTU_models/gcn_skeleton_memory_optimized_10p.yaml'
    
    parser.add_argument('--extractor', default=None, help=f'路径到 MARSTransformerModel 权重 (默认: {default_extractor})')
    parser.add_argument('--gcn_ckpt', default=None, help=f'路径到 GCN 重构器权重 (默认: {default_gcn_ckpt})')
    parser.add_argument('--gcn_cfg', default=None, help=f'路径到 GCN 配置 yaml (默认: {default_gcn_cfg})')
    parser.add_argument('--out_dir', default='data/MARS_recon_tokens', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--per_sample', action='store_true', help='是否为每个样本保存单独的 npz 文件')
    parser.add_argument('--use_enhanced_mapper', action='store_true', help='使用 EnhancedSkeletonMapper (默认 False)')
    parser.add_argument('--device', default=None, help='设备，如 cuda:0 或 cpu')
    parser.add_argument('--train', default='/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_train.npy')
    parser.add_argument('--test', default='/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy')
    parser.add_argument('--validate', default='/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_validate.npy')
    parser.add_argument('--auto', action='store_true', help='自动模式：使用默认值，不进行交互式确认')

    args = parser.parse_args()
    
    print("="*80)
    print("MARS Token Generation Pipeline")
    print("="*80)
    print()
    
    # 如果没有指定模型路径，尝试自动查找或使用默认值
    if args.extractor is None:
        if os.path.exists(default_extractor):
            args.extractor = default_extractor
            print(f"✓ 使用默认提取器: {args.extractor}")
        else:
            print("🔍 自动查找提取器模型...")
            extractor_files, _, _ = find_model_files()
            if extractor_files:
                print(f"找到 {len(extractor_files)} 个提取器模型:")
                for i, f in enumerate(extractor_files, 1):
                    print(f"  [{i}] {f}")
                
                if not args.auto:
                    choice = input(f"\n请选择模型 [1-{len(extractor_files)}] (回车使用第1个): ").strip()
                    idx = int(choice) - 1 if choice else 0
                    args.extractor = extractor_files[idx]
                else:
                    args.extractor = extractor_files[0]
                print(f"✓ 选择: {args.extractor}")
            else:
                print(f"❌ 未找到提取器模型，请使用 --extractor 参数指定")
                return
    
    if args.gcn_ckpt is None:
        if os.path.exists(default_gcn_ckpt):
            args.gcn_ckpt = default_gcn_ckpt
            print(f"✓ 使用默认GCN模型: {args.gcn_ckpt}")
        else:
            print("🔍 自动查找GCN模型...")
            _, gcn_files, _ = find_model_files()
            if gcn_files:
                print(f"找到 {len(gcn_files)} 个GCN模型:")
                for i, f in enumerate(gcn_files, 1):
                    print(f"  [{i}] {f}")
                
                if not args.auto:
                    choice = input(f"\n请选择模型 [1-{len(gcn_files)}] (回车使用第1个): ").strip()
                    idx = int(choice) - 1 if choice else 0
                    args.gcn_ckpt = gcn_files[idx]
                else:
                    args.gcn_ckpt = gcn_files[0]
                print(f"✓ 选择: {args.gcn_ckpt}")
            else:
                print(f"❌ 未找到GCN模型，请使用 --gcn_ckpt 参数指定")
                return
    
    if args.gcn_cfg is None:
        if os.path.exists(default_gcn_cfg):
            args.gcn_cfg = default_gcn_cfg
            print(f"✓ 使用默认GCN配置: {args.gcn_cfg}")
        else:
            print("🔍 自动查找GCN配置...")
            _, _, cfg_files = find_model_files()
            if cfg_files:
                print(f"找到 {len(cfg_files)} 个配置文件:")
                for i, f in enumerate(cfg_files, 1):
                    print(f"  [{i}] {f}")
                
                if not args.auto:
                    choice = input(f"\n请选择配置 [1-{len(cfg_files)}] (回车使用第1个): ").strip()
                    idx = int(choice) - 1 if choice else 0
                    args.gcn_cfg = cfg_files[idx]
                else:
                    args.gcn_cfg = cfg_files[0]
                print(f"✓ 选择: {args.gcn_cfg}")
            else:
                print(f"❌ 未找到GCN配置，请使用 --gcn_cfg 参数指定")
                return
    
    # 验证文件是否存在
    print()
    print("="*80)
    print("验证文件...")
    print("="*80)
    
    required_files = {
        '提取器模型': args.extractor,
        'GCN模型': args.gcn_ckpt,
        'GCN配置': args.gcn_cfg
    }
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
            print(f"✓ {name}: {path} ({size_str})")
        else:
            print(f"❌ {name}: {path} (不存在)")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 某些必需文件不存在，无法继续")
        return
    
    # 检查数据文件
    print()
    print("="*80)
    print("检查数据文件...")
    print("="*80)
    
    data_files = {
        'train': args.train,
        'test': args.test,
        'validate': args.validate
    }
    
    available_splits = []
    for split, path in data_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024/1024:.1f}MB"
            num_samples = len(np.load(path, mmap_mode='r'))
            print(f"✓ {split}: {path} ({size_str}, {num_samples} samples)")
            available_splits.append(split)
        else:
            print(f"⚠️ {split}: {path} (不存在，将跳过)")
    
    if not available_splits:
        print("\n❌ 未找到任何数据文件")
        return
    
    # 显示配置
    print()
    print("="*80)
    print("处理配置:")
    print("="*80)
    print(f"输出目录: {args.out_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"生成单样本文件: {'是' if args.per_sample else '否'}")
    print(f"使用增强映射器: {'是' if args.use_enhanced_mapper else '否'}")
    print(f"设备: {args.device if args.device else '自动检测'}")
    print(f"将处理的数据集: {', '.join(available_splits)}")
    
    # 确认
    if not args.auto:
        print()
        confirm = input("按回车键开始处理，或输入 'n' 取消: ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return
    
    print()
    print("="*80)
    print("开始处理...")
    print("="*80)
    print()

    try:
        saver = SkeletonReconstructionSaver(
            args.extractor,
            args.gcn_ckpt,
            args.gcn_cfg,
            device=args.device,
            use_enhanced_mapper=args.use_enhanced_mapper
        )
    except Exception as exc:
        print("\n❌ 初始化失败: 无法创建 SkeletonReconstructionSaver")
        print(f"原因: {exc}")
        return

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 依次处理 train/test/validate（如果存在）
    import time
    start_time = time.time()
    
    processed_splits = []
    for split_name, path in [('train', args.train), ('test', args.test), ('validate', args.validate)]:
        if path and os.path.exists(path):
            print(f"\n{'='*80}")
            print(f"Processing split: {split_name}")
            print(f"{'='*80}")
            saver.process_and_save(path, out_dir, split_name=split_name, batch_size=args.batch_size, 
                                  per_sample=args.per_sample, analyze_tokens=True)
            processed_splits.append(split_name)
        else:
            if path:
                print(f"\n⚠️ 跳过 {split_name}：文件不存在 {path}")
    
    # 如果生成了单样本文件，合并所有split的CSV索引
    if args.per_sample and processed_splits:
        print()
        print("="*80)
        print("生成合并CSV索引...")
        print("="*80)
        
        combined_csv_path = os.path.join(out_dir, 'index.csv')
        import csv
        
        with open(combined_csv_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['index', 'split', 'file_path', 'tokens_str', 'token_first', 'vq_loss'])
            
            for split_name in processed_splits:
                split_csv = os.path.join(out_dir, f'{split_name}_index.csv')
                if os.path.exists(split_csv):
                    with open(split_csv, 'r') as f_in:
                        reader = csv.reader(f_in)
                        next(reader)  # 跳过表头
                        for row in reader:
                            writer.writerow(row)
                    print(f"  ✓ 合并 {split_name}")
        
        print(f"✅ 合并CSV索引已保存: {combined_csv_path}")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print()
    print("="*80)
    print(f"✅ 处理完成！耗时: {minutes}分{seconds}秒")
    print("="*80)
    print()
    print("生成的文件:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
            print(f"  {f} ({size_str})")
    
    print()
    print("下一步:")
    print("  1. 查看Token分析: cat", os.path.join(out_dir, "*_token_analysis.json"))
    print("  2. 加载数据进行批注:")
    print(f"     python tools/example_load_mars_tokens.py")
    print()


if __name__ == '__main__':
    main()
