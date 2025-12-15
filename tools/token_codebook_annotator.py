#!/usr/bin/env python3
"""
Token Codebook标注工具
用于标注VQ-VAE生成的Token，为每个Token提供语义描述
"""

import os
import sys
import json
import numpy as np
import warnings
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_SEMANTIC_GROUPS_5 = {
    'head_spine': [0, 1, 2, 3, 20],
    'left_arm': [4, 5, 6, 7, 21, 22],
    'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg': [12, 13, 14, 15],
    'right_leg': [16, 17, 18, 19]
}


DEFAULT_GROUP_DISPLAY_NAMES = {
    'head_spine': {'zh': '头部脊柱', 'en': 'Head & Spine'},
    'head_neck': {'zh': '头颈', 'en': 'Head & Neck'},
    'spine': {'zh': '脊柱', 'en': 'Spine'},
    'left_arm': {'zh': '左上臂', 'en': 'Left Upper Arm'},
    'left_forearm': {'zh': '左前臂与手', 'en': 'Left Forearm & Hand'},
    'right_arm': {'zh': '右上臂', 'en': 'Right Upper Arm'},
    'right_forearm': {'zh': '右前臂与手', 'en': 'Right Forearm & Hand'},
    'left_leg': {'zh': '左大腿', 'en': 'Left Thigh'},
    'left_foot': {'zh': '左小腿与脚', 'en': 'Left Calf & Foot'},
    'right_leg': {'zh': '右大腿', 'en': 'Right Thigh'},
    'right_foot': {'zh': '右小腿与脚', 'en': 'Right Calf & Foot'}
}


DEFAULT_GROUP_TEMPLATES = {
    'head_spine': [
        "正常姿态", "抬头", "低头", "左侧转", "右侧转",
        "挺直站立", "前倾", "后仰", "左倾斜", "右倾斜"
    ],
    'head_neck': [
        "头部正视", "低头查看", "抬头观察", "向左侧倾", "向右侧倾",
        "快速点头", "缓慢摇头", "左顾", "右盼", "轻微扭头"
    ],
    'spine': [
        "脊柱挺直", "轻微弯腰", "前屈", "后伸", "躯干左倾",
        "躯干右倾", "微微含胸", "扩胸", "稳定站立", "核心收紧"
    ],
    'left_arm': [
        "自然垂落", "上举", "前伸", "侧举", "叉腰",
        "向内合拢", "向外展开", "快速摆动", "停于体侧", "防守姿态"
    ],
    'left_forearm': [
        "下垂放松", "抬起", "向内弯曲", "拿捏姿态", "握拳",
        "手心向上", "手心向外", "指向前方", "支撑动作", "微旋转"
    ],
    'right_arm': [
        "自然垂落", "上举", "前伸", "侧举", "叉腰",
        "向内合拢", "向外展开", "快速摆动", "停于体侧", "防守姿态"
    ],
    'right_forearm': [
        "下垂放松", "抬起", "向内弯曲", "握拳", "手掌展开",
        "手心向上", "手心向外", "指向目标", "支撑姿态", "微旋转"
    ],
    'left_leg': [
        "站立稳固", "大腿前抬", "大腿后摆", "转髋外展", "准备跨步",
        "弓步支撑", "快速收回", "向侧展开", "提膝", "跨步起势"
    ],
    'left_foot': [
        "脚掌着地", "脚尖点地", "脚跟抬起", "向前伸展", "向后撤离",
        "侧向滑步", "蹬地发力", "缓慢蹲降", "踮脚", "收脚回位"
    ],
    'right_leg': [
        "站立稳固", "大腿前抬", "大腿后摆", "转髋外展", "准备跨步",
        "弓步支撑", "快速收回", "向侧展开", "提膝", "跨步起势"
    ],
    'right_foot': [
        "脚掌着地", "脚尖点地", "脚跟抬起", "向前伸展", "向后撤离",
        "侧向滑步", "蹬地发力", "缓慢蹲降", "踮脚", "收脚回位"
    ]
}


GENERIC_TEMPLATES = [
    "默认姿态", "轻微调整", "自然放松", "快速收回", "准备动作",
    "保持稳定", "渐进施力", "平衡控制", "动作预备", "放松回位"
]

# 禁用字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# 配置matplotlib使用基本字体（避免中文字体问题）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 尝试导入可视化窗口（如果可用）
try:
    from tools.visualization_window import SkeletonVisualizationWindow
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ 可视化窗口不可用，将使用matplotlib基础可视化")
    VISUALIZATION_AVAILABLE = False


class TokenCodebookAnnotator:
    """Token Codebook标注工具"""
    
    def __init__(self):
        """初始化标注工具"""
        self.project_root = project_root
        self.token_analysis_dir = self.project_root / "token_analysis"
        self.codebook_path = self.token_analysis_dir / "codebook_annotation_template.json"
        self.output_path = self.token_analysis_dir / "codebook_annotations.json"
        
        # 加载动态语义组配置（兼容10部位与旧5部位）
        self.token_schema_info = self.load_token_schema()
        self.token_schema_path = self.token_schema_info.get('source_path')
        self.tokenizer_model = self.token_schema_info.get('tokenizer_model')
        self.semantic_groups = self.token_schema_info['semantic_groups']
        self.group_offsets = self.token_schema_info['group_offsets']
        self.group_token_sizes = self.token_schema_info['group_token_sizes']
        self.tokens_config = self.token_schema_info['tokens_config']
        self.part_names = self.token_schema_info['group_order']
        self.part_display_map = self.token_schema_info['display_map']
        self.part_display_names = [self.part_display_map[name]['zh'] for name in self.part_names]
        self.mars_action_templates = self._build_action_templates(self.part_names)
        self.default_tokens_per_group = self.token_schema_info.get('default_tokens_per_group', 128)
        
        # 加载MARS Token数据集
        self.dataset = None
        self.load_mars_token_dataset()
        
        # 加载Token模板（需要在数据集加载后，因为可能需要生成模板）
        self.load_token_template()
        
        # 注意：不需要加载VQ-VAE模型，因为我们直接使用数据集中的真实骨架样本进行可视化
        
        # 当前标注状态
        self.current_annotations = {}
        self.load_existing_annotations()
    
    @staticmethod
    def _title_case(name: str) -> str:
        return name.replace('_', ' ').title()

    def _format_display_name(self, group_name: str, lang: str = 'zh') -> str:
        defaults = DEFAULT_GROUP_DISPLAY_NAMES.get(group_name)
        if defaults:
            candidate = defaults.get(lang)
            if candidate:
                return candidate
            alt = defaults.get('en' if lang == 'zh' else 'zh')
            if alt:
                return alt
        title = self._title_case(group_name)
        return title

    def _build_action_templates(self, group_names: List[str]) -> Dict[str, List[str]]:
        templates = {}
        for name in group_names:
            base = DEFAULT_GROUP_TEMPLATES.get(name, GENERIC_TEMPLATES)
            templates[name] = list(base)
        return templates

    def load_token_schema(self) -> Dict:
        """加载 token_schema.json，兼容10部位与旧5部位配置"""

        schema_candidates = [
            Path("/home/uo/myProject/CRSkeleton/data/MARS_recon_tokens/token_schema.json"),
            Path("/home/uo/myProject/HumanPoint-BERT/data/MARS_recon_tokens/token_schema.json"),
            self.project_root / "data" / "MARS_recon_tokens" / "token_schema.json",
            self.token_analysis_dir / "token_schema.json"
        ]

        schema_data = None
        schema_path = None

        for candidate in schema_candidates:
            candidate = Path(candidate)
            if candidate.is_dir():
                candidate = candidate / "token_schema.json"
            if candidate.exists():
                try:
                    with open(candidate, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                    schema_path = candidate
                    print(f"✅ 加载Token Schema: {candidate}")
                    break
                except Exception as exc:
                    print(f"⚠️ 无法读取 token_schema.json ({candidate}): {exc}")

        if schema_data is None:
            print("⚠️ 未找到 token_schema.json，使用默认5部位配置")
            group_order = list(DEFAULT_SEMANTIC_GROUPS_5.keys())
            group_token_sizes = [128] * len(group_order)
            group_offsets = []
            offset = 0
            for size in group_token_sizes:
                group_offsets.append(offset)
                offset += size

            display_map = {}
            for name in group_order:
                display_map[name] = {
                    'zh': self._format_display_name(name, 'zh'),
                    'en': self._format_display_name(name, 'en')
                }

            return {
                'semantic_groups': {name: list(indices) for name, indices in DEFAULT_SEMANTIC_GROUPS_5.items()},
                'group_order': group_order,
                'group_offsets': group_offsets,
                'group_token_sizes': group_token_sizes,
                'tokens_config': {name: 128 for name in group_order},
                'display_map': display_map,
                'default_tokens_per_group': 128,
                'tokenizer_model': 'GCNSkeletonTokenizer',
                'source_path': None,
                'raw_metadata': {}
            }

        group_order = schema_data.get('group_order')
        group_joint_indices = schema_data.get('group_joint_indices') or schema_data.get('semantic_groups') or {}
        if not group_order:
            group_order = list(group_joint_indices.keys()) or list(DEFAULT_SEMANTIC_GROUPS_5.keys())

        default_tokens_per_group = int(schema_data.get('default_tokens_per_group', 128))

        tokens_config_raw = schema_data.get('tokens_config', {}) or {}
        if hasattr(tokens_config_raw, 'items'):
            tokens_config = {str(k): int(v) for k, v in tokens_config_raw.items()}
        else:
            tokens_config = {str(k): int(v) for k, v in tokens_config_raw} if tokens_config_raw else {}

        semantic_groups = {}
        for name in group_order:
            joints = group_joint_indices.get(name)
            if joints is None and name in DEFAULT_SEMANTIC_GROUPS_5:
                joints = DEFAULT_SEMANTIC_GROUPS_5[name]
            if joints is None:
                joints = []
            semantic_groups[name] = [int(j) for j in joints]

        group_token_sizes = schema_data.get('group_token_sizes')
        if group_token_sizes is None:
            group_token_sizes = []
            for name in group_order:
                size = tokens_config.get(name)
                if size is None:
                    size = default_tokens_per_group
                group_token_sizes.append(int(size))
        else:
            group_token_sizes = [int(v) for v in group_token_sizes]

        group_offsets = schema_data.get('group_offsets')
        if group_offsets is None:
            group_offsets = []
            offset = 0
            for size in group_token_sizes:
                group_offsets.append(offset)
                offset += int(size)
        else:
            group_offsets = [int(v) for v in group_offsets]

        display_source = schema_data.get('group_display_names', {})
        display_map = {}
        for name in group_order:
            info = display_source.get(name, {}) if isinstance(display_source, dict) else {}
            zh_name = info.get('zh') or info.get('cn') or self._format_display_name(name, 'zh')
            en_name = info.get('en') or info.get('english') or self._format_display_name(name, 'en')
            display_map[name] = {'zh': zh_name, 'en': en_name}

        final_tokens_config = {}
        for idx, name in enumerate(group_order):
            final_tokens_config[name] = int(tokens_config.get(name, group_token_sizes[idx]))

        return {
            'semantic_groups': semantic_groups,
            'group_order': group_order,
            'group_offsets': group_offsets,
            'group_token_sizes': group_token_sizes,
            'tokens_config': final_tokens_config,
            'display_map': display_map,
            'default_tokens_per_group': default_tokens_per_group,
            'tokenizer_model': schema_data.get('tokenizer_model'),
            'source_path': str(schema_path) if schema_path else None,
            'raw_metadata': schema_data
        }

    def load_token_template(self):
        """加载Token模板"""
        if not self.codebook_path.exists():
            print(f"⚠️ 模板文件不存在，正在生成...")
            self.generate_token_template()
        
        with open(self.codebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token_template = data['codebook_annotation']
            self.metadata = data.get('metadata', {}) or {}

        # 对齐元数据与最新schema
        self.metadata.setdefault('group_order', self.part_names)
        self.metadata.setdefault('group_offsets', self.group_offsets)
        self.metadata.setdefault('group_token_sizes', self.group_token_sizes)
        self.metadata.setdefault('tokens_config', self.tokens_config)
        self.metadata.setdefault('group_display_names', self.part_display_map)
        self.metadata.setdefault('default_tokens_per_group', self.default_tokens_per_group)
        self.metadata.setdefault('total_token_vocab', int(sum(self.group_token_sizes)))
        self.metadata.setdefault('schema_version', '1.1.0')
        self.metadata['tokenizer_model'] = self.tokenizer_model
        self.metadata['token_schema_path'] = self.token_schema_path
        if 'total_unique_tokens' not in self.metadata:
            self.metadata['total_unique_tokens'] = sum(len(tokens) for tokens in self.token_template.values())
        
        print(f"✅ 加载Token模板: {self.metadata['total_unique_tokens']} 个Token")
    
    def generate_token_template(self):
        """生成Token模板（从数据集中提取唯一Token）"""
        if self.dataset is None:
            print("❌ 无法生成模板：数据集未加载")
            sys.exit(1)
        
        # 创建目录
        self.token_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计所有唯一Token及其出现次数
        token_counts = {part: {} for part in self.part_names}
        total_samples = len(self.dataset)
        
        print(f"📊 分析 {total_samples} 个样本，提取唯一Token...")
        for idx, sample in enumerate(self.dataset):
            tokens = sample['tokens']
            for i, part in enumerate(self.part_names):
                token_id = int(tokens[i])
                token_counts[part][token_id] = token_counts[part].get(token_id, 0) + 1
            
            # 显示进度
            if (idx + 1) % 5000 == 0:
                print(f"   已分析: {idx + 1}/{total_samples} ({(idx+1)/total_samples*100:.1f}%)")
        
        # 创建模板并生成统计信息
        template = {}
        total_tokens = 0
        
        print(f"\n✨ Token统计:")
        for part, display_name in zip(self.part_names, self.part_display_names):
            template[part] = {}
            sorted_tokens = sorted(token_counts[part].items(), key=lambda x: x[1], reverse=True)
            
            print(f"  {display_name}: {len(sorted_tokens)} 个唯一Token")
            for token_id, count in sorted_tokens[:3]:  # 显示前3个最常见的
                print(f"    - Token {token_id}: {count} 次 ({count/total_samples*100:.2f}%)")
            
            for token_id, count in sorted_tokens:
                template[part][str(token_id)] = f"[待标注] Token{token_id}"
                total_tokens += 1
        
        # 保存模板
        if not self.group_offsets:
            offsets = []
            offset_val = 0
            for size in self.group_token_sizes:
                offsets.append(offset_val)
                offset_val += size
            self.group_offsets = offsets

        total_vocab = sum(self.group_token_sizes) if self.group_token_sizes else len(self.part_names) * self.default_tokens_per_group

        metadata_payload = {
            'schema_version': '1.1.0',
            'total_samples': total_samples,
            'total_unique_tokens': total_tokens,
            'estimated_annotation_time_hours': total_tokens * 0.5 / 60,
            'token_counts': {part: dict(sorted(counts.items())) for part, counts in token_counts.items()},
            'created_at': datetime.now().isoformat(),
            'group_order': self.part_names,
            'group_offsets': self.group_offsets,
            'group_token_sizes': self.group_token_sizes,
            'group_display_names': self.part_display_map,
            'tokens_config': self.tokens_config,
            'default_tokens_per_group': self.default_tokens_per_group,
            'total_token_vocab': int(total_vocab),
            'tokenizer_model': self.tokenizer_model,
            'token_schema_path': self.token_schema_path
        }

        output_data = {
            'codebook_annotation': template,
            'metadata': metadata_payload
        }
        
        with open(self.codebook_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 生成模板: {self.codebook_path}")
        print(f"   总计 {total_tokens} 个唯一Token")
        print(f"   预计标注时间: {total_tokens * 0.5:.1f} 分钟")

        self.token_template = template
        self.metadata = metadata_payload
        
    def load_existing_annotations(self):
        """加载已有标注"""
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current_annotations = data.get('codebook_annotation', {})
            print(f"✅ 加载已有标注: {self.count_annotated()} / {self.metadata['total_unique_tokens']}")
        else:
            self.current_annotations = {part: {} for part in self.part_names}
            print("📝 初始化新的标注文件")

        self._normalize_annotation_groups()
    
    def load_mars_token_dataset(self):
        """加载MARS Token数据集（所有样本）"""
        print("📊 加载MARS Token数据集...")
        
        data_dir_paths = [
            "/home/uo/myProject/CRSkeleton/data/MARS_recon_tokens",
            "/home/uo/myProject/HumanPoint-BERT/data/MARS_recon_tokens",
            self.project_root / "data" / "MARS_recon_tokens"
        ]
        
        for dir_path in data_dir_paths:
            dir_path = Path(dir_path)
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # 加载所有train样本（不限制数量）
                    npz_files = list(dir_path.glob("train_sample_*.npz"))
                    
                    if not npz_files:
                        # 如果没有train样本，尝试validate样本
                        npz_files = list(dir_path.glob("validate_sample_*.npz"))
                    
                    if not npz_files:
                        continue
                    
                    npz_files = sorted(npz_files)
                    total_files = len(npz_files)
                    
                    print(f"   发现 {total_files} 个样本文件，开始加载...")
                    
                    self.dataset = []
                    for i, npz_file in enumerate(npz_files):
                        try:
                            data = np.load(npz_file, allow_pickle=True)
                            
                            # 优先使用纯码本重构（base_reconstructed），如果没有则用reconstructed
                            if 'base_reconstructed' in data:
                                skeleton_data = data['base_reconstructed']  # 纯码本重构（无残差）
                            else:
                                skeleton_data = data['reconstructed']  # 含残差的重构
                            
                            self.dataset.append({
                                'index': len(self.dataset),
                                'tokens': data['tokens'],  # (5,) 的token序列
                                'skeleton': skeleton_data,  # (25, 3) 的骨架坐标
                                'file': npz_file.name
                            })
                            
                            # 每加载1000个显示进度
                            if (i + 1) % 1000 == 0:
                                print(f"   已加载: {i + 1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
                                
                        except Exception as e:
                            continue
                    
                    print(f"✅ 成功加载数据集: {dir_path}")
                    print(f"   总样本数: {len(self.dataset)}")
                    
                    if len(self.dataset) > 0:
                        sample = self.dataset[0]
                        print(f"   示例Token: {sample['tokens']}")
                        print(f"   骨架形状: {sample['skeleton'].shape}")
                    
                    return
                    
                except Exception as e:
                    print(f"⚠️ 加载失败 {dir_path}: {e}")
        
        print("❌ 未找到MARS Token数据集，某些功能可能不可用")
        self.dataset = None
    
    def count_annotated(self) -> int:
        """统计已标注Token数量"""
        count = 0
        for part in self.part_names:
            if part in self.current_annotations:
                for token_id, annotation in self.current_annotations[part].items():
                    if not annotation.startswith("[待标注]"):
                        count += 1
        return count
    
    def get_samples_with_token(self, body_part: str, token_id: int, max_samples: int = 5) -> List[Dict]:
        """获取包含指定Token的样本"""
        if self.dataset is None or len(self.dataset) == 0:
            return []
        
        part_index = self.part_names.index(body_part)
        samples = []
        
        try:
            for sample in self.dataset:
                if len(samples) >= max_samples:
                    break
                
                tokens = sample['tokens']
                
                # 检查是否匹配Token
                if len(tokens) > part_index and int(tokens[part_index]) == token_id:
                    samples.append(sample)
        
        except Exception as e:
            print(f"⚠️ 搜索样本时出错: {e}")
        
        return samples
    
    def visualize_token_samples(self, body_part: str, token_id: int):
        """可视化包含指定Token的样本"""
        samples = self.get_samples_with_token(body_part, token_id, max_samples=5)
        
        if not samples:
            print(f"⚠️ 未找到包含 Token {token_id} 的样本")
            print(f"   尝试使用模型解码可视化...")
            
            # 如果有VQ-VAE模型，尝试解码可视化
            if self.vqvae_model is not None:
                self.visualize_token_from_model(body_part, token_id)
            else:
                print(f"   提示: Token {token_id} ({self.part_display_names[self.part_names.index(body_part)]}) 存在但暂无样本")
                print(f"   请根据身体部位描述进行标注")
            return
        
        part_display_info = self.part_display_map.get(body_part, {'en': self._title_case(body_part)})
        part_name_en = part_display_info.get('en', self._title_case(body_part))
        
        print(f"\n📊 Found {len(samples)} samples with Token {token_id} ({part_name_en})")
        
        # 使用matplotlib可视化（非阻塞模式）
        num_samples = min(len(samples), 3)
        fig = plt.figure(figsize=(5 * num_samples, 5))
        
        for i, sample in enumerate(samples[:num_samples]):
            ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
            
            skeleton = sample['skeleton']
            if skeleton is not None:
                # 处理不同的skeleton格式
                if len(skeleton.shape) == 3:  # (C, T, V) 或 (T, V, C)
                    # 取第一帧
                    if skeleton.shape[-1] == 25:  # (T, V, C)
                        skeleton = skeleton[0]  # 取第一帧
                    elif skeleton.shape[0] == 3:  # (C, T, V)
                        skeleton = skeleton[:, 0, :].T  # 转换为 (V, C)
                    else:  # (T, V, C) 其他情况
                        skeleton = skeleton[0]
                
                elif len(skeleton.shape) == 2:  # (V, C)
                    pass  # 已经是正确格式
                
                if skeleton.shape[0] == 25 and skeleton.shape[1] == 3:
                    self._plot_skeleton(ax, skeleton, body_part)
                    ax.set_title(f"Sample {sample['index']}\nTokens: {sample['tokens']}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, 0.5, f"Invalid shape\n{skeleton.shape}", ha='center', va='center')
                    ax.set_title(f"Sample {sample['index']}")
            else:
                ax.text(0.5, 0.5, 0.5, "No skeleton data", ha='center', va='center')
                ax.set_title(f"Sample {sample['index']}")
        
        # 不设置窗口标题，避免中文字体警告
        # fig.canvas.manager.set_window_title(f'Token {token_id} - {part_name_en}')
        
        plt.tight_layout()
        plt.ion()  # 开启交互模式
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.1)
        
        print("\n💡 Window opened (non-blocking). You can close it manually or continue annotation.")
    
    def visualize_token_from_model(self, body_part: str, token_id: int):
        """使用VQ-VAE模型解码Token进行可视化（开发中）"""
        print(f"   模型解码功能开发中...")
        # TODO: 实现从单个Token解码的功能
        # 这需要对VQ-VAE模型进行修改，支持部分解码
    
    def _plot_skeleton(self, ax, skeleton, highlight_part: str = None):
        """绘制骨架（改进版，使用GCNSkeletonTokenizer的连接和分组）"""
        # 坐标转换: (x, y, z) → (x, y, -z)
        skeleton = skeleton.copy()
        skeleton[:, 2] = -skeleton[:, 2]  # 翻转Z轴
        
        # NTU RGB+D骨架连接（来自GCNSkeletonTokenizer）
        connections = [
            # 头部和脊柱
            (3, 2), (2, 20), (20, 1), (1, 0),
            # 左臂
            (20, 4), (4, 5), (5, 6), (6, 22), (6, 7), (7, 21),
            # 右臂
            (20, 8), (8, 9), (9, 10), (10, 24), (10, 11), (11, 23),
            # 左腿
            (0, 12), (12, 13), (13, 14), (14, 15),
            # 右腿
            (0, 16), (16, 17), (17, 18), (18, 19)
        ]
        
        # 获取当前部位的关节索引
        highlight_joints = set(self.semantic_groups.get(highlight_part, [])) if highlight_part else set()
        
        # 绘制连接线
        for conn in connections:
            i, j = conn[0], conn[1]
            if i < len(skeleton) and j < len(skeleton):
                points = skeleton[[i, j]]
                
                # 判断连接是否属于高亮部位
                if highlight_part and (i in highlight_joints or j in highlight_joints):
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                             'r-', linewidth=3, alpha=0.9)
                else:
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                             'b-', linewidth=1, alpha=0.3)
        
        # 绘制关节点
        for i, point in enumerate(skeleton):
            if i in highlight_joints:
                # 高亮当前部位的关节
                ax.scatter(point[0], point[1], point[2], 
                          c='red', s=100, marker='o', edgecolors='darkred', linewidths=2)
            else:
                # 其他关节
                ax.scatter(point[0], point[1], point[2], 
                          c='lightblue', s=30, marker='o', alpha=0.4)
        
        # 添加关节编号（仅高亮部位）
        if highlight_part:
            for i in highlight_joints:
                if i < len(skeleton):
                    point = skeleton[i]
                    ax.text(point[0], point[1], point[2], f' {i}', 
                           fontsize=8, color='darkred', weight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Up)')
        ax.set_zlabel('Z')
        
        # 设置相同的坐标轴范围
        max_range = np.array([
            skeleton[:, 0].max() - skeleton[:, 0].min(),
            skeleton[:, 1].max() - skeleton[:, 1].min(),
            skeleton[:, 2].max() - skeleton[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (skeleton[:, 0].max() + skeleton[:, 0].min()) * 0.5
        mid_y = (skeleton[:, 1].max() + skeleton[:, 1].min()) * 0.5
        mid_z = (skeleton[:, 2].max() + skeleton[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 设置视角：Y轴向上（重要！）
        ax.view_init(elev=15, azim=45)
        
        # 添加标注说明Y轴是垂直方向（头在上）
        ax.text2D(0.02, 0.98, f"Y↑(Head up)", transform=ax.transAxes, 
                 fontsize=10, color='red', weight='bold', va='top')
    
    def annotate_token(self, body_part: str, token_id: int, auto_visualize: bool = False):
        """标注单个Token
        
        Args:
            body_part: 身体部位
            token_id: Token ID
            auto_visualize: 如果为True，跳过询问直接进入标注（可视化已在外部显示）
        """
        part_idx = self.part_names.index(body_part)
        part_display = self.part_display_names[part_idx]
        
        print(f"\n{'='*60}")
        print(f"📝 标注 Token {token_id} ({part_display})")
        print(f"{'='*60}")
        
        # 检查是否已标注
        if body_part in self.current_annotations and str(token_id) in self.current_annotations[body_part]:
            existing = self.current_annotations[body_part][str(token_id)]
            if not existing.startswith("[待标注]"):
                print(f"当前标注: {existing}")
                overwrite = input("是否覆盖现有标注? (y/n): ").strip().lower()
                if overwrite != 'y':
                    return
        
        # 如果不是自动模式，询问是否显示可视化
        if not auto_visualize:
            show_samples = input("是否显示包含此Token的样本? (y/n/c=关闭所有窗口, 默认y): ").strip().lower()
            if show_samples == 'c':
                plt.close('all')
                print("✅ 已关闭所有可视化窗口")
            elif show_samples == 'y':
                self.visualize_token_samples(body_part, token_id)
        
        # 显示快速选项
        print(f"\n快速选择 ({part_display}):")
        templates = self.mars_action_templates[body_part]
        for i, action in enumerate(templates, 1):
            print(f"  {i}. {action}")
        print("  0. 自定义输入")
        
        choice = input(f"\n选择 (1-{len(templates)}, 0=自定义): ").strip()
        
        if choice == '0':
            annotation = input("请输入自定义标注: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(templates):
            annotation = templates[int(choice) - 1]
            
            # 允许添加详细描述
            add_detail = input(f"是否添加更详细的描述? (y/n, 默认n): ").strip().lower()
            if add_detail == 'y':
                detail = input("详细描述: ").strip()
                annotation = f"{annotation}（{detail}）"
        else:
            print("❌ 无效选择，跳过")
            return
        
        # 保存标注
        if body_part not in self.current_annotations:
            self.current_annotations[body_part] = {}
        
        self.current_annotations[body_part][str(token_id)] = annotation
        print(f"✅ 已标注: Token {token_id} = {annotation}")
        
        # 自动保存
        self.save_annotations()

    def _normalize_annotation_groups(self):
        """确保标注结构与当前语义组对齐"""
        normalized = {part: {} for part in self.part_names}
        for part, annotations in (self.current_annotations or {}).items():
            if part in normalized and isinstance(annotations, dict):
                normalized[part] = dict(annotations)
        self.current_annotations = normalized
    
    def annotate_body_part(self, body_part: str):
        """标注某个身体部位的所有Token"""
        part_idx = self.part_names.index(body_part)
        part_display = self.part_display_names[part_idx]
        
        tokens = list(self.token_template[body_part].keys())
        tokens = [int(t) for t in tokens]
        tokens.sort()
        
        print(f"\n{'='*60}")
        print(f"📝 标注身体部位: {part_display}")
        print(f"   共 {len(tokens)} 个Token")
        print(f"{'='*60}")
        
        for i, token_id in enumerate(tokens, 1):
            print(f"\n进度: {i}/{len(tokens)}")
            
            # 自动显示可视化
            self.visualize_token_samples(body_part, token_id)
            
            # 标注Token
            self.annotate_token(body_part, token_id, auto_visualize=True)
            
            if i < len(tokens):
                continue_choice = input("\n继续下一个Token? (y/n/q=退出, 默认y): ").strip().lower()
                if continue_choice == 'q':
                    print("退出标注")
                    break
                elif continue_choice == 'n':
                    return
    
    def save_annotations(self):
        """保存标注结果"""
        annotated = self.count_annotated()

        self.metadata['total_samples'] = self.metadata.get('total_samples', 0)
        self.metadata['total_unique_tokens'] = self.metadata.get('total_unique_tokens', sum(len(tokens) for tokens in self.current_annotations.values()))
        self.metadata['annotated_tokens'] = annotated
        self.metadata['annotation_progress'] = f"{annotated}/{self.metadata['total_unique_tokens']}"
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['group_order'] = self.part_names
        self.metadata['group_offsets'] = self.group_offsets
        self.metadata['group_token_sizes'] = self.group_token_sizes
        self.metadata['group_display_names'] = self.part_display_map
        self.metadata['tokens_config'] = self.tokens_config
        self.metadata['default_tokens_per_group'] = self.default_tokens_per_group
        self.metadata['total_token_vocab'] = self.metadata.get('total_token_vocab', int(sum(self.group_token_sizes)))
        self.metadata['tokenizer_model'] = self.tokenizer_model
        self.metadata['token_schema_path'] = self.token_schema_path

        output_data = {
            'codebook_annotation': self.current_annotations,
            'metadata': self.metadata
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 已保存标注: {self.output_path}")
    
    def show_progress(self):
        """显示标注进度"""
        print(f"\n{'='*60}")
        print(f"📊 标注进度统计")
        print(f"{'='*60}")
        
        total = self.metadata['total_unique_tokens']
        annotated = self.count_annotated()
        percentage = (annotated / total * 100) if total > 0 else 0
        
        print(f"总体进度: {annotated}/{total} ({percentage:.1f}%)")
        print(f"剩余Token: {total - annotated}")
        
        print(f"\n各部位进度:")
        for part, display_name in zip(self.part_names, self.part_display_names):
            part_tokens = list(self.token_template[part].keys())
            part_annotated = sum(
                1 for tid in part_tokens
                if part in self.current_annotations 
                and str(tid) in self.current_annotations[part]
                and not self.current_annotations[part][str(tid)].startswith("[待标注]")
            )
            part_total = len(part_tokens)
            part_pct = (part_annotated / part_total * 100) if part_total > 0 else 0
            
            print(f"  {display_name:8s}: {part_annotated:2d}/{part_total:2d} ({part_pct:5.1f}%)")
        
        print(f"{'='*60}\n")
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print(f"\n{'='*60}")
            print("Token Codebook 标注工具")
            print(f"{'='*60}")
            print("1. 标注头部脊柱 Token")
            print("2. 标注左臂 Token")
            print("3. 标注右臂 Token")
            print("4. 标注左腿 Token")
            print("5. 标注右腿 Token")
            print("6. 标注单个Token")
            print("7. 查看标注进度")
            print("8. 导出标注结果")
            print("9. 查看Token样本")
            print("c. 关闭所有可视化窗口")
            print("0. 退出")
            print(f"{'='*60}")
            
            choice = input("请选择 (0-9/c): ").strip().lower()
            
            if choice == '1':
                self.annotate_body_part('head_spine')
            elif choice == '2':
                self.annotate_body_part('left_arm')
            elif choice == '3':
                self.annotate_body_part('right_arm')
            elif choice == '4':
                self.annotate_body_part('left_leg')
            elif choice == '5':
                self.annotate_body_part('right_leg')
            elif choice == '6':
                self._annotate_single_token_menu()
            elif choice == '7':
                self.show_progress()
            elif choice == '8':
                self.save_annotations()
                print("✅ 标注结果已保存")
            elif choice == '9':
                self._view_token_samples_menu()
            elif choice == 'c':
                plt.close('all')
                print("✅ 已关闭所有可视化窗口")
            elif choice == '0':
                print("保存并退出...")
                self.save_annotations()
                plt.close('all')  # 关闭所有窗口
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择")
    
    def _annotate_single_token_menu(self):
        """标注单个Token的菜单"""
        print("\n选择身体部位:")
        for i, name in enumerate(self.part_display_names, 1):
            print(f"  {i}. {name}")
        
        part_choice = input("选择部位 (1-5): ").strip()
        if not part_choice.isdigit() or not (1 <= int(part_choice) <= 5):
            print("❌ 无效选择")
            return
        
        body_part = self.part_names[int(part_choice) - 1]
        tokens = list(self.token_template[body_part].keys())
        
        print(f"\n{self.part_display_names[int(part_choice) - 1]} 的Token:")
        print(", ".join(tokens))
        
        token_id = input("\n输入Token ID: ").strip()
        if token_id not in tokens:
            print(f"❌ Token {token_id} 不在该部位中")
            return
        
        self.annotate_token(body_part, int(token_id))
    
    def _view_token_samples_menu(self):
        """查看Token样本的菜单"""
        print("\n选择身体部位:")
        for i, name in enumerate(self.part_display_names, 1):
            print(f"  {i}. {name}")
        
        part_choice = input("选择部位 (1-5): ").strip()
        if not part_choice.isdigit() or not (1 <= int(part_choice) <= 5):
            print("❌ 无效选择")
            return
        
        body_part = self.part_names[int(part_choice) - 1]
        tokens = list(self.token_template[body_part].keys())
        
        print(f"\n{self.part_display_names[int(part_choice) - 1]} 的Token:")
        print(", ".join(tokens))
        
        token_id = input("\n输入Token ID: ").strip()
        if token_id not in tokens:
            print(f"❌ Token {token_id} 不在该部位中")
            return
        
        self.visualize_token_samples(body_part, int(token_id))


def main():
    """主函数"""
    print("🚀 启动 Token Codebook 标注工具")
    
    annotator = TokenCodebookAnnotator()
    annotator.show_progress()
    annotator.interactive_menu()


if __name__ == "__main__":
    main()
