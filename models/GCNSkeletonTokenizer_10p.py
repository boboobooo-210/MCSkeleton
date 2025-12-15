"""
时空图卷积骨架Tokenizer
基于25个关节点的直接处理，使用语义分组和时空图卷积
不再需要720点的密集化扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 导入模型注册器
try:
    from .build import MODELS
except ImportError:
    # 如果作为独立脚本运行，创建一个简单的注册器
    class SimpleRegistry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator
    MODELS = SimpleRegistry()

class SkeletonGraph:
    """NTU RGB+D 25关节点的骨架图结构定义"""
    
    def __init__(self):
        # NTU RGB+D 25关节点连接关系 (修正后的正确连接顺序)
        self.skeleton_edges = [
            # 头部和脊柱连接
            (3, 2),   # 头顶点 → 颈椎上段
            (2, 20),  # 颈椎上段 → 锁骨/肩关节区域
            (20, 1),  # 锁骨/肩关节区域 → 胸椎段
            (1, 0),   # 胸椎段 → 腰椎/骨盆衔接处
            
            # 左臂连接 (从肩膀到手)
            (20, 4),  # 锁骨/肩关节区域 → 左上臂关节
            (4, 5),   # 左上臂关节 → 左肘关节
            (5, 6),   # 左肘关节 → 左腕关节
            (6, 22),  # 左腕关节 → 左手指近端关节
            (6, 7),   # 左腕关节 → 左手掌关节
            (7, 21),  # 左手掌关节 → 左手指远端关节
            
            # 右臂连接 (从肩膀到手)
            (20, 8),  # 锁骨/肩关节区域 → 右上臂关节
            (8, 9),   # 右上臂关节 → 右肘关节
            (9, 10),  # 右肘关节 → 右腕关节
            (10, 24), # 右腕关节 → 右手指近端关节
            (10, 11), # 右腕关节 → 右手掌关节
            (11, 23), # 右手掌关节 → 右手指远端关节
            
            # 左腿连接 (从骨盆到脚)
            (0, 12),  # 腰椎/骨盆衔接处 → 左髋关节
            (12, 13), # 左髋关节 → 左膝关节
            (13, 14), # 左膝关节 → 左踝关节
            (14, 15), # 左踝关节 → 左脚趾近端关节
            
            # 右腿连接 (从骨盆到脚)
            (0, 16),  # 腰椎/骨盆衔接处 → 右髋关节
            (16, 17), # 右髋关节 → 右膝关节
            (17, 18), # 右膝关节 → 右踝关节
            (18, 19), # 右踝关节 → 右脚趾远端关节
        ]
        
        # 转换为0-based索引
        self.edges = [(i-1, j-1) for i, j in self.skeleton_edges]
        
        # 细粒度语义分组：10个主要身体区域（头颈、脊柱、左臂、左前臂、右臂、右前臂、左腿、左脚、右腿、右脚）
        # V4 Update: Context-Aware Grouping (引入父节点作为参照锚点，解决姿态歧义)
        self.semantic_groups = {
            'head_neck': [2, 3, 20],                      # 头部+颈部 (3个关节)
            'spine': [0, 1, 2, 20],                       # 脊柱 (4个关节)
            
            # 上肢 (引入 SpineShoulder=20 和 Shoulder=4/8 作为参照)
            'left_arm': [20, 4, 5],                       # 左上臂 (增加20作为躯干参照)
            'left_forearm': [4, 5, 6, 7, 21, 22],         # 左前臂 (增加4作为上臂参照)
            'right_arm': [20, 8, 9],                      # 右上臂 (增加20作为躯干参照)
            'right_forearm': [8, 9, 10, 11, 23, 24],      # 右前臂 (增加8作为上臂参照)
            
            # 下肢 (引入 SpineMid=1 和 Hip=12/16 作为参照)
            'left_leg': [1, 0, 12, 13],                   # 左大腿 (增加1作为脊柱参照，解决坐/站歧义)
            'left_foot': [12, 13, 14, 15],                # 左小腿 (增加12作为大腿参照)
            'right_leg': [1, 0, 16, 17],                  # 右大腿 (增加1作为脊柱参照，解决坐/站歧义)
            'right_foot': [16, 17, 18, 19]                # 右小腿 (增加16作为大腿参照)
        }
        
        self.num_joints = 25
        self.adjacency_matrix = self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        """构建邻接矩阵"""
        adj = torch.zeros(self.num_joints, self.num_joints)
        
        # 添加边连接
        for i, j in self.edges:
            if 0 <= i < self.num_joints and 0 <= j < self.num_joints:
                adj[i, j] = 1
                adj[j, i] = 1  # 无向图
        
        # 添加自连接
        adj += torch.eye(self.num_joints)
        
        # 归一化
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        adj = adj / degree
        
        return adj

class ST_GCN_Layer(nn.Module):
    """时空图卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 空间图卷积
        self.spatial_gcn = nn.Conv2d(in_channels, out_channels * kernel_size, 1)
        
        # 时间卷积
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, 
                                     (kernel_size, 1), (stride, 1), (padding, 0))
        
        # 残差连接
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix):
        """
        x: (B, C, T, V) - Batch, Channels, Time, Vertices(joints)
        adj_matrix: (V, V) - 邻接矩阵
        """
        residual = self.residual(x)

        # 空间图卷积
        B, C, T, V = x.size()

        # 先应用图卷积，再应用空间卷积
        # 重塑为图卷积格式
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x_reshaped = x_reshaped.view(B*T, V, C)

        # 应用图卷积 A * X
        x_gcn = torch.matmul(adj_matrix.to(x.device), x_reshaped)  # (B*T, V, C)

        # 恢复原始形状
        x_gcn = x_gcn.view(B, T, V, C).permute(0, 3, 1, 2)  # (B, C, T, V)

        # 应用空间卷积
        x = self.spatial_gcn(x_gcn)  # (B, out_channels*K, T, V)

        # 重新整形以分离kernel维度
        out_channels = x.size(1) // self.kernel_size
        x = x.view(B, self.kernel_size, out_channels, T, V)

        # 聚合不同kernel的结果
        x = x.sum(dim=1)  # (B, out_channels, T, V)

        # 时间卷积
        x = self.temporal_conv(x)

        # 残差连接和归一化
        x = self.bn(x + residual)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class SemanticGroupProcessor(nn.Module):
    """语义分组处理器"""
    
    def __init__(self, group_joints, in_channels, out_channels):
        super().__init__()
        self.group_joints = group_joints
        self.num_joints = len(group_joints)
        
        # 为该分组构建子图邻接矩阵
        self.register_buffer('sub_adj', self._build_sub_adjacency())
        
        # ST-GCN层
        self.st_gcn_layers = nn.ModuleList([
            ST_GCN_Layer(in_channels, 64),
            ST_GCN_Layer(64, 128), 
            ST_GCN_Layer(128, out_channels)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _build_sub_adjacency(self):
        """为当前分组构建子图邻接矩阵"""
        # 创建完整骨架图
        skeleton_graph = SkeletonGraph()
        full_adj = skeleton_graph.adjacency_matrix

        # 提取子图，保持连接性
        sub_adj = full_adj[self.group_joints][:, self.group_joints].clone()

        # 确保子图连通性：如果子图不连通，添加必要的连接
        num_joints = len(self.group_joints)

        # 检查连通性并修复
        if num_joints > 1:
            # 添加自连接
            sub_adj += torch.eye(num_joints)

            # 如果某些节点没有连接，连接到第一个节点（作为根节点）
            degree = sub_adj.sum(dim=1)
            isolated_nodes = (degree == 1).nonzero(as_tuple=True)[0]  # 只有自连接的节点

            for node in isolated_nodes:
                if node != 0:  # 不是根节点
                    sub_adj[0, node] = 1
                    sub_adj[node, 0] = 1

        # 重新归一化
        degree = sub_adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        sub_adj = sub_adj / degree

        return sub_adj
        
    def forward(self, x):
        """
        x: (B, C, T, V) 其中V是该分组的关节数
        """
        for layer in self.st_gcn_layers:
            x = layer(x, self.sub_adj)
        
        # 全局特征聚合
        features = self.global_pool(x)  # (B, C, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, C)
        
        return features

class VectorQuantizer(nn.Module):
    """向量量化模块（码本）"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.9):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 码本（部位自适应初始化尺度）
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 根据码本大小调整初始化尺度，避免大码本初始化过小
        if num_embeddings >= 100:  # 腿部大码本(128，已废弃)
            init_scale = 0.1
        elif num_embeddings >= 60:  # 腿部码本(64)
            init_scale = 0.05
        elif num_embeddings >= 45:  # 手臂码本(48) ← 修复：添加这个分支
            init_scale = 0.05
        else:  # 头部小码本(32)
            init_scale = 0.02
        self.embedding.weight.data.uniform_(-init_scale, init_scale)
        
    def forward(self, inputs):
        """
        inputs: (B, embedding_dim)
        """
        # 计算距离
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # 找到最近的码字
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,
                               device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)

        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        # 确保返回的indices是1D张量
        indices = encoding_indices.squeeze(1)  # 从(B,1)变为(B,)
        if indices.dim() == 0:  # 如果是标量，转为1D张量
            indices = indices.unsqueeze(0)

        return quantized, loss, indices

class IndependentSemanticCodebooks(nn.Module):
    """独立语义组码本 - 支持部位自适应码本大小和动态 commitment cost"""

    def __init__(self, semantic_groups, tokens_per_group=128, token_dim=64, tokens_config=None):
        super().__init__()

        self.semantic_groups = semantic_groups
        self.token_dim = token_dim
        
        # 部位自适应码本大小配置
        if tokens_config is None:
            # V3.1优化：手臂48，腿部64（基于实测利用率调整）
            self.tokens_config = {
                'head_neck': 32,
                'spine': 32,
                'left_arm': 64,       # ← V3.1: 64→48（利用率25%）
                'left_forearm': 128,   # ← V3.1: 64→48（利用率42%，进一步优化）
                'right_arm': 64,      # ← V3.1: 64→48（利用率18.8%）
                'right_forearm': 128,  # ← V3.1: 64→48（利用率18.8%）
                'left_leg': 64,       # ← V3: 128→64，利用率31%✓
                'left_foot': 64,      # 利用率40.6%✓
                'right_leg': 64,      # 利用率31%✓
                'right_foot': 64,     # 利用率48.4%✓
            }
        else:
            self.tokens_config = tokens_config
        
        # 兼容旧接口：如果某个部位未配置，使用默认值
        self.tokens_per_group = tokens_per_group  # 保留兼容性
        
        # 部位自适应 commitment cost 基准值（会被动态调整）
        # V3.1优化：恢复V2的commitment设置（腿部码本64已足够防坍缩）
        self.base_commitment_costs = {
            # 上半身：V2设置
            'head_neck': 0.5,
            'spine': 0.5,
            'left_arm': 0.4,
            'left_forearm': 0.4,
            'right_arm': 0.4,
            'right_forearm': 0.4,
            
            # 下半身：恢复V2的0.8（配合64码本，利用率已达30-48%）
            'left_leg': 0.8,      # ← V3.1: 恢复V2设置（V3的0.3→0.8）
            'left_foot': 0.8,
            'right_leg': 0.8,
            'right_foot': 0.8,
        }
        
        # 当前 epoch（用于动态调整）
        self.current_epoch = 0
        
        # 为每个语义组创建独立码本（使用自适应大小和commitment cost）
        self.group_codebooks = nn.ModuleDict()
        for group_name in semantic_groups.keys():
            num_tokens = self.tokens_config.get(group_name, tokens_per_group)
            commitment_cost = self.base_commitment_costs.get(group_name, 0.7)
            
            self.group_codebooks[group_name] = VectorQuantizer(
                num_embeddings=num_tokens,
                embedding_dim=token_dim,
                commitment_cost=commitment_cost
            )

    def forward(self, group_features_dict):
        """
        group_features_dict: {group_name: (B, token_dim)}
        返回: {group_name: (quantized, loss, indices)}
        """
        results = {}
        total_loss = 0.0

        for group_name, features in group_features_dict.items():
            if group_name in self.group_codebooks:
                quantized, loss, indices = self.group_codebooks[group_name](features)
                results[group_name] = {
                    'quantized': quantized,
                    'loss': loss,
                    'indices': indices
                }
                total_loss += loss

        return results, total_loss

    def update_epoch(self, epoch):
        """
        更新当前 epoch，用于动态调整 commitment cost
        
        V3.1反转退火策略（基于V2成功经验）：
        - Epoch 0-20:   scale=0.5  (低约束探索期，腿部0.8×0.5=0.4)
        - Epoch 20-50:  scale=0.7  (逐步增强)
        - Epoch 50-80:  scale=1.0  (标准值)
        - Epoch 80+:    scale=1.2  (微调期，增强稳定性)
        """
        self.current_epoch = epoch
        
        # 计算缩放因子（V3.1策略：恢复V2经验，配合64码本）
        if epoch < 20:
            scale = 0.5   # 早期20轮：低commitment（腿部0.8×0.5=0.4）
        elif epoch < 50:
            scale = 0.7   # 中期：逐步增强
        elif epoch < 80:
            scale = 1.0   # 后期：标准值
        else:
            scale = 1.2   # 微调期：轻微增强
        
        # 动态更新所有码本的 commitment cost
        for group_name, codebook in self.group_codebooks.items():
            base_cost = self.base_commitment_costs.get(group_name, 0.7)
            new_cost = base_cost * scale
            codebook.commitment_cost = new_cost
    
    def get_token_sequence(self, group_results):
        """
        将各组的token索引组合成序列，用于LLM
        返回: (B, num_groups) 的token序列
        注意：由于各部位码本大小不同，需要添加偏移量
        """
        batch_size = None
        token_sequence = []
        device = None

        # 按固定顺序排列各组token (10个部位)
        group_order = ['head_neck', 'spine', 'left_arm', 'left_forearm', 
                      'right_arm', 'right_forearm', 'left_leg', 'left_foot',
                      'right_leg', 'right_foot']

        # 首先确定batch_size和device
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']

                # 确保indices是张量且至少是1D
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                if batch_size is None:
                    batch_size = indices.size(0)
                    device = indices.device
                break

        # 如果没有找到有效的组结果，返回空张量
        if batch_size is None:
            return torch.empty(0, len(group_order))

        # 处理每个组的token
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']

                # 确保indices是正确格式的张量
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, device=device)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                # 确保batch维度匹配
                if indices.size(0) != batch_size:
                    if indices.size(0) == 1:
                        indices = indices.expand(batch_size)
                    else:
                        # 如果维度不匹配，用第一个值填充
                        indices = torch.full((batch_size,), indices[0].item(), device=device)

                # 为每个组添加偏移量，确保token ID不重复
                # 计算累积偏移量（考虑各组码本大小不同）
                group_offset = 0
                for prev_group in group_order[:group_order.index(group_name)]:
                    group_offset += self.tokens_config.get(prev_group, self.tokens_per_group)
                
                offset_indices = indices + group_offset
                token_sequence.append(offset_indices)
            else:
                # 如果某组缺失，用特殊token填充
                padding_token = torch.full((batch_size,), -1, device=device)
                token_sequence.append(padding_token)

        if token_sequence:
            return torch.stack(token_sequence, dim=1)  # (B, num_groups)
        else:
            return torch.empty(0, len(group_order))

@MODELS.register_module()
class GCNSkeletonTokenizer_10p(nn.Module):
    """
    基于时空图卷积的骨架Tokenizer (10-part版本)
    直接处理25个关节点，使用10个语义分组，提供更细粒度的肢体表示
    - head_neck, spine: 分离头颈和脊柱
    - left_arm, left_forearm, right_arm, right_forearm: 分离大臂和小臂
    - left_leg, left_foot, right_leg, right_foot: 分离大腿和小腿+脚
    """
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        # 处理配置参数 - 兼容不同的参数传递方式
        if config is not None:
            # 如果传递了config对象
            if hasattr(config, '__dict__'):
                config_dict = config.__dict__
            else:
                config_dict = config
        else:
            config_dict = {}
        
        # 合并所有参数
        params = {**config_dict, **kwargs}
        
        self.num_tokens = params.get('num_tokens', 512)
        self.token_dim = params.get('token_dim', 128)
        self.temporal_length = params.get('temporal_length', 1)
        
        # 新增配置参数
        tokens_per_group = params.get('tokens_per_group', 64)  # 每组64个token（默认值，实际会被tokens_config覆盖）
        group_token_dim = params.get('group_token_dim', 128)   # 组token维度128
        
        # 部位自适应码本大小配置
        tokens_config = params.get('tokens_config', None)  # 如果未提供，使用默认策略
        
        # 骨架图结构
        self.skeleton_graph = SkeletonGraph()
        
        # 关节角度计算配置
        self.use_angles = params.get('use_angles', True) # 默认开启角度计算
        input_dim = 4 if self.use_angles else 3
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, 64)  # xyz坐标(+角度) -> 特征维度
        
        # 语义分组处理器 - 使用配置的token维度
        self.group_processors = nn.ModuleDict()

        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            self.group_processors[group_name] = SemanticGroupProcessor(
                joints, 64, group_token_dim
            )

        # 独立语义组码本 - 使用配置的参数和自适应码本大小
        self.semantic_codebooks = IndependentSemanticCodebooks(
            semantic_groups=self.skeleton_graph.semantic_groups,
            tokens_per_group=tokens_per_group,  # 默认值（用于未在tokens_config中指定的部位）
            token_dim=group_token_dim,          # 使用配置的token维度
            tokens_config=tokens_config         # 部位自适应配置
        )

        # 全局特征融合（用于重构）- 使用配置的维度
        total_group_dim = len(self.skeleton_graph.semantic_groups) * group_token_dim
        self.global_fusion = nn.Sequential(
            nn.Linear(total_group_dim, self.token_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim)
        )
        
        # 增强重构头 - 多层深度网络，更好的特征提取能力
        self.reconstruction_head = nn.Sequential(
            # 第一阶段：特征扩展
            nn.Linear(self.token_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二阶段：特征细化
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三阶段：空间感知
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            # 第四阶段：关节重构
            nn.Linear(256, 25 * 3)  # 重建25个关节点的xyz坐标
        )
        
        # 残差连接控制：通过sigmoid门限限制残差的放大
        # 初始值设为 -5.0 让初期训练几乎完全依赖码本
        self.residual_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007
        self.residual_reg_weight = params.get('residual_reg_weight', 0.1)  # 提高到0.1
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
    def compute_joint_angles(self, x):
        """
        计算核心关节角度 (Cosine Similarity)
        x: (..., 25, 3)
        Returns: (..., 25, 1)
        """
        # 定义三元组 (A, B, C) 其中 B 是中心关节
        # 索引基于 NTU RGB+D 25
        triplets = [
            (4, 5, 6),    # 左肘 (Left Elbow)
            (8, 9, 10),   # 右肘 (Right Elbow)
            (12, 13, 14), # 左膝 (Left Knee)
            (16, 17, 18), # 右膝 (Right Knee)
            (0, 1, 20),   # 脊柱中段 (Spine Mid)
            (20, 4, 5),   # 左肩 (Left Shoulder)
            (20, 8, 9),   # 右肩 (Right Shoulder)
            (0, 12, 13),  # 左髋 (Left Hip)
            (0, 16, 17)   # 右髋 (Right Hip)
        ]
        
        # 初始化角度特征 (默认为0)
        angles = torch.zeros_like(x[..., :1]) # (..., 25, 1)
        
        for a, b, c in triplets:
            # 计算向量 BA 和 BC
            vec_ba = x[..., a, :] - x[..., b, :]
            vec_bc = x[..., c, :] - x[..., b, :]
            
            # 归一化
            norm_ba = torch.norm(vec_ba, dim=-1, keepdim=True) + 1e-6
            norm_bc = torch.norm(vec_bc, dim=-1, keepdim=True) + 1e-6
            
            # 计算余弦相似度
            cosine = torch.sum(vec_ba * vec_bc, dim=-1, keepdim=True) / (norm_ba * norm_bc)
            
            # 赋值给中心关节
            angles[..., b, :] = cosine
            
        return angles

    def forward(self, inp=None, temperature=1.0, hard=False, return_recon=True, skeleton_data=None, eval=False, **kwargs):
        """
        inp: (B, 25, 3) 或 (B, T, 25, 3) - 25个关节点的xyz坐标（主要输入参数，与原框架一致）
        skeleton_data: 兼容参数名，等同于inp
        temperature: 温度参数（为了兼容原框架，在GCN中不使用）
        hard: 硬采样标志（为了兼容原框架，在GCN中不使用）
        eval: 验证模式标志（为了兼容原框架）
        """
        # 兼容不同的输入参数名（优先使用inp以匹配原框架）
        if inp is not None:
            skeleton_data = inp
        elif skeleton_data is not None:
            pass  # 使用skeleton_data
        else:
            raise ValueError("Either inp or skeleton_data must be provided")

        # 处理不同的输入维度
        if skeleton_data.dim() == 3:
            # (B, 25, 3) -> (B, 1, 25, 3)
            skeleton_data = skeleton_data.unsqueeze(1)
        elif skeleton_data.dim() == 4:
            # (B, T, 25, 3) 已经是正确格式
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {skeleton_data.dim()}D")

        batch_size, temporal_length, num_joints, coord_dim = skeleton_data.shape

        # 输入嵌入：处理每一帧
        x = skeleton_data.reshape(batch_size * temporal_length, num_joints, coord_dim)
        
        # 如果启用角度计算，计算并拼接角度特征
        if self.use_angles:
            angles = self.compute_joint_angles(x) # (B*T, 25, 1)
            x = torch.cat([x, angles], dim=-1)    # (B*T, 25, 4)
            
        x = self.input_embedding(x)  # (B*T, 25, 64)
        x = x.reshape(batch_size, temporal_length, num_joints, -1)

        # 为时空卷积准备数据格式 (B, C, T, V)
        x = x.permute(0, 3, 1, 2)  # (B, 64, T, 25)
        
        # 分组处理
        group_features_dict = {}
        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            # 提取该分组的关节特征
            group_x = x[:, :, :, joints]  # (B, 64, T, num_group_joints)

            # 通过分组处理器
            group_feat = self.group_processors[group_name](group_x)
            group_features_dict[group_name] = group_feat

        # 独立语义组量化
        group_results, total_vq_loss = self.semantic_codebooks(group_features_dict)

        # 全局特征融合（用于重构）
        group_quantized_features = []
        for group_name in self.skeleton_graph.semantic_groups.keys():
            if group_name in group_results:
                group_quantized_features.append(group_results[group_name]['quantized'])

        combined_features = torch.cat(group_quantized_features, dim=1)  # (B, total_group_dim)
        global_features = self.global_fusion(combined_features)

        if return_recon:
            input_flat = skeleton_data.reshape(batch_size, temporal_length, -1)  # (B, T, 75)
            base_recon_flat = self.reconstruction_head(global_features)  # (B, 75)

            if temporal_length > 1:
                base_recon_flat = base_recon_flat.unsqueeze(1).repeat(1, temporal_length, 1)  # (B, T, 75)
            else:
                base_recon_flat = base_recon_flat.unsqueeze(1)  # (B, 1, 75)

            residual_scale = torch.sigmoid(self.residual_gate)
            final_flat = base_recon_flat + residual_scale * (input_flat - base_recon_flat)

            final_reconstructed = final_flat.reshape(batch_size, temporal_length, 25, 3)
            base_reconstructed = base_recon_flat.reshape(batch_size, temporal_length, 25, 3)

            if temporal_length == 1:
                final_reconstructed = final_reconstructed.squeeze(1)
                base_reconstructed = base_reconstructed.squeeze(1)
                original = skeleton_data.squeeze(1)
            else:
                original = skeleton_data

            # 如果是验证模式，返回兼容原框架的元组格式
            if eval or hard:
                # 返回 (coarse_points, fine_points, ...) 格式以兼容验证代码
                # 对于骨架数据，coarse和fine都返回重建的骨架
                return (final_reconstructed, final_reconstructed, None, None, None, None)

            # 只在非验证模式下生成token序列
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)

            return {
                'quantized': global_features,
                'reconstructed': final_reconstructed,
                'vq_loss': total_vq_loss,
                'token_sequence': token_sequence,
                'group_results': group_results,
                'original': original,
                'base_reconstructed': base_reconstructed,
                'residual_scale': residual_scale
            }
        else:
            # 如果是验证模式但不需要重建，返回原始数据
            if eval or hard:
                return (skeleton_data, skeleton_data, None, None, None, None)

            # 只在非验证模式下生成token序列
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)

            return {
                'quantized': global_features,
                'vq_loss': total_vq_loss,
                'token_sequence': token_sequence,
                'group_results': group_results
            }
    
    def get_loss(self, ret, gt, compute_metrics=False):
        """
        计算总损失 - 与框架接口一致
        
        Args:
            ret: forward 的输出
            gt: 真实骨架数据
            compute_metrics: 是否计算骨架专用指标（验证时使用）
            
        Returns:
            total_loss: 重构损失 + 残差正则化
            vq_loss: VQ码本损失
            metrics (optional): 如果 compute_metrics=True，返回骨架指标字典
        """
        # ret 是 forward 的输出，gt 是真实骨架数据
        forward_output = ret
        vq_loss = forward_output['vq_loss']  # 码本量化损失
        
        if 'reconstructed' in forward_output:
            reconstructed = forward_output['reconstructed']
            original = forward_output['original']
            
            # 部位加权重构损失（V3平衡优化：均衡手臂和腿部权重）
            joint_weights = torch.ones(25, device=reconstructed.device)
            # 腿部关节加权3x（NTU关节索引12-19）V3平衡: 5.0→3.0（避免过度偏向腿部）
            leg_joints = [12, 13, 14, 15, 16, 17, 18, 19]
            joint_weights[leg_joints] = 3.0
            # 手臂关节加权2.5x（V3平衡: 1.5→2.5，提升手臂细节保留）
            arm_joints = [4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24]
            joint_weights[arm_joints] = 2
            
            # 逐关节加权MSE
            diff_squared = (reconstructed - original) ** 2  # (B, 25, 3) or (B, T, 25, 3)
            if diff_squared.dim() == 4:  # 时序数据
                weighted_diff = diff_squared * joint_weights.view(1, 1, 25, 1)
            else:  # 单帧数据
                weighted_diff = diff_squared * joint_weights.view(1, 25, 1)
            recon_loss = weighted_diff.mean()
            
            # 残差正则化
            residual_scale = forward_output.get('residual_scale')
            residual_reg = torch.tensor(0.0, device=reconstructed.device)
            if residual_scale is not None:
                residual_reg = self.residual_reg_weight * (residual_scale ** 2)

            total_loss = recon_loss + residual_reg
            
            # 如果需要计算骨架专用指标（验证阶段）
            if compute_metrics:
                with torch.no_grad():
                    skeleton_metrics = self.compute_skeleton_metrics(reconstructed, original)
                return total_loss, vq_loss, skeleton_metrics
            
            return total_loss, vq_loss
        else:
            # 如果没有重构，返回零重构损失和VQ损失
            zero_loss = torch.tensor(0.0, device=vq_loss.device)
            if compute_metrics:
                return zero_loss, vq_loss, {}
            return zero_loss, vq_loss
    
    def encode(self, skeleton_data):
        """编码：骨架 -> 离散token序列"""
        with torch.no_grad():
            output = self.forward(skeleton_data, return_recon=False)
            return output['token_sequence']  # (B, num_groups)

    def decode(self, token_sequence):
        """解码:离散token序列 -> 骨架"""
        with torch.no_grad():
            batch_size = token_sequence.size(0)
            group_order = ['head_neck', 'spine', 'left_arm', 'left_forearm', 'right_arm', 'right_forearm', 'left_leg', 'left_foot', 'right_leg', 'right_foot']

            # 从token序列重建各组特征
            group_features = []
            for i, group_name in enumerate(group_order):
                if i < token_sequence.size(1):
                    # 移除偏移量获取原始token索引
                    group_offset = i * self.semantic_codebooks.tokens_per_group
                    original_indices = token_sequence[:, i] - group_offset

                    # 从对应组的码本获取特征
                    if group_name in self.semantic_codebooks.group_codebooks:
                        quantized_features = self.semantic_codebooks.group_codebooks[group_name].embedding(original_indices)
                        group_features.append(quantized_features)

            # 融合特征
            if group_features:
                combined_features = torch.cat(group_features, dim=1)
                global_features = self.global_fusion(combined_features)

                base = self.reconstruction_head(global_features)
                reconstructed = base.reshape(batch_size, 25, 3)

                return reconstructed
            else:
                return torch.zeros(batch_size, 25, 3)

    # ----- 骨架专用指标计算 -----
    def compute_skeleton_metrics(self, pred, gt):
        """
        计算骨架重建专用指标
        
        Args:
            pred: (B, 25, 3) 预测骨架
            gt: (B, 25, 3) 真实骨架
            
        Returns:
            dict: 包含 MPJPE, PA-MPJPE, PCK@0.1 等指标
        """
        metrics = {}
        
        # MPJPE (Mean Per Joint Position Error) - 毫米
        mpjpe = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()
        metrics['mpjpe'] = mpjpe.item() * 1000  # 转为毫米
        
        # PA-MPJPE (Procrustes Aligned MPJPE)
        pred_aligned = self._procrustes_align(pred, gt)
        pa_mpjpe = torch.sqrt(((pred_aligned - gt) ** 2).sum(dim=-1)).mean()
        metrics['pa_mpjpe'] = pa_mpjpe.item() * 1000
        
        # PCK@0.1 (Percentage of Correct Keypoints, 阈值=10%躯干长度)
        pck = self._compute_pck(pred, gt, threshold=0.1)
        metrics['pck_01'] = pck * 100  # 转为百分比
        
        # 部位细分 MPJPE
        part_mpjpes = self._compute_mpjpe_by_part(pred, gt)
        metrics.update(part_mpjpes)
        
        return metrics
    
    @staticmethod
    def _procrustes_align(pred, gt):
        """
        Procrustes 对齐（消除全局旋转/平移/缩放）
        
        Args:
            pred: (B, 25, 3)
            gt: (B, 25, 3)
            
        Returns:
            pred_aligned: (B, 25, 3) 对齐后的预测
        """
        # 中心化
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        gt_centered = gt - gt.mean(dim=1, keepdim=True)
        
        # SVD 求最优旋转矩阵
        H = torch.bmm(pred_centered.transpose(1, 2), gt_centered)
        U, S, V = torch.svd(H)
        R = torch.bmm(V, U.transpose(1, 2))
        
        # 处理反射情况（确保是旋转而非反射）
        det = torch.det(R)
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        V_corrected = V.clone()
        V_corrected[:, :, -1] = V_corrected[:, :, -1] * sign.squeeze(-1)
        R = torch.bmm(V_corrected, U.transpose(1, 2))
        
        # 计算缩放因子
        scale = (gt_centered ** 2).sum() / ((pred_centered ** 2).sum() + 1e-9)
        
        # 应用变换
        pred_aligned = scale * torch.bmm(pred_centered, R) + gt.mean(dim=1, keepdim=True)
        return pred_aligned
    
    @staticmethod
    def _compute_pck(pred, gt, threshold=0.1):
        """
        计算 PCK (Percentage of Correct Keypoints)
        
        Args:
            pred: (B, 25, 3)
            gt: (B, 25, 3)
            threshold: 相对于躯干长度的阈值
            
        Returns:
            pck: 正确关节的比例 (0-1)
        """
        # 计算躯干长度（头顶(3)到骨盆(0)的距离）
        torso_length = torch.norm(gt[:, 3] - gt[:, 0], dim=-1, keepdim=True)  # (B, 1)
        
        # 计算每个关节的误差
        joint_errors = torch.norm(pred - gt, dim=-1)  # (B, 25)
        
        # 归一化到躯干长度
        normalized_errors = joint_errors / (torso_length + 1e-6)
        
        # 计算正确率（误差小于阈值的关节比例）
        correct = (normalized_errors < threshold).float()
        pck = correct.mean()
        
        return pck.item()
    
    def _compute_mpjpe_by_part(self, pred, gt):
        """
        按身体部位计算 MPJPE
        
        Args:
            pred: (B, 25, 3)
            gt: (B, 25, 3)
            
        Returns:
            dict: 各部位的 MPJPE (毫米)
        """
        # NTU RGB+D 25关节点索引分组
        parts = {
            'torso': [0, 1, 2, 20],  # 骨盆、脊柱、颈、肩中
            'head': [3],  # 头顶
            'left_arm': [4, 5, 6, 7, 21, 22],  # 左臂+手
            'right_arm': [8, 9, 10, 11, 23, 24],  # 右臂+手
            'left_leg': [12, 13, 14, 15],  # 左腿
            'right_leg': [16, 17, 18, 19],  # 右腿
        }
        
        part_mpjpes = {}
        for part_name, indices in parts.items():
            pred_part = pred[:, indices]
            gt_part = gt[:, indices]
            mpjpe = torch.sqrt(((pred_part - gt_part) ** 2).sum(dim=-1)).mean()
            part_mpjpes[f'mpjpe_{part_name}'] = mpjpe.item() * 1000
        
        return part_mpjpes
    
    # ----- 诊断 / 工具方法 -----
    def analyze_codebook_usage(self, skeleton_data, topk=10, device='cpu'):
        """
        分析一批输入上的码本使用情况与残差贡献。

        返回一个字典，包含每个语义组的token频次(hist)、总体vq_loss均值、
        以及残差分量范数的统计（mean/std）。用于训练/导出后快速诊断码本是否collapse。
        """
        was_training = self.training
        self.eval()

        if isinstance(skeleton_data, np.ndarray):
            skeleton_data = torch.tensor(skeleton_data, dtype=torch.float32)

        if skeleton_data.dim() == 3:
            skeleton_data = skeleton_data.unsqueeze(1)

        skeleton_data = skeleton_data.to(device)

        with torch.no_grad():
            out = self.forward(skeleton_data, return_recon=True)

        # group_results 是字典
        group_results = out.get('group_results', {})
        usage = {}
        for gname, gres in group_results.items():
            indices = gres['indices']
            if isinstance(indices, torch.Tensor):
                arr = indices.detach().cpu().numpy().ravel()
            else:
                arr = np.array(indices).ravel()
            # 统计前 topk 出现的token
            vals, counts = np.unique(arr, return_counts=True)
            order = np.argsort(-counts)
            top_vals = vals[order][:topk] if len(order) > 0 else np.array([])
            top_counts = counts[order][:topk] if len(order) > 0 else np.array([])
            usage[gname] = {
                'unique_tokens': int(len(vals)),
                'topk_tokens': top_vals.tolist(),
                'topk_counts': top_counts.tolist(),
                'total': int(arr.size)
            }

        vq_loss = out.get('vq_loss')
        vq_loss_val = float(vq_loss.detach().cpu().mean()) if isinstance(vq_loss, torch.Tensor) else (float(vq_loss) if vq_loss is not None else None)

        residual_norms = None
        residual_scale = out.get('residual_scale')
        base_recon = out.get('base_reconstructed')
        final_recon = out.get('reconstructed')

        if base_recon is not None and final_recon is not None:
            base_np = base_recon.detach().cpu().numpy()
            final_np = final_recon.detach().cpu().numpy()
            residual_component = final_np - base_np
            if residual_component.ndim == 4:  # (B, T, 25, 3)
                flat = residual_component.reshape(residual_component.shape[0], -1)
            else:  # (B, 25, 3)
                flat = residual_component.reshape(residual_component.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1)
            residual_norms = {
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
                'min': float(np.min(norms)),
                'max': float(np.max(norms))
            }
        residual_scale_val = None
        if isinstance(residual_scale, torch.Tensor):
            residual_scale_val = float(residual_scale.detach().cpu().item())

        if was_training:
            self.train()

        return {
            'usage': usage,
            'vq_loss_mean': vq_loss_val,
            'residual_norms': residual_norms,
            'residual_scale': residual_scale_val
        }
    
    @staticmethod
    def format_skeleton_metrics(metrics):
        """
        格式化骨架指标输出
        
        Args:
            metrics: compute_skeleton_metrics 返回的指标字典
            
        Returns:
            str: 格式化的指标字符串
        """
        if not metrics:
            return "No metrics available"
        
        lines = []
        lines.append("=" * 60)
        lines.append("Skeleton Reconstruction Metrics")
        lines.append("=" * 60)
        
        # 主要指标
        if 'mpjpe' in metrics:
            lines.append(f"  MPJPE:          {metrics['mpjpe']:>8.2f} mm  (越小越好)")
        if 'pa_mpjpe' in metrics:
            lines.append(f"  PA-MPJPE:       {metrics['pa_mpjpe']:>8.2f} mm  (对齐后)")
        if 'pck_01' in metrics:
            lines.append(f"  PCK@0.1:        {metrics['pck_01']:>8.2f} %   (10%躯干长度阈值)")
        
        # 部位细分
        lines.append("-" * 60)
        lines.append("Body Part Breakdown:")
        part_order = ['torso', 'head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        part_names = {
            'torso': '躯干',
            'head': '头部',
            'left_arm': '左臂',
            'right_arm': '右臂',
            'left_leg': '左腿',
            'right_leg': '右腿'
        }
        
        for part in part_order:
            key = f'mpjpe_{part}'
            if key in metrics:
                name = part_names.get(part, part)
                lines.append(f"  {name:>6s}:        {metrics[key]:>8.2f} mm")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

def create_gcn_skeleton_tokenizer(config=None, **kwargs):
    """创建GCN骨架Tokenizer的工厂函数"""
    
    # 默认配置
    default_config = {
        'num_tokens': 512,
        'token_dim': 256,
        'temporal_length': 1
    }
    
    if config:
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        default_config.update(config_dict)
    
    default_config.update(kwargs)
    
    model = GCNSkeletonTokenizer_10p(config=default_config)
    
    # 计算实际总词汇量（考虑自适应码本大小）
    if model.semantic_codebooks.tokens_config:
        total_vocab = sum(
            model.semantic_codebooks.tokens_config.get(group_name, model.semantic_codebooks.tokens_per_group)
            for group_name in model.skeleton_graph.semantic_groups.keys()
        )
    else:
        total_vocab = len(model.skeleton_graph.semantic_groups) * model.semantic_codebooks.tokens_per_group
    
    print(f"Created GCNSkeletonTokenizer_10p with Independent Semantic Codebooks (10 groups):")
    print(f"  - Global tokens: {default_config['num_tokens']}")
    print(f"  - Token dim: {default_config['token_dim']}")
    print(f"  - Semantic groups: {len(model.skeleton_graph.semantic_groups)}")
    
    # 显示每个部位的码本大小（如果使用自适应配置）
    if model.semantic_codebooks.tokens_config:
        print(f"  - Adaptive codebook sizes:")
        for group_name in model.skeleton_graph.semantic_groups.keys():
            num_tokens = model.semantic_codebooks.tokens_config.get(
                group_name, model.semantic_codebooks.tokens_per_group
            )
            print(f"    * {group_name}: {num_tokens} tokens")
    else:
        print(f"  - Tokens per group: {model.semantic_codebooks.tokens_per_group}")
    
    print(f"  - Total vocabulary size: {total_vocab}")
    print(f"  - Group names: {list(model.skeleton_graph.semantic_groups.keys())}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("Testing GCNSkeletonTokenizer_10p with Skeleton-Specific Metrics")
    print("=" * 80)
    
    model = create_gcn_skeleton_tokenizer()
    
    # 模拟输入：批次大小为4，25个关节点，xyz坐标
    test_input = torch.randn(4, 25, 3)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # 前向传播
    output = model(test_input)
    
    print(f"Quantized features shape: {output['quantized'].shape}")
    print(f"Reconstructed shape: {output['reconstructed'].shape}")
    print(f"Token sequence shape: {output['token_sequence'].shape}")
    print(f"Number of semantic groups: {len(output['group_results'])}")

    # 显示各组的token信息
    print("\nSemantic Group Token Indices:")
    for group_name, group_result in output['group_results'].items():
        print(f"  {group_name}: {group_result['indices'].shape}")

    # 计算损失（不含指标）
    total_loss, vq_loss = model.get_loss(output, test_input)
    print(f"\n--- Training Loss (Original Framework) ---")
    print(f"Total loss (loss_1): {total_loss.item():.4f}")
    print(f"VQ loss (loss_2): {vq_loss.item():.4f}")

    # 计算骨架专用指标
    print("\n--- Skeleton-Specific Metrics (New) ---")
    total_loss_with_metrics, vq_loss_with_metrics, skeleton_metrics = model.get_loss(
        output, test_input, compute_metrics=True
    )
    
    # 打印格式化的指标
    print(model.format_skeleton_metrics(skeleton_metrics))
    
    # 显示原始指标字典
    print("\nRaw Metrics Dictionary:")
    for key, value in skeleton_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 测试编码解码
    print("\n--- Encoding/Decoding Test ---")
    token_sequence = model.encode(test_input)
    decoded_skeleton = model.decode(token_sequence)
    print(f"Encoded token sequence shape: {token_sequence.shape}")
    print(f"Decoded skeleton shape: {decoded_skeleton.shape}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
