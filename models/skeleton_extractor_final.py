#!/usr/bin/env python3
"""
MARS骨架提取简化稳定版 (skeleton_extractor_final.py)
==========================================================

核心策略：
1. 移除复杂的手部专家分支（导致梯度不稳定）
2. 使用空间保留主干 + 简单回归头
3. 优化的损失函数权重
4. 渐进式特征降维

预期：稳定训练，手部精度提升15-25%
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ============================================================================
# 全局配置
# ============================================================================
NUM_JOINTS = 19
OUTPUT_DIM = NUM_JOINTS * 3
BONE_CONNECTIONS = [
    (2, 3), (2, 18), (18, 4), (4, 5), (5, 6),
    (18, 7), (7, 8), (8, 9), (18, 1), (1, 0),
    (0, 10), (10, 11), (11, 12), (12, 13), (0, 14),
    (14, 15), (15, 16), (16, 17)
]
# 手部关节: 左手[12,13], 右手[16,17]
HAND_JOINT_INDICES = [12, 13, 16, 17]
# 肩部关节: 左肩7, 右肩14
SHOULDER_JOINT_INDICES = [7, 14]
# 核心部位关节: 躯干+头部+肩部
CORE_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 14, 18]

# 优化后的损失权重配置
BASE_LR = 1e-3             # 提高学习率
MSE_WEIGHT = 0.7           # 主要损失
L1_WEIGHT = 0.3            # 辅助损失
GRAD_CLIP_NORM = 1.0       # 标准梯度裁剪


# ============================================================================
# GPU配置
# ============================================================================
def configure_gpu():
    """配置GPU使用"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    print("❌ 未检测到GPU，将使用CPU")
    return torch.device('cpu')


# ============================================================================
# 基础注意力模块
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """空间注意力模块 - 保留位置信息"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module - 通道+空间双重注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================================
# 优化后的主干网络 - 移除全局池化，保留空间信息
# ============================================================================
class SpatialPreservingBackbone(nn.Module):
    """
    空间保留主干网络
    
    关键优化:
    1. ❌ 移除全局池化 AdaptiveAvgPool2d(1) - 避免信息丢失
    2. ✅ 使用空间展平 + 注意力 - 保留位置信息
    3. ✅ 多尺度特征融合 - 综合粗粒度和细粒度信息
    """
    def __init__(self, input_channels=5):
        super().__init__()
        # Stage 1: 8×8 → 8×8 (细粒度特征)
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64)
        )
        
        # Stage 2: 8×8 → 4×4 (中等粒度)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 8×8 → 4×4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128)
        )
        
        # Stage 3: 4×4 → 4×4 (语义特征)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        
        # 🔑 关键改进: 使用1×1卷积降维而非全局池化
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(256, 128, 1),  # 256→128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 输出维度: 4×4×128 = 2048 → flatten后用于后续处理
        self.output_dim = 4 * 4 * 128

    def forward(self, x):
        """
        输入: (B, 5, 8, 8)
        输出: (B, 2048) - 保留空间结构的展平特征
        """
        x = self.stage1(x)      # (B, 64, 8, 8)
        x = self.stage2(x)      # (B, 128, 4, 4)
        x = self.stage3(x)      # (B, 256, 4, 4)
        x = self.spatial_compress(x)  # (B, 128, 4, 4)
        
        # 空间展平而非全局池化
        x = x.flatten(1)  # (B, 2048)
        return x


# ============================================================================
# 简化的回归头 - 渐进式降维
# ============================================================================
class SimplifiedRegressionHead(nn.Module):
    """
    简化回归头 - 移除复杂融合，使用渐进降维
    
    架构: 2048 → 1024 → 512 → 256 → 128 → 57
    """
    def __init__(self, input_dim=2048, output_dim=OUTPUT_DIM):
        super().__init__()
        
        self.regressor = nn.Sequential(
            # 第1层: 2048 → 1024
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 第2层: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 第3层: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 第4层: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # 输出层: 128 → 57
            nn.Linear(128, output_dim)
        )
        
        # Xavier初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # 降低初始化增益
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.regressor(x)


# ============================================================================
# 完整模型 - 简化版
# ============================================================================
class OptimizedMARSModel(nn.Module):
    """
    简化优化版MARS骨架提取模型
    
    核心改进:
    1. 空间保留主干网络 (移除全局池化)
    2. 渐进式回归头 (稳定降维)
    3. 优化损失函数
    """
    def __init__(self, input_channels=5, output_dim=OUTPUT_DIM):
        super().__init__()
        self.backbone = SpatialPreservingBackbone(input_channels)
        self.regression_head = SimplifiedRegressionHead(
            input_dim=self.backbone.output_dim,
            output_dim=output_dim
        )
        
        print(f"✓ 简化模型初始化完成")
        print(f"  - 主干输出维度: {self.backbone.output_dim}")
        print(f"  - 回归头: 渐进式5层降维")

    def forward(self, x):
        """
        前向传播
        
        输入: (B, 5, 8, 8) 雷达特征图
        输出: (B, 57) 关节坐标
        """
        backbone_feat = self.backbone(x)  # (B, 2048)
        output = self.regression_head(backbone_feat)
        return output


# ============================================================================
# 数据集与加载器
# ============================================================================
class RadarSkeletonDataset(Dataset):
    """雷达骨架数据集 - 简化版（无增强）"""
    def __init__(self, features, labels, augment=False, noise_std=0.02, enhance_features=False):
        # 直接使用原始数据，不做任何预处理
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feats = self.features[idx]
        labels = self.labels[idx]
        # 训练时也不做增强，避免破坏数据分布
        return feats, labels


def load_and_preprocess_data():
    """加载并预处理数据 - 简化版（无过滤）"""
    print("🔄 加载MARS数据...")
    featuremap_train = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_train.npy')
    featuremap_validate = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_validate.npy')
    featuremap_test = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy')
    
    labels_train = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_train.npy')
    labels_validate = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_validate.npy')
    labels_test = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy')
    
    print(f"训练数据: {featuremap_train.shape}")
    print(f"验证数据: {featuremap_validate.shape}")
    print(f"测试数据: {featuremap_test.shape}")
    
    # 转换为NCHW格式
    featuremap_train = np.transpose(featuremap_train, (0, 3, 1, 2))
    featuremap_validate = np.transpose(featuremap_validate, (0, 3, 1, 2))
    featuremap_test = np.transpose(featuremap_test, (0, 3, 1, 2))
    
    print(f"\n✓ 数据预处理完成")
    print(f"  训练集: {featuremap_train.shape}")
    print(f"  验证集: {featuremap_validate.shape}")
    print(f"  测试集: {featuremap_test.shape}")
    
    return (featuremap_train, featuremap_validate, featuremap_test,
            labels_train, labels_validate, labels_test)


def create_data_loaders(train_features, train_labels,
                        val_features, val_labels,
                        test_features, test_labels,
                        batch_size=32):
    # 所有数据集都不使用增强
    train_dataset = RadarSkeletonDataset(train_features, train_labels, 
                                         augment=False, enhance_features=False)
    val_dataset = RadarSkeletonDataset(val_features, val_labels, 
                                       augment=False, enhance_features=False)
    test_dataset = RadarSkeletonDataset(test_features, test_labels,
                                        augment=False, enhance_features=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 优化的损失函数
# ============================================================================
def reshape_to_joints(data: torch.Tensor) -> torch.Tensor:
    """将(B, 57)重塑为(B, 19, 3)"""
    return data.view(-1, NUM_JOINTS, 3)


def compute_bone_length_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """骨长一致性损失"""
    preds_j = reshape_to_joints(preds)
    targets_j = reshape_to_joints(targets)
    
    losses = []
    for i, j in BONE_CONNECTIONS:
        pred_len = torch.norm(preds_j[:, i] - preds_j[:, j], dim=1)
        target_len = torch.norm(targets_j[:, i] - targets_j[:, j], dim=1)
        losses.append(torch.abs(pred_len - target_len))
    
    return torch.stack(losses, dim=1).mean()


def compute_hand_specific_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """手部专项损失函数 - 简化版：只用L1，权重1.5"""
    preds_j = reshape_to_joints(preds)
    targets_j = reshape_to_joints(targets)
    
    # 只用简单的L1损失，权重1.5
    hand_l1_losses = []
    for hand_idx in HAND_JOINT_INDICES:
        hand_error = torch.abs(preds_j[:, hand_idx] - targets_j[:, hand_idx]).mean(dim=1)
        hand_l1_losses.append(hand_error)
    hand_l1_loss = torch.stack(hand_l1_losses).mean() * 1.5
    
    return hand_l1_loss


def compute_total_loss(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """计算总损失 - 极简版：MSE(0.7) + L1(0.3)"""
    # 检查输入是否包含NaN或Inf
    if torch.isnan(preds).any() or torch.isinf(preds).any():
        print("⚠️ 警告: 预测值包含NaN或Inf")
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)
    
    mse_loss = F.mse_loss(preds, targets)
    l1_loss = F.l1_loss(preds, targets)
    
    # 极简损失：MSE主导，L1辅助
    total_loss = 0.7 * mse_loss + 0.3 * l1_loss
    
    return {
        'total': total_loss,
        'mse': mse_loss,
        'l1': l1_loss
    }


# ============================================================================
# 训练与评估
# ============================================================================
def train_model(model, train_loader, val_loader, device, num_epochs=150):
    """训练优化模型"""
    print("🚀 开始训练极简版MARS模型...")
    print("📊 损失函数配置:")
    print(f"   MSE权重:     {MSE_WEIGHT}")
    print(f"   L1权重:      {L1_WEIGHT}")
    print(f"   学习率:      {BASE_LR}")
    print(f"   梯度裁剪:    {GRAD_CLIP_NORM}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_metrics = {'mse': 0.0, 'l1': 0.0}
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # 计算损失
            loss_dict = compute_total_loss(outputs, batch_labels)
            loss = loss_dict['total']
            
            loss.backward()
            
            # 梯度裁剪
            total_norm = clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += loss_dict[key].item()

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_metrics = {'mse': 0.0, 'l1': 0.0}
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                
                loss_dict = compute_total_loss(outputs, batch_labels)
                val_loss += loss_dict['total'].item()
                for key in val_metrics:
                    val_metrics[key] += loss_dict[key].item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
            val_metrics[key] /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 检测验证损失异常
        if val_loss > 100.0 or np.isnan(val_loss) or np.isinf(val_loss):
            print(f"\n❌ 验证损失异常: {val_loss:.2e}")
            print("   检测到数值不稳定，停止训练")
            print(f"   MSE: {val_metrics['mse']:.2e}, L1: {val_metrics['l1']:.2e}")
            break
        
        # 打印训练信息
        print(f"Epoch {epoch+1:3d}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f} "
              f"(MSE {train_metrics['mse']:.4f}, L1 {train_metrics['l1']:.4f}) - "
              f"Val Loss: {val_loss:.6f} "
              f"(MSE {val_metrics['mse']:.4f}, L1 {val_metrics['l1']:.4f})")

        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), 'mars_optimized_best_tmp.pth')
                if os.path.exists('mars_optimized_best.pth'):
                    os.remove('mars_optimized_best.pth')
                os.rename('mars_optimized_best_tmp.pth', 'mars_optimized_best.pth')
                print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
            except Exception as exc:
                print(f"⚠️ 模型保存失败: {exc}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), 'mars_optimized_final.pth')
    print("✓ 保存最终模型")
    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """评估模型并输出分关节指标 - 增强版（分析质量依赖性）"""
    print("📊 评估模型性能...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_labels, axis=0)
    
    # 整体指标
    mae = mean_absolute_error(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)

    print("\n测试集整体性能:")
    print(f"MAE:  {mae:.6f} m ({mae*100:.2f} cm)")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f} m ({rmse*100:.2f} cm)")

    # 分轴指标
    print("\n分轴误差:")
    preds_joints = predictions.reshape(-1, NUM_JOINTS, 3)
    gt_joints = ground_truth.reshape(-1, NUM_JOINTS, 3)
    
    for axis_idx, axis_name in enumerate(['X(左右)', 'Y(前后)', 'Z(竖直)']):
        axis_pred = preds_joints[:, :, axis_idx].flatten()
        axis_gt = gt_joints[:, :, axis_idx].flatten()
        axis_mae = mean_absolute_error(axis_gt, axis_pred)
        axis_rmse = np.sqrt(mean_squared_error(axis_gt, axis_pred))
        print(f"{axis_name} - MAE: {axis_mae:.6f}m ({axis_mae*100:.2f}cm), "
              f"RMSE: {axis_rmse:.6f}m ({axis_rmse*100:.2f}cm)")

    # 手部关节专项评估
    print("\n手部关节专项评估:")
    hand_preds = preds_joints[:, HAND_JOINT_INDICES, :]
    hand_gt = gt_joints[:, HAND_JOINT_INDICES, :]
    hand_mae = mean_absolute_error(hand_gt.flatten(), hand_preds.flatten())
    hand_rmse = np.sqrt(mean_squared_error(hand_gt.flatten(), hand_preds.flatten()))
    print(f"手部MAE:  {hand_mae:.6f}m ({hand_mae*100:.2f}cm)")
    print(f"手部RMSE: {hand_rmse:.6f}m ({hand_rmse*100:.2f}cm)")
    
    # 核心部位评估
    print("\n核心部位评估:")
    core_preds = preds_joints[:, CORE_JOINT_INDICES, :]
    core_gt = gt_joints[:, CORE_JOINT_INDICES, :]
    core_mae = mean_absolute_error(core_gt.flatten(), core_preds.flatten())
    core_rmse = np.sqrt(mean_squared_error(core_gt.flatten(), core_preds.flatten()))
    print(f"核心部位MAE:  {core_mae:.6f}m ({core_mae*100:.2f}cm)")
    print(f"核心部位RMSE: {core_rmse:.6f}m ({core_rmse*100:.2f}cm)")
    
    # 误差分布分析
    print("\n误差分布分析:")
    sample_errors = np.sqrt(np.sum((preds_joints - gt_joints)**2, axis=(1,2)))
    print(f"平均误差: {sample_errors.mean():.6f}m ({sample_errors.mean()*100:.2f}cm)")
    print(f"中位数误差: {np.median(sample_errors):.6f}m ({np.median(sample_errors)*100:.2f}cm)")
    print(f"标准差: {sample_errors.std():.6f}m ({sample_errors.std()*100:.2f}cm)")
    print(f"最小误差: {sample_errors.min():.6f}m ({sample_errors.min()*100:.2f}cm)")
    print(f"最大误差: {sample_errors.max():.6f}m ({sample_errors.max()*100:.2f}cm)")
    
    # 分位数分析
    percentiles = [25, 50, 75, 90, 95]
    print(f"\n误差分位数:")
    for p in percentiles:
        val = np.percentile(sample_errors, p)
        print(f"  {p:2d}%: {val:.6f}m ({val*100:.2f}cm)")
    
    # 高质量预测占比
    excellent_threshold = 0.25  # 25cm
    good_threshold = 0.40      # 40cm
    excellent_ratio = (sample_errors < excellent_threshold).sum() / len(sample_errors) * 100
    good_ratio = (sample_errors < good_threshold).sum() / len(sample_errors) * 100
    print(f"\n预测质量分布:")
    print(f"  优秀样本 (误差<{excellent_threshold*100:.0f}cm): {excellent_ratio:.1f}%")
    print(f"  良好样本 (误差<{good_threshold*100:.0f}cm): {good_ratio:.1f}%")

    # 保存预测结果
    np.save('predictions_mars_optimized.npy', predictions)
    print("\n✓ 预测结果已保存: predictions_mars_optimized.npy")
    
    return predictions, ground_truth, mae, mse, rmse


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主函数"""
    print("=" * 70)
    print("MARS骨架提取优化版 v2.0 (数据质量增强)")
    print("=" * 70)
    print("\n🎯 核心优化:")
    print("  1. 移除所有数据过滤和增强")
    print("  2. 极简损失: MSE(0.7) + L1(0.3)")
    print("  3. 空间保留主干: 4×4空间结构")
    print("  4. 提高学习率到1e-3\n")
    
    device = configure_gpu()
    
    # 加载数据（含质量过滤）
    (train_features, val_features, test_features,
     train_labels, val_labels, test_labels) = load_and_preprocess_data()
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        batch_size=32
    )
    
    print("\n🔧 模型构建:")
    model = OptimizedMARSModel(input_channels=5, output_dim=OUTPUT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}\n")

    # 训练模型
    print("\n" + "=" * 70)
    train_model(model, train_loader, val_loader, device, num_epochs=150)

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('mars_optimized_best.pth', map_location=device))
        print("\n✓ 成功加载最佳模型")
    except Exception as exc:
        print(f"\n⚠️ 加载最佳模型失败: {exc}")
        print("使用当前训练后的模型进行评估")

    # 评估模型
    print("\n" + "=" * 70)
    evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("🎉 优化版MARS训练完成!")
    print("=" * 70)
    print("✓ 最佳模型: mars_optimized_best.pth")
    print("✓ 最终模型: mars_optimized_final.pth")
    print("✓ 预测结果: predictions_mars_optimized.npy")
    print("\n📈 预期改进效果:")
    print("  - 平均误差: 35-40cm (相比基线↓25-30%)")
    print("  - 优秀样本(误差<25cm): >30%")
    print("  - 极端失败样本(误差>121cm): <5%")
    print("  - 手部精度: 相比基线提升20-30%")
    print("\n📈 预期改进效果:")
    print("  - 手部关节精度提升: 40-50%")
    print("  - 整体性能: 保持稳定或略有提升")
    print("  - 空间信息保留: 显著改善")


if __name__ == "__main__":
    main()
