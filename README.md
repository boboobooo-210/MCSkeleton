## 推送代码到 GitHub (Simplified)

```bash
# 1. 提交本地更改
git init
git add .
git commit -m "Update project"

# 2. 设置远程仓库并推送
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/boboobooo-210/MCSkeleton.git
git push -u origin main
```

# CRSkeleton - GCN Skeleton Tokenizer

A PyTorch implementation of Graph Convolutional Network (GCN) based skeleton tokenizer for human action recognition.

## Features

- GCN-based skeleton tokenization
- Support for multiple datasets: NTU RGB+D, MARS, MMFI
- Memory-optimized training pipeline
- DVAE (Discrete Variational Autoencoder) integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CRSkeleton.git
cd CRSkeleton
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage

### Training

Train the GCN skeleton tokenizer with memory optimization:

```bash
# 激活conda环境
conda activate pb_final

# 训练（内存优化版本 - 适用于32GB内存）
python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml
```

**内存优化特性:**
- ✅ 批大小优化: 4 (配合梯度累积=2，等效批大小=8)
- ✅ 分组重构损失: 对每个语义组单独计算损失
- ✅ 关节权重优化: 头部、手部、脚部关节权重×2
- ✅ GPU内存管理: 自动内存清理和优化分配
- ✅ 数据加载优化: 2个worker进程，减少内存占用

### Supported Datasets

- **NTU RGB+D**: Human action recognition dataset with skeleton data
- **MARS**: Multi-modal action recognition dataset
- **MMFI**: Multi-modal fitness dataset

### Configuration

Model configurations are stored in the `cfgs/` directory:
- `cfgs/NTU_models/` - NTU RGB+D dataset configurations
- `cfgs/MARS_models/` - MARS dataset configurations  
- `cfgs/MMFI_models/` - MMFI dataset configurations

## Project Structure

```
CRSkeleton/
├── main.py                 # Main training script
├── models/                 # Model implementations
│   ├── GCNSkeletonTokenizer.py
│   ├── Tokenizer.py
│   └── dvae.py
├── datasets/              # Dataset implementations
├── cfgs/                  # Configuration files
├── tools/                 # Training utilities
└── utils/                 # Common utilities
```

## Models

### GCNSkeletonTokenizer
- Graph Convolutional Network for skeleton feature extraction
- Tokenization of skeleton sequences
- Integration with DVAE for reconstruction

### DVAE (Discrete Variational Autoencoder)
- Discrete latent space representation
- Reconstruction loss optimization
- KL divergence regularization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on PointNet++ PyTorch implementation
- Inspired by BERT tokenization mechanisms for skeleton data
# Pomelo

# CRSkeleton - GCN Skeleton Tokenizer

A PyTorch implementation of Graph Convolutional Network (GCN) based skeleton tokenizer for human action recognition.

## Features

- GCN-based skeleton tokenization
- Support for multiple datasets: NTU RGB+D, MARS, MMFI
- Memory-optimized training pipeline
- DVAE (Discrete Variational Autoencoder) integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CRSkeleton.git
cd CRSkeleton
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 常用命令速查 (Quick Reference Commands)

### 1. 训练 Context-Aware 10-Part 模型
```bash
# 激活环境
conda activate pb_final

# 启动训练
python main.py \
    --config cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml \
    --exp_name gcn_skeleton_context_aware_10p
```

### 2. 生成联合部位标注素材 (Joint Annotation GIFs)
用于生成 360° 旋转的肢体组合 GIF，辅助人工标注。
```bash
python tools/generate_joint_annotation_gifs.py \
    --config cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml \
    --checkpoint experiments/gcn_skeleton_context_aware_10p/NTU_models/gcn_skeleton_context_aware_10p/ckpt-best.pth \
    --output_dir annotation_materials_joint \
    --max_batches -1
```

### 3. 生成重构效果对比 (Reconstruction Visualization)
验证模型的重构能力（生成直立的骨架对比 GIF）。
```bash
python visualizations/gif_10p_final/generate_reconstruction_gifs.py
```

## 完整项目执行流程 (Updated 2025.12)

### Phase 1: 模型训练 (Model Training)

#### 1. 训练骨架提取模块 (Skeleton Extractor)
**当前版本**: `Optimized MARS Model v2.0`
**训练脚本**: `models/skeleton_extractor_final.py`
```bash
python models/skeleton_extractor_final.py
```
**模型特点**:
- 架构: SpatialPreservingBackbone + SimplifiedRegressionHead
- 优势: 极简设计，训练稳定，收敛快 (Val Loss ~0.017)
- **可视化验证**: `visualizations/skeleton_extraction_final/vis_skeleton_extractor_final.py`
  - 支持时序平滑 (Temporal Smoothing) 后处理，消除抖动

#### 2. 训练骨架重构模块 (Skeleton Tokenizer)
**当前版本**: `10-Part GCN Tokenizer` (10部位细粒度分词器)
**训练脚本**: `main.py`
```bash
python main.py \
    --config cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml \
    --exp_name gcn_skeleton_context_aware_10p
```
**变更说明**:
- 从原先的 5 语义组升级为 **10 语义组** (10p)，提供更细粒度的动作编码。
- **注意**: 之前的 5p 码本和标注已失效，需重新进行后续步骤。

---

### Phase 2: 数据集处理与Token化 (Data Processing)

#### 3. 运行提取-重构流水线 (Pipeline Demo)
**流水线脚本**: `tools/run_multi_group_pipeline.py`
```bash
python tools/run_multi_group_pipeline.py --mode 10p
```
**功能**: 验证 "视频 -> 骨架提取 -> 10p Tokenizer -> 骨架重构" 的完整链路。
- 确保提取的骨架质量（已通过时序平滑优化）。
- 确保 10p Tokenizer 能正确重构动作。

#### 4. 码本使用率分析 (Codebook Usage Analysis)
**脚本**: 
- MARS (目标域): `tools/skeleton_extraction_reconstruction_saver.py` (需更新适配10p)
- NTU (源域): `analyze_ntu_codebook_usage.py`

**目标**: 
1. **MARS分析**: 统计目标数据集（MARS）在预训练模型上的码本激活情况，识别"码本坍塌"（Codebook Collapse）现象（如仅使用 <10% 的Token）。
2. **NTU分析**: 统计源数据集（NTU）的码本使用率作为基准（Baseline），确认模型本身的表达能力。
3. **对比**: 确定哪些Token是通用的（两边都高频），哪些是特定数据集独有的。

**执行命令**:
```bash
# 分析 MARS 数据集
python tools/skeleton_extraction_reconstruction_saver.py \
    --groups 10 \
    --model_path mars_optimized_best.pth

# 分析 NTU 数据集
python analyze_ntu_codebook_usage.py \
    --config cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml \
    --checkpoint experiments/gcn_skeleton_context_aware_10p/NTU_models/gcn_skeleton_context_aware_10p/ckpt-best.pth
```

#### 5. [关键步骤] 10p码本语义标注 (Codebook Annotation)
**脚本**: `tools/token_codebook_annotator.py` (标注工具) / `tools/generate_annotation_gifs.py` (生成可视化素材)

**分支策略**:
由于 MARS 数据集动作单一（主要是步行），而 NTU 数据集动作丰富，我们采用双分支标注策略：

**分支 A: 目标域优先 (MARS Annotation)**
- **适用场景**: 仅关注 MARS 数据集中的特定动作（如行人重识别、步态分析）。
- **方法**: 仅标注 Step 4 中 MARS 数据集的高频 Token（通常 <100个）。
- **优点**: 工作量极小，快速启动。
- **缺点**: 无法泛化到其他动作。

**分支 B: 源域全面标注 (NTU Annotation) - 推荐**
- **适用场景**: 构建通用的骨架动作生成模型。
- **策略**: **联合部位标注 (Joint Limb Annotation)**。
- **原理**: 10p 模型将四肢拆分为 Arm/Forearm 和 Leg/Foot。单独标注 Forearm 很难判断动作（如"弯曲"可能是举手也可能是敬礼）。将它们组合起来标注（Arm+Forearm）能显著提升语义清晰度。
- **工作量分析**: 
    - 下肢（Leg+Foot）：组合极度集中，Top 50 组合覆盖 95% 数据。
    - 上肢（Arm+Forearm）：组合相对分散，约 500-600 种有效组合。
    - **总计**: 约 1200-1500 个待标注条目（过滤掉极低频噪声后）。
- **执行步骤**:
    1. **生成素材**: 运行 `tools/generate_joint_annotation_gifs.py`。该脚本会自动扫描 NTU 数据集，识别所有出现频率 > 5 次的肢体 Token 组合，并生成对应的 GIF。
    2. **标注**: 观察生成的 GIF（文件名包含频率信息），对组合动作进行描述（例如 Left_Upper_Limb_TokenA_TokenB -> "左手-高举挥手"）。
    3. **输出**: `token_analysis/joint_codebook_annotations.json`

**执行流程**:
1. **生成素材**: 
   ```bash
   python tools/generate_joint_annotation_gifs.py \
       --config cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml \
       --checkpoint experiments/gcn_skeleton_context_aware_10p/NTU_models/gcn_skeleton_context_aware_10p/ckpt-best.pth \
       --output_dir annotation_materials_joint \
       --max_batches -1
   ```
2. **标注**: 参考生成的 GIF 录入语义描述。

---

### Phase 3: LLM 对接与训练 (LLM Integration)

#### 6. 构建指令微调数据集 (Instruction Dataset Construction)
**目标**: 将 Token 序列与文本描述配对。
**数据格式示例**:
```json
{
  "instruction": "Generate a skeleton motion for 'walking forward'",
  "input": "",
  "output": "<group1_token_A> <group2_token_B> ... <group10_token_J> [Next Frame] ..."
}
```
**步骤**:
1. **清洗**: 加载 Step 4 生成的 `.npz` Token 数据。
2. **配对**: 利用 MARS 数据集的标签（Label）或使用 VLM (如 Video-LLaMA) 生成视频的文本描述。
3. **格式化**: 将 `(Text, Token_Sequence)` 转换为 JSONL 格式，适配 LLM 训练框架 (如 LLaMA-Factory)。

#### 7. LLM 训练 (Training)
- **模型选择**: LLaMA-2-7B / Qwen-7B / TinyLlama (视算力而定)。
- **Tokenizer扩展**: 将骨架 Token (如 `<s_0_123>`) 加入 LLM 的词表，或直接使用数字编码。
- **训练任务**: Next Token Prediction (自回归生成)。

#### 8. 推理与可视化 (Inference)
- **输入**: 文本提示 ("A person is waving hand")
- **输出**: 预测的 Token 序列
- **解码**: Token 序列 -> `10p GCN Decoder` -> 骨架坐标 -> `vis_skeleton_extractor_final.py` (复用可视化逻辑) -> GIF动画
python tools/token_codebook_annotator.py
```
**输出**: `token_analysis/codebook_annotations.json`
- 人工标注高频token的语义描述
- 自动读取 `token_schema.json` 识别语义组（10组: head_neck, spine, left_arm, left_forearm, right_arm, right_forearm, left_leg, left_foot, right_leg, right_foot）
- 示例: `{"head_neck": {"25": "点头"}, ...}`

---

### Phase 3: LLM集成与对齐

#### 6. 转换Token字典格式
**转换脚本**: `llm_tools/build_token_dictionary.py`
```bash
# 基本转换（使用默认路径）
python llm_tools/build_token_dictionary.py

# 或指定输入输出文件
python llm_tools/build_token_dictionary.py \
  --input token_analysis/codebook_annotations.json \
  --output llm_tools/token_dictionary.json

# 验证现有字典格式
python llm_tools/build_token_dictionary.py --verify-only
```

**转换逻辑详解**:

输入格式 (`codebook_annotations.json`):
```json
{
  "codebook_annotation": {
    "head_spine": {
      "35": "左倾斜",
      "38": "右倾斜",
      "44": "正常姿态（微微左倾）"
    },
    "left_arm": {
      "128": "向内弯曲（起起抬起抬起）",
      "143": "向内弯曲（向前自然抬起）"
    },
    ...
  },
  "metadata": {
    "total_unique_tokens": 54,
    "annotated_tokens": 54
  }
}
```

输出格式 (`token_dictionary.json`):
```json
{
  "metadata": {
    "total_tokens": 96,
    "group_order": [
      "head_neck",
      "spine",
      "left_arm",
      "left_forearm",
      "right_arm",
      "right_forearm",
      "left_leg",
      "left_foot",
      "right_leg",
      "right_foot"
    ],
    "group_offsets": [0, 32, 64, 112, 160, 208, 256, 320, 384, 448],
    "group_token_sizes": [32, 32, 48, 48, 48, 48, 64, 64, 64, 64],
    "last_updated": "2025-11-06T15:02:41.508910",
    "source": "token_analysis/codebook_annotations.json"
  },
  "groups": {
    "head_neck": {
      "name": "头颈",
      "token_range": [0, 31],
      "annotated_tokens": 9,
      "token_ids": [5, 12, 18, ...]
    },
    "left_forearm": {
      "name": "左前臂与手",
      "token_range": [112, 159],
      "annotated_tokens": 11,
      "token_ids": [128, 133, 147, ...]
    },
    ...
  },
  "tokens": [
    {
      "id": 5,
      "group": "head_neck",
      "description": "微微抬头",
      "frequency": 0
    },
    {
      "id": 128,
      "group": "left_forearm",
      "description": "手掌向前伸展",
      "frequency": 0
    },
    ...
  ]
}
```

**核心转换步骤**:
1. **读取源文件**: 加载 `codebook_annotation` 中的嵌套字典
2. **构建groups元数据**: 
   - 读取 `metadata.group_order` 与 `group_offsets`
   - 生成 `token_range = [offset, offset + size - 1]`
   - 写入语义组显示名称 (`group_display_names`)
   - 统计已标注token数量 (`annotated_tokens`)
3. **扁平化token列表**: 将嵌套的字典转换为数组
   - 每个token包含: `id`, `group`, `description`, `frequency`
   - 按token ID排序，方便查询
4. **验证输出**: 自动检查格式完整性和数据一致性

**转换目的**:
- ✅ **扁平化结构**: 方便LLM快速查询（O(1)复杂度）
- ✅ **动态元数据**: 自动记录 `group_order/group_offsets` 以适配5组或10组Tokenizer
- ✅ **标准化格式**: 统一的JSON schema，易于维护
- ✅ **预留扩展**: `frequency`字段可后续统计token使用频率

**注意事项**:
- 如果修改了 `codebook_annotations.json` 的标注，需要重新运行转换脚本
- 转换脚本会自动验证输出格式的正确性
- 保持 `token_schema.json` 与最新模型一致（schema 变更后需重新导出/标注）
- 若存在自定义Tokenizer，请确认 `metadata.group_offsets` 与训练配置匹配

**详细转换文档**: 📖 `llm_tools/TOKEN_DICTIONARY_CONVERSION.md`

#### 7. 配置LLM API密钥
**支持的国产LLM**: 通义千问(推荐) / GLM-4 / DeepSeek
```bash
# 设置通义千问API Key (推荐)
export DASHSCOPE_API_KEY='your-api-key-here'

# 或设置智谱AI
export ZHIPUAI_API_KEY='your-api-key-here'

# 或设置DeepSeek
export DEEPSEEK_API_KEY='your-api-key-here'
```

**获取API Key**:
- 通义千问: https://help.aliyun.com/zh/dashscope/
- 智谱AI: https://open.bigmodel.cn/
- DeepSeek: https://platform.deepseek.com/

#### 8. 测试LLM集成
**测试脚本**: 使用 `llm_tools/chinese_llm_integration.py`
```bash
# 测试单个样本
python -c "
from llm_tools.chinese_llm_integration import SkeletonLLMAlignment

workflow = SkeletonLLMAlignment(
    llm_provider='qianwen',
    token_dict_path='llm_tools/token_dictionary.json',
    recon_data_dir='data/MARS_recon_tokens'
)

# 测试token序列
result = workflow.process_single_sample([125, 252, 327, 489, 608])
print(f'Token: {result[\"token_sequence\"]}')
print(f'描述: {result[\"llm_description\"]}')
print(f'耗时: {result[\"processing_time\"]:.2f}秒')
"
```

**预期输出**:
```
Token: [125, 252, 327, 489, 608]
描述: 人物身体稍微向左倾斜，双臂自然弯曲下垂，双腿直立站立。
耗时: 0.87秒
```

#### 9. 批量处理MARS数据集
**批量处理脚本**: `llm_tools/batch_process_all.py`
```bash
python llm_tools/batch_process_all.py
```

**交互式选项**:
1. 小批量测试 (100个样本, ~2分钟)
2. 中批量验证 (500个样本, ~8分钟)
3. 完整处理 (7,984个样本, ~2小时)

**输出**: `llm_tools/batch_XXX_aligned.json`
```json
[
  {
    "token_sequence": [125, 252, 327, 489, 608],
    "llm_description": "人物身体稍微向左倾斜，双臂自然弯曲下垂，双腿直立站立。",
    "processing_time": 0.87
  },
  ...
]
```

#### 10. 可视化验证LLM描述准确性
**可视化脚本**: `visualizations/skeleton_extractor/vis_mars_recon_tokens.py`
```bash
# 查看指定样本的骨架可视化
python visualizations/skeleton_extractor/vis_mars_recon_tokens.py --split test --index 1

# 叠加模式(原始 vs 重构)
python visualizations/skeleton_extractor/vis_mars_recon_tokens.py --split test --index 1 --overlay
```

**功能**:
- 3D骨架可视化 (原始 vs 基础重构 vs 最终重构)
- 显示token序列和VQ损失
- 按5个语义组着色
- 计算重构误差

**验证流程**:
```
1. 运行可视化 → 人眼观察骨架姿态
2. 查看对应的LLM描述
3. 验证描述是否准确匹配骨架
4. 如不准确 → 调整prompt或补充token标注
5. 重新运行LLM处理
```

---

### Phase 4: 数据集构建(可选)

#### 11. 生成LLM微调训练数据
**生成脚本**: 使用 `SkeletonLLMAlignment.create_token_llm_training_data()`
```bash
python -c "
from llm_tools.chinese_llm_integration import SkeletonLLMAlignment

workflow = SkeletonLLMAlignment(llm_provider='qianwen')
workflow.create_token_llm_training_data(
    num_samples=500,
    output_path='llm_tools/token_llm_training_data.jsonl'
)
"
```

**输出格式** (JSONL):
```json
{"instruction": "请描述以下骨架姿态token代表的动作", "input": "Token序列: [125, 252, 327, 489, 608]", "output": "人物身体稍微向左倾斜..."}
{"instruction": "请描述以下骨架姿态token代表的动作", "input": "Token序列: [44, 218, 265, 489, 608]", "output": "人物直立站姿..."}
```

**用途**: 微调LLM使其直接理解token序列，无需码本查询

---

## 核心文件清单 (Core File Inventory)

### 1. 模型定义 (Model Definitions)
- **骨架提取器 (Skeleton Extractor)**: `models/skeleton_extractor_final.py`
  - 用于从视频或原始数据中提取高质量骨架。
- **骨架分词器 (Skeleton Tokenizer)**: `models/GCNSkeletonTokenizer_10p.py`
  - **Context-Aware 10-Part** 版本，负责将骨架编码为离散 Token 并重构。
  - 包含 `SkeletonGraph` 定义（10个语义组）和 `ST_GCN_Layer`。

### 2. 训练脚本 (Training Scripts)
- **训练提取器**: `models/skeleton_extractor_final.py` (直接运行)
- **训练分词器**: `main.py`
  - 配合配置文件: `cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml`

### 3. 数据处理与流水线 (Pipeline & Data Processing)
- **多组流水线演示**: `tools/run_multi_group_pipeline.py`
  - 验证 "输入 -> 提取 -> Token化 -> 重构" 的完整流程。
  - 核心逻辑实现: `tools/multi_group_skeleton_pipeline.py`
- **码本使用率分析**: `analyze_ntu_codebook_usage.py`
  - 统计 NTU 数据集上的 Token 分布，检测码本坍塌。
- **MARS 数据集重构保存**: `tools/skeleton_extraction_reconstruction_saver.py`
  - 将 MARS 数据集处理为 Token 序列并保存为 `.npz`。

### 4. 标注与可视化工具 (Annotation & Visualization)
- **联合部位标注素材生成**: `tools/generate_joint_annotation_gifs.py`
  - **功能**: 扫描数据集，生成肢体组合（如左臂+左前臂）的 360° 旋转 GIF。
  - **用途**: 辅助人工进行语义标注。
- **重构效果可视化**: `visualizations/gif_10p_final/generate_reconstruction_gifs.py`
  - **功能**: 生成 "原始骨架 vs 重构骨架" 的对比 GIF。
  - **特点**: 修正了坐标系（直立显示），支持 Context-Aware 模型的字典输出。
- **标注工具**: `tools/token_codebook_annotator.py`
  - 用于录入和管理 Token 的语义描述。

### 5. LLM 集成 (LLM Integration)
- **Token 字典构建**: `llm_tools/build_token_dictionary.py`
  - 将标注好的 JSON 转换为 LLM 可读的字典格式。
- **LLM 对齐核心**: `llm_tools/chinese_llm_integration.py`
  - 实现 "Token 序列 -> 自然语言描述" 的转换逻辑。
- **批量处理**: `llm_tools/batch_process_all.py`

### 6. 配置文件 (Configurations)
- **NTU 10-Part Context-Aware**: `cfgs/NTU_models/gcn_skeleton_context_aware_10p.yaml`
- **数据集配置**: `cfgs/dataset_configs/NTU_skeleton_raw.yaml`

---

## 技术架构：码本-LLM对齐

### 对齐原理
```
骨架(25,3) → GCN编码 → Token序列[5个] → 码本查询 → 语义描述 → LLM理解 → 自然语言
```

### 关键机制
1. **码本建立**: 人工标注token语义 (54个高频token已标注)
2. **查询映射**: Token ID → 语义描述 (如: 125 → "左倾斜")
3. **Prompt构建**: 将语义描述转换为结构化prompt
4. **LLM理解**: 综合各部位描述，生成完整动作描述

### 示例流程
```python
# Token序列
[125, 252, 327, 489, 608]

# 码本查询
head_spine: 125 → "左倾斜"
left_arm: 252 → "自然弯曲"
right_arm: 327 → "自然弯曲"
left_leg: 489 → "站立（直立）"
right_leg: 608 → "站立"

# LLM生成
"人物身体稍微向左倾斜，双臂自然弯曲下垂，双腿直立站立。"
```

**完整技术文档**: 📖 `llm_tools/CODEBOOK_LLM_ALIGNMENT.md`

---

## 成本估算

### LLM处理成本 (通义千问 qwen-turbo)
- 单样本: ~250 tokens × 0.008元/千tokens = 0.002元
- 100样本: 0.2元 (~2分钟)
- 7,984样本(全部测试集): ~16元 (~2小时)
- **免费额度**: 100万tokens/月 (足够处理40,086个样本)

---

## 快速开始

### 最小可行流程
```bash
# 1. 假设模型已训练完成
# 2. 配置API密钥
export DASHSCOPE_API_KEY='your-key'

# 3. 测试LLM集成
python -c "from llm_tools.chinese_llm_integration import SkeletonLLMAlignment; \
           workflow = SkeletonLLMAlignment(llm_provider='qianwen'); \
           result = workflow.process_single_sample([125, 252, 327, 489, 608]); \
           print(result['llm_description'])"

# 4. 可视化验证
python visualizations/skeleton_extractor/vis_mars_recon_tokens.py --split test --index 1

# 5. 批量处理
python llm_tools/batch_process_all.py
# 选择: 1 (100样本测试)
```

---
```



**内存优化特性:**
- ✅ 批大小优化: 4 (配合梯度累积=2，等效批大小=8)
- ✅ 分组重构损失: 对每个语义组单独计算损失
- ✅ 关节权重优化: 头部、手部、脚部关节权重×2
- ✅ GPU内存管理: 自动内存清理和优化分配
- ✅ 数据加载优化: 2个worker进程，减少内存占用

### Supported Datasets

- **NTU RGB+D**: Human action recognition dataset with skeleton data
- **MARS**: Multi-modal action recognition dataset
- **MMFI**: Multi-modal fitness dataset

### Configuration

Model configurations are stored in the `cfgs/` directory:
- `cfgs/NTU_models/` - NTU RGB+D dataset configurations
- `cfgs/MARS_models/` - MARS dataset configurations  
- `cfgs/MMFI_models/` - MMFI dataset configurations

## Project Structure

```
CRSkeleton/
├── main.py                 # Main training script
├── models/                 # Model implementations
│   ├── GCNSkeletonTokenizer.py
│   ├── Tokenizer.py
│   └── dvae.py
├── datasets/              # Dataset implementations
├── cfgs/                  # Configuration files
├── tools/                 # Training utilities
└── utils/                 # Common utilities
```

## Models

### GCNSkeletonTokenizer
- Graph Convolutional Network for skeleton feature extraction
- Tokenization of skeleton sequences
- Integration with DVAE for reconstruction

### DVAE (Discrete Variational Autoencoder)
- Discrete latent space representation
- Reconstruction loss optimization
- KL divergence regularization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on PointNet++ PyTorch implementation
- Inspired by BERT tokenization mechanisms for skeleton data

