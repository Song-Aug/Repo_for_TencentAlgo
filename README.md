# GenRank

GenRank是一个基于深度学习的通用推荐排序模型，专为序列推荐和多模态特征融合而设计。该项目实现了一个基于Transformer架构的基线模型，支持多种特征类型的处理和融合。

## 🚀 特性

- **多模态特征支持**: 支持稀疏特征、连续特征、数组特征和嵌入特征
- **序列建模**: 基于Transformer的序列推荐模型
- **Flash Attention**: 集成Flash Attention优化，提升训练效率
- **灵活的特征工程**: 支持用户和物品的多种特征类型
- **高效的数据处理**: 优化的数据加载和批处理机制

## 📁 项目结构

```
GenRank/
├── main.py              # 主训练脚本
├── model.py             # 基线模型实现
├── model_rqvae.py       # RQVAE模型实现
├── dataset.py           # 数据集处理
├── run.sh              # 运行脚本
├── TencentGR_1k/       # 数据集目录
│   ├── seq.jsonl       # 用户行为序列数据
│   ├── indexer.pkl     # ID映射索引
│   ├── item_feat_dict.json  # 物品特征字典
│   ├── seq_offsets.pkl # 序列偏移量
│   └── creative_emb/   # 多模态嵌入特征
└── README.md           # 项目说明
```

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm
- tensorboard

## 📦 安装依赖

```bash
pip install torch numpy tqdm tensorboard
```

## 🏃‍♂️ 快速开始

### 1. 数据准备

确保数据集放在正确的目录下：
- `TencentGR_1k/seq.jsonl`: 用户行为序列数据
- `TencentGR_1k/indexer.pkl`: ID映射文件
- `TencentGR_1k/item_feat_dict.json`: 物品特征字典
- `TencentGR_1k/creative_emb/`: 多模态嵌入特征

### 2. 设置环境变量

```bash
export TRAIN_DATA_PATH="./TencentGR_1k"
export TRAIN_LOG_PATH="./logs"
export TRAIN_TF_EVENTS_PATH="./tensorboard"
export TRAIN_CKPT_PATH="./checkpoints"
```

### 3. 训练模型

```bash
# 使用默认参数训练
python main.py

# 或使用自定义参数
python main.py --batch_size 64 --lr 0.001 --num_epochs 5 --hidden_units 64
```

### 4. 使用脚本运行

```bash
chmod +x run.sh
./run.sh
```

## 🔧 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--maxlen` | 101 | 序列最大长度 |
| `--hidden_units` | 32 | 隐藏层维度 |
| `--num_blocks` | 1 | Transformer块数量 |
| `--num_epochs` | 3 | 训练轮数 |
| `--num_heads` | 1 | 注意力头数 |
| `--dropout_rate` | 0.2 | Dropout比率 |
| `--device` | 'cuda' | 设备类型 |
| `--mm_emb_id` | ['81'] | 多模态特征ID |

## 📊 数据格式

### 序列数据格式 (seq.jsonl)
每行包含一个用户的行为序列：
```json
[
    [user_id, item_id, user_feature, item_feature, action_type, timestamp],
    ...
]
```

### 特征类型支持
- **稀疏特征 (Sparse)**: 类别特征，如用户性别、物品类别
- **连续特征 (Continual)**: 数值特征，如用户年龄、物品价格
- **数组特征 (Array)**: 多值特征，如用户兴趣列表
- **嵌入特征 (Embedding)**: 预训练的多模态特征向量

## 🏗️ 模型架构

GenRank采用基于Transformer的序列推荐架构：

1. **特征嵌入层**: 将不同类型的特征转换为统一的嵌入表示
2. **位置编码**: 为序列添加位置信息
3. **Transformer编码器**: 使用自注意力机制建模序列依赖关系
4. **预测层**: 输出物品的预测得分

## 🔍 核心组件

### BaselineModel
- 主要的推荐模型实现
- 支持多种特征类型的融合
- 集成Flash Attention优化

### MyDataset
- 高效的数据加载器
- 支持动态批处理和特征对齐
- 优化的内存使用

### FlashMultiHeadAttention
- 优化的多头注意力实现
- 兼容PyTorch 2.0+ Flash Attention
- 降级支持标准注意力机制

## 📈 监控和日志

训练过程中会生成以下输出：
- **TensorBoard日志**: 训练和验证损失曲线
- **训练日志**: 详细的训练过程记录
- **模型检查点**: 定期保存的模型状态

查看训练进度：
```bash
tensorboard --logdir=./tensorboard
```

## 💡 使用技巧

1. **内存优化**: 对于大型数据集，建议调整`batch_size`和`maxlen`参数
2. **特征选择**: 通过`mm_emb_id`参数选择合适的多模态特征
3. **超参数调优**: 根据数据集特点调整`hidden_units`、`num_blocks`等参数
4. **GPU加速**: 确保安装CUDA版本的PyTorch以获得最佳性能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用 [LICENSE](LICENSE) 许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目为研究和学习目的开发，请确保在使用时遵循相关的数据使用协议和隐私保护规定。
