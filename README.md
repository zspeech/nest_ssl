# NEST SSL Project

一个从 NeMo 框架中提取的独立项目，专门用于训练**自监督学习的去噪掩码 Token 预测模型**（Denoising Masked Token Prediction）。本项目包含了运行 `masked_token_pred_pretrain.py` 所需的所有核心代码，移除了不必要的依赖。

## 📋 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [项目状态](#项目状态)
- [常见问题](#常见问题)
- [许可证](#许可证)

## ✨ 功能特性

- **自监督学习模型**: 实现了 `EncDecDenoiseMaskedTokenPredModel`，用于语音自监督预训练
- **去噪训练**: 支持带噪声的音频数据进行去噪训练
- **掩码 Token 预测**: 实现了掩码语言模型（MLM）风格的训练目标
- **独立运行**: 不依赖完整的 NeMo 框架，可以独立运行
- **简化代码**: 只保留运行训练所需的核心功能

## 📁 项目结构

```
nest_ssl_project/
├── train.py                 # 主训练脚本
├── models/                  # 模型定义
│   └── ssl_models.py        # SSL 模型类
├── data/                    # 数据集相关
│   ├── ssl_dataset.py       # SSL 数据集
│   └── audio_to_text_dataset.py  # 音频数据集工具
├── modules/                 # 神经网络模块
│   └── ssl_modules/         # SSL 专用模块
│       ├── quantizers.py    # 向量量化器
│       ├── masking.py       # 掩码模块
│       ├── multi_softmax_decoder.py  # 多 softmax 解码器
│       └── augmentation.py  # 数据增强
├── losses/                  # 损失函数
│   └── ssl_losses/
│       └── mlm.py           # MLM 损失
├── config/                  # 配置文件
│   └── nest_fast-conformer.yaml  # 模型配置
├── core/                    # 核心基类
│   ├── classes/             # 模型基类
│   └── neural_types/        # 神经网络类型
├── parts/                   # 辅助模块
│   ├── mixins/              # 混入类
│   └── preprocessing/       # 预处理
├── common/                  # 通用工具
│   ├── data/                # 数据工具
│   └── parts/               # 预处理工具
├── utils/                   # 工具函数
│   ├── logging.py          # 日志
│   ├── exp_manager.py      # 实验管理
│   └── config.py           # 配置工具
├── requirements.txt         # Python 依赖
└── README.md               # 本文件
```

## 🚀 安装

### 系统要求

- Python >= 3.8
- CUDA >= 11.0 (如果使用 GPU)
- 足够的磁盘空间用于数据集和模型检查点

### 安装步骤

1. **克隆或下载项目**

```bash
cd nest_ssl_project
```

2. **创建虚拟环境（推荐）**

```bash
# 使用 conda
conda create -n nest_ssl python=3.10
conda activate nest_ssl

# 或使用 venv
python -m venv nest_ssl_env
source nest_ssl_env/bin/activate  # Linux/Mac
nest_ssl_env\Scripts\activate     # Windows
```

3. **安装 PyTorch**

根据你的 CUDA 版本安装 PyTorch：

```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchaudio
```

4. **安装项目依赖**

```bash
pip install -r requirements.txt
```

详细的安装说明请参考 [INSTALL.md](INSTALL.md)。

## 🏃 快速开始

### 1. 准备数据

准备训练数据的 manifest 文件（JSON 格式），每行一个样本：

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 10.5, "text": "transcription"}
{"audio_filepath": "/path/to/audio2.wav", "duration": 8.3, "text": "transcription"}
```

同样准备噪声数据的 manifest 文件（可选，用于数据增强）。

### 2. 运行训练

```bash
python train.py \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.train_ds.noise_manifest=/path/to/noise_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    model.validation_ds.noise_manifest=/path/to/noise_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```

## 📝 配置说明

训练配置通过 Hydra 管理，主要配置文件位于 `config/nest_fast-conformer.yaml`。

### 主要配置项

- **模型配置** (`model`): 模型架构、预处理器、编码器、解码器等
- **数据配置** (`model.train_ds`, `model.validation_ds`): 数据集路径、批次大小等
- **训练配置** (`trainer`): 设备、epochs、学习率等
- **优化器配置** (`model.optim`): 优化器类型、学习率调度等
- **实验管理** (`exp_manager`): 日志、检查点保存等

### 常用配置示例

```bash
# 单 GPU 训练
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=50

# 多 GPU 训练（DDP）
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100

# 自定义学习率
python train.py \
    model.train_ds.manifest_filepath=train.json \
    model.optim.lr=0.0001 \
    model.optim.sched.warmup_steps=1000
```

## 💡 使用示例

### 基本训练

```bash
python train.py \
    --config-path=config \
    --config-name=nest_fast-conformer \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    model.train_ds.noise_manifest=data/noise_manifest.json \
    model.validation_ds.manifest_filepath=data/val_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```

### 从检查点恢复训练

```bash
python train.py \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=200 \
    model.restore_from=/path/to/checkpoint.nemo
```

### 使用 WandB 记录实验

```bash
python train.py \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="my_experiment" \
    exp_manager.wandb_logger_kwargs.project="ssl_pretraining"
```

## 📊 项目状态

**✅ 项目已完成！**

当前状态：

- ✅ 核心模型实现完成
- ✅ 数据集加载功能完成
- ✅ 训练脚本可用
- ✅ 所有 NeMo 依赖已移除
- ✅ 项目完全独立运行
- ✅ 文档完整

项目已完全从 NeMo 框架中剥离，可以独立运行。详细进度请参考 [PROGRESS.md](PROGRESS.md) 和 [COMPLETION_STATUS.md](COMPLETION_STATUS.md)。

## ❓ 常见问题

### Q: 如何检查 CUDA 是否可用？

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Q: 内存不足怎么办？

- 减少 `batch_size`
- 使用梯度累积
- 启用混合精度训练（在配置中设置）

### Q: 如何查看训练日志？

训练日志默认保存在 `nemo_experiments/` 目录下，或使用 TensorBoard：

```bash
tensorboard --logdir=nemo_experiments
```

### Q: 支持哪些音频格式？

支持常见的音频格式：WAV、MP3、FLAC、OPUS 等。

### Q: 如何自定义模型架构？

修改 `config/nest_fast-conformer.yaml` 中的模型配置，或创建新的配置文件。

更多问题请参考 [INSTALL.md](INSTALL.md) 或查看项目文档。

## 📚 相关文档

- [INSTALL.md](INSTALL.md) - 详细安装指南
- [PROGRESS.md](PROGRESS.md) - 项目开发进度
- [NEXT_STEPS.md](NEXT_STEPS.md) - 下一步工作计划
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构说明
- [IMPORTANT_NOTES.md](IMPORTANT_NOTES.md) - 重要注意事项

## 🤝 贡献

本项目是从 NeMo 框架中提取的简化版本。如需贡献：

1. 确保代码符合项目风格
2. 添加必要的测试
3. 更新相关文档

## 📄 许可证

本项目基于 Apache License 2.0 许可证。详见 LICENSE 文件。

## 🙏 致谢

本项目基于 NVIDIA NeMo 框架开发。感谢 NeMo 团队提供的优秀框架。

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。

---

**注意**: 本项目仍在开发中，部分功能可能不完整。使用前请仔细阅读 [IMPORTANT_NOTES.md](IMPORTANT_NOTES.md)。
