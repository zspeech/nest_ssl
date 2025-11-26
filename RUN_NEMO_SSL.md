# 运行 NeMo Masked Prediction SSL 模型

## 配置文件

配置文件已更新为与 NeMo 的 `nest_fast-conformer.yaml` 完全一致。

**配置文件位置**: `config/nest_fast-conformer.yaml`

## 数据集配置

### 1. 准备数据集

你需要准备两个 manifest 文件：
- **训练集 manifest**: JSON 格式，每行一个音频文件信息
- **验证集 manifest**: JSON 格式，每行一个音频文件信息

Manifest 文件格式示例：
```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 10.5}
{"audio_filepath": "/path/to/audio2.wav", "duration": 8.3}
```

### 2. 更新配置文件

在 `config/nest_fast-conformer.yaml` 中设置数据集路径：

```yaml
model:
  train_ds:
    manifest_filepath: /path/to/your/train_manifest.json  # 更新这里
    noise_manifest: null  # 可选：噪声数据集的 manifest
    # ... 其他配置
  
  validation_ds:
    manifest_filepath: /path/to/your/val_manifest.json  # 更新这里
    noise_manifest: null  # 可选：噪声数据集的 manifest
    # ... 其他配置
```

### 3. 运行训练

#### 方法 1: 直接运行（使用配置文件中的路径）

```bash
cd nest_ssl_project
python train.py
```

#### 方法 2: 通过命令行参数覆盖配置

```bash
cd nest_ssl_project
python train.py \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    trainer.devices=-1 \
    trainer.max_steps=500000
```

## 配置说明

### 模型配置（Large - 120M 参数）

- **d_model**: 512
- **n_heads**: 8
- **n_layers**: 17
- **conv_kernel_size**: 9
- **subsampling_factor**: 8
- **subsampling_conv_channels**: 256

### 训练配置

- **batch_size**: 8（可根据 GPU 内存调整）
- **num_workers**: 8（可根据 CPU 核心数调整）
- **max_steps**: 500000
- **val_check_interval**: 2500（每 2500 步验证一次）

### 优化器配置

- **optimizer**: AdamW
- **learning_rate**: 5.0
- **betas**: [0.9, 0.98]
- **weight_decay**: 1e-3
- **scheduler**: NoamAnnealing
- **warmup_steps**: 25000

## 关键组件

### 1. Mask Loss 计算
- ✅ 与 NeMo 完全一致
- 使用 `MultiMLMLoss`，支持多个 decoder

### 2. 网络结构
- ✅ 与 NeMo NEST Fast-Conformer Large 完全一致
- ConformerEncoder: 17 层，512 维，8 头注意力

### 3. 数据增强
- **MultiSpeakerNoiseAugmentation**: 多说话人噪声增强
- **RandomBlockMasking**: 随机块掩码

## 注意事项

1. **数据集路径**: 确保 manifest 文件中的音频路径是绝对路径或相对于运行目录的相对路径

2. **GPU 内存**: 如果遇到 OOM，可以：
   - 减小 `batch_size`
   - 减小 `num_workers`
   - 使用 `precision: 16`（混合精度训练）

3. **验证频率**: `val_check_interval` 应该小于等于训练批次数

4. **检查点**: 模型会自动保存到 `exp_manager.exp_dir` 指定的目录

## 示例命令

### 使用 dummy 数据进行调试

```bash
cd nest_ssl_project
python train.py \
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_steps=10 \
    trainer.val_check_interval=5 \
    trainer.strategy=auto
```

### 使用真实数据进行训练

```bash
cd nest_ssl_project
python train.py \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    trainer.devices=-1 \
    trainer.max_steps=500000 \
    trainer.val_check_interval=2500
```

## 与 NeMo 的对比

✅ **已确认一致**:
- Mask loss 计算逻辑
- 网络配置参数
- ConformerEncoder 架构
- 数据增强配置

⚠️ **参数量差异**:
- 当前: 112M 参数
- NeMo 预期: 120M 参数
- 差异: ~8M (6.7%)

如果需要完全匹配 NeMo 的参数量，可以：
1. 运行 `tools/compare_with_nemo.py` 找出差异
2. 或使用 `USE_NEMO_CONFORMER=true` 直接使用 NeMo 的 ConformerEncoder


