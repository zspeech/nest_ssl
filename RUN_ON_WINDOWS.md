# Windows 上运行 NeMo SSL 训练

## 问题说明

在 Windows 上运行 NeMo 训练脚本时，如果使用 CPU，会遇到以下错误：

```
TypeError: `devices` selected with `CPUAccelerator` should be an int > 0.
```

这是因为 `devices: -1`（使用所有可用 GPU）在 CPU 模式下不被支持。

## 解决方案

### 方法 1: 通过命令行参数覆盖（推荐）

运行训练时，通过命令行参数指定 `devices`：

```bash
# 使用 CPU（1 个核心）
python NeMo/examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.strategy=auto \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path>

# 如果有 GPU，使用 GPU
python NeMo/examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.strategy=auto \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path>
```

### 方法 2: 修改配置文件

如果经常在 Windows CPU 环境下运行，可以临时修改配置文件：

**NeMo 配置文件**: `NeMo/examples/asr/conf/ssl/nest/nest_fast-conformer.yaml`

```yaml
trainer:
  devices: 1  # 改为 1 而不是 -1
  accelerator: cpu  # 明确指定 CPU
  strategy: auto  # 改为 auto 而不是 ddp（DDP 在 Windows 上可能有问题）
```

**nest_ssl_project 配置文件**: `nest_ssl_project/config/nest_fast-conformer.yaml`

同样修改 `trainer` 部分。

### 方法 3: 使用我们的项目（已优化）

我们的 `nest_ssl_project` 项目已经独立，可以直接运行：

```bash
cd nest_ssl_project

# 使用 CPU
python train.py \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.strategy=auto \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path>

# 使用 GPU（如果有）
python train.py \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.strategy=auto \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path>
```

## Windows 特定注意事项

### 1. DDP 策略问题

Windows 上 PyTorch Lightning 的 DDP 策略可能有问题，建议使用：
- `strategy: auto` - 自动选择
- `strategy: null` - 单进程训练

### 2. 多进程数据加载

Windows 上 `num_workers > 0` 可能导致问题，如果遇到错误，可以设置为：

```bash
model.train_ds.num_workers=0
model.validation_ds.num_workers=0
```

### 3. 路径分隔符

Windows 使用反斜杠 `\`，但在 YAML 配置和命令行中，使用正斜杠 `/` 或双反斜杠 `\\` 都可以。

## 完整示例

### 使用 NeMo 运行（CPU）

```bash
python NeMo/examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.strategy=auto \
    trainer.max_steps=10 \
    trainer.val_check_interval=5 \
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json \
    model.train_ds.batch_size=2 \
    model.validation_ds.batch_size=2 \
    model.train_ds.num_workers=0 \
    model.validation_ds.num_workers=0
```

### 使用 nest_ssl_project 运行（CPU）

```bash
cd nest_ssl_project

python train.py \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.strategy=auto \
    trainer.max_steps=10 \
    trainer.val_check_interval=5 \
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json \
    model.train_ds.batch_size=2 \
    model.validation_ds.batch_size=2 \
    model.train_ds.num_workers=0 \
    model.validation_ds.num_workers=0
```

## 检查 GPU 可用性

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

如果输出 `CUDA available: True`，可以使用 GPU；否则使用 CPU。

## 常见错误

### 错误 1: `devices` 必须是正整数

**错误信息**:
```
TypeError: `devices` selected with `CPUAccelerator` should be an int > 0.
```

**解决方法**: 设置 `trainer.devices=1`

### 错误 2: DDP 策略失败

**错误信息**: 多进程相关错误

**解决方法**: 设置 `trainer.strategy=auto` 或 `trainer.strategy=null`

### 错误 3: 数据加载器错误

**错误信息**: `RuntimeError: An attempt has been made to start a new process...`

**解决方法**: 设置 `num_workers=0`

## 性能建议

### CPU 训练
- 使用较小的 `batch_size`（如 2-4）
- 设置 `num_workers=0`（避免多进程开销）
- 使用 `precision=32`（FP32）

### GPU 训练
- 可以使用较大的 `batch_size`（如 8-16）
- 可以使用 `num_workers=4-8`
- 可以使用 `precision=16`（混合精度）

## 总结

在 Windows 上运行 NeMo SSL 训练的关键点：

1. ✅ **设置 `devices=1`** 而不是 `-1`
2. ✅ **使用 `strategy=auto`** 而不是 `ddp`
3. ✅ **设置 `num_workers=0`** 如果遇到多进程问题
4. ✅ **明确指定 `accelerator=cpu`** 或 `accelerator=gpu`

按照以上方法，应该可以在 Windows 上成功运行训练。

