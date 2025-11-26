# 安装说明

## ✅ 重要提示

**本项目已经完全独立于 NeMo 框架，不需要安装 NeMo！**

如果遇到 `megatron_core` 安装错误，这是正常的，因为我们不需要它。

## 快速安装

### 1. 安装基础依赖

```bash
cd nest_ssl_project
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; import lightning; print('安装成功！')"
```

## 依赖说明

### 核心依赖（必须）

- **torch** >= 2.0.0 - PyTorch 深度学习框架
- **lightning** >= 2.0.0 - PyTorch Lightning 训练框架
- **hydra-core** >= 1.3.0 - 配置管理
- **omegaconf** >= 2.3.0 - 配置解析
- **librosa** >= 0.10.0 - 音频处理
- **soundfile** >= 0.12.0 - 音频文件读取

### 完整依赖列表

见 `requirements.txt` 文件。

## Windows 用户注意事项

### 如果遇到编译错误

某些包（如 `megatron_core`）在 Windows 上需要 C++ 编译器。**但我们的项目不需要这些包**，可以安全地忽略这些错误。

### 如果遇到音频库问题

```bash
# 如果 soundfile 安装失败，尝试：
pip install soundfile --no-binary soundfile

# 或者使用 conda：
conda install -c conda-forge soundfile
```

## 可选：安装开发依赖

```bash
pip install -r requirements-dev.txt
```

## 验证项目是否正常工作

### 1. 检查导入

```bash
cd nest_ssl_project
python -c "from models.ssl_models import EncDecDenoiseMaskedTokenPredModel; print('✓ 模型导入成功')"
```

### 2. 运行 dummy 数据测试

```bash
cd nest_ssl_project
python train.py \
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_steps=1 \
    trainer.strategy=auto
```

## 如果需要与 NeMo 比较（可选）

如果你需要运行 `tools/compare_with_nemo.py` 来比较我们的实现与 NeMo 的差异，可以：

### 方法 1: 跳过 megatron_core 安装

```bash
# 只安装 NeMo 的核心部分，跳过 megatron_core
pip install nemo_toolkit[asr] --no-deps
pip install -r <(pip show nemo_toolkit | grep Requires | cut -d: -f2 | tr ',' '\n' | grep -v megatron)
```

### 方法 2: 使用 conda（推荐）

```bash
conda install -c conda-forge nemo_toolkit
```

### 方法 3: 在 Linux 上安装（如果有 Linux 环境）

NeMo 在 Linux 上安装更简单，`megatron_core` 可以正常编译。

## 常见问题

### Q: 为什么不需要 NeMo？

A: 我们已经将所有必要的 NeMo 代码提取并本地化了，包括：
- ✅ ModelPT 基类
- ✅ ConformerEncoder
- ✅ AudioToMelSpectrogramPreprocessor
- ✅ 所有 SSL 模块
- ✅ 损失函数

### Q: 如果我想使用 NeMo 的 ConformerEncoder 怎么办？

A: 设置环境变量：
```bash
set USE_NEMO_CONFORMER=true  # Windows
# 或
export USE_NEMO_CONFORMER=true  # Linux/Mac
```

然后安装 NeMo（可能需要跳过 megatron_core）。

### Q: 安装后仍然报错找不到模块？

A: 确保：
1. 在 `nest_ssl_project` 目录下运行
2. Python 路径正确
3. 所有依赖都已安装

## 下一步

安装完成后，查看 `RUN_NEMO_SSL.md` 了解如何运行训练。
