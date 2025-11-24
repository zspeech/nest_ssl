# 重要说明

## 关于依赖

本项目**仍然需要 NeMo 库**才能运行，因为：

1. **核心基类**：`ModelPT`, `ASRModuleMixin` 等是 NeMo 的核心组件
2. **基础模块**：`ConformerEncoder`, `AudioToMelSpectrogramPreprocessor` 等是模型架构的核心
3. **工具函数**：数据集处理、配置管理等工具函数

## 本项目的价值

虽然仍依赖 NeMo，但本项目提供了：

1. **清晰的项目结构**：只包含运行 SSL 训练所需的文件
2. **易于理解**：去除了大量不相关的代码
3. **便于维护**：核心代码集中在一个项目中
4. **便于定制**：可以更容易地修改和扩展功能

## 文件说明

### 核心文件（必须）
- `train.py` - 训练入口
- `models/ssl_models.py` - 模型定义
- `data/ssl_dataset.py` - 数据集
- `config/nest_fast-conformer.yaml` - 配置文件

### 模块文件（必须）
- `modules/ssl_modules/quantizers.py` - 量化器
- `modules/ssl_modules/multi_softmax_decoder.py` - 解码器
- `modules/ssl_modules/masking.py` - 掩码处理
- `modules/ssl_modules/augmentation.py` - 数据增强

### 损失函数（必须）
- `losses/ssl_losses/mlm.py` - MLM 损失

## 如何运行

```bash
# 确保在项目根目录
cd nest_ssl_project

# 运行训练
python train.py \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path> \
    trainer.devices=1 \
    trainer.accelerator="gpu"
```

## 注意事项

1. **路径问题**：如果遇到导入错误，确保 Python 路径正确
2. **配置文件**：配置文件中的 `_target_` 路径指向 `nemo.*`，这是正确的
3. **模块导入**：本地模块使用相对导入，确保项目结构正确

## 进一步简化

如果需要完全独立于 NeMo，需要：
1. 实现 `ModelPT` 的替代品
2. 实现所有 ASR 模块（encoder, preprocessor 等）
3. 实现数据集基类
4. 实现配置管理系统

这需要大量工作，可能不值得。建议保持对 NeMo 的依赖。

