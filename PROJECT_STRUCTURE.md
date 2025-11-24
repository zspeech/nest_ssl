# 项目结构说明

本项目是从 NeMo 中提取的用于运行 `masked_token_pred_pretrain.py` 的最小化项目。

## 核心文件

### 1. 训练脚本
- `train.py` - 主训练脚本（原 `masked_token_pred_pretrain.py`）

### 2. 模型定义
- `models/ssl_models.py` - 包含以下模型类：
  - `SpeechEncDecSelfSupervisedModel` - 基础自监督模型
  - `EncDecMaskedTokenPredModel` - 掩码 token 预测模型
  - `EncDecDenoiseMaskedTokenPredModel` - 去噪掩码 token 预测模型（主要使用的模型）

### 3. 数据集
- `data/ssl_dataset.py` - 包含：
  - `AudioNoiseItem` - 单个样本数据类
  - `AudioNoiseBatch` - 批次数据类
  - `AudioNoiseDataset` - 数据集类
  - `get_audio_noise_dataset_from_config` - 数据集工厂函数

### 4. 神经网络模块
- `modules/ssl_modules/quantizers.py` - `RandomProjectionVectorQuantizer`
- `modules/ssl_modules/multi_softmax_decoder.py` - `MultiSoftmaxDecoder`
- `modules/ssl_modules/masking.py` - `RandomBlockMasking`, `ConvFeatureMaksingWrapper`
- `modules/ssl_modules/augmentation.py` - `MultiSpeakerNoiseAugmentation`

### 5. 损失函数
- `losses/ssl_losses/mlm.py` - `MultiMLMLoss`, `MLMLoss`

### 6. 配置文件
- `config/nest_fast-conformer.yaml` - 模型训练配置

## 依赖关系

本项目仍然依赖 NeMo 的以下核心模块：
- `nemo.core.classes.ModelPT` - 基础模型类
- `nemo.collections.asr.parts.mixins.ASRModuleMixin` - ASR 模块混入
- `nemo.collections.asr.data.audio_to_text_dataset` - 音频到文本数据集基类
- `nemo.collections.asr.modules.*` - 各种 ASR 模块（preprocessor, encoder 等）

## 简化说明

本项目保留了运行训练脚本所需的所有核心功能，但移除了：
- 测试代码
- 文档生成代码
- 其他不相关的模型类
- 其他不相关的数据集类

## 使用方法

1. 确保已安装 NeMo 和相关依赖
2. 运行训练脚本：
```bash
python train.py \
    model.train_ds.manifest_filepath=<path> \
    model.validation_ds.manifest_filepath=<path> \
    ...
```

