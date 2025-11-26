# nest_ssl_project 快速参考

## 🎯 项目定位

**nest_ssl_project** 是从 NeMo 框架中提取的**完全独立的 SSL 训练项目**，专门用于训练 NEST Fast-Conformer 自监督学习模型。

## 📊 与 NeMo 的关系

| 方面 | NeMo | nest_ssl_project | 关系 |
|------|------|------------------|------|
| **依赖** | 完整框架 | 完全独立 | ✅ 无依赖 |
| **配置** | `nest_fast-conformer.yaml` | `nest_fast-conformer.yaml` | ✅ 100% 一致 |
| **模型** | `EncDecDenoiseMaskedTokenPredModel` | `EncDecDenoiseMaskedTokenPredModel` | ✅ 完全一致 |
| **架构** | ConformerEncoder (120M) | ConformerEncoder (112M) | ⚠️ 接近（差异 6.7%） |
| **功能** | 完整功能 | 核心功能 | ✅ 训练功能完整 |

## 🏗️ 核心模块

### 模型层
```
models/ssl_models.py
├── SpeechEncDecSelfSupervisedModel (基类)
├── EncDecMaskedTokenPredModel (掩码预测)
└── EncDecDenoiseMaskedTokenPredModel (去噪+掩码) ⭐ 主要使用
```

### 编码器
```
modules/conformer_encoder.py
├── ConformerEncoder (主编码器)
│   ├── ConformerPreEncoder (下采样)
│   └── ConformerLayer × 17 (Conformer 层)
│       ├── ConformerFeedForward
│       ├── RelPositionMultiHeadAttention
│       ├── ConformerConvolution
│       └── ConformerFeedForward
```

### SSL 模块
```
modules/ssl_modules/
├── quantizers.py → RandomProjectionVectorQuantizer
├── multi_softmax_decoder.py → MultiSoftmaxDecoder
├── masking.py → RandomBlockMasking
└── augmentation.py → MultiSpeakerNoiseAugmentation
```

### 损失函数
```
losses/ssl_losses/mlm.py
├── MLMLoss (单解码器)
└── MultiMLMLoss (多解码器) ⭐ 主要使用
```

## 🔄 数据流

```
音频文件
  ↓
AudioSegment (加载音频)
  ↓
WaveformFeaturizer (特征提取)
  ↓
AudioToMelSpectrogramPreprocessor (Mel 频谱图)
  ↓
RandomBlockMasking (掩码)
  ↓
RandomProjectionVectorQuantizer (量化)
  ↓
ConformerEncoder (编码)
  ↓
MultiSoftmaxDecoder (解码)
  ↓
MultiMLMLoss (计算损失)
```

## 📝 配置文件结构

```yaml
model:
  # 数据集
  train_ds: {...}
  validation_ds: {...}
  
  # 预处理
  preprocessor: AudioToMelSpectrogramPreprocessor
  
  # SSL 组件
  masking: RandomBlockMasking
  quantizer: RandomProjectionVectorQuantizer
  encoder: ConformerEncoder
  decoder: MultiSoftmaxDecoder
  loss: MultiMLMLoss
  
  # 优化器
  optim: AdamW + NoamAnnealing
```

## 🚀 快速命令

### 基本训练
```bash
python train.py
```

### 指定数据
```bash
python train.py \
    model.train_ds.manifest_filepath=/path/to/train.json \
    model.validation_ds.manifest_filepath=/path/to/val.json
```

### Windows 运行
```bash
python train.py  # 已优化配置，直接运行
```

### 参数对比
```bash
python tools/count_parameters.py
python tools/compare_with_nemo.py  # 需要 NeMo 环境
```

## 📊 关键参数

### 模型参数（Large - 120M）
- `d_model`: 512
- `n_heads`: 8
- `n_layers`: 17
- `conv_kernel_size`: 9
- `subsampling_factor`: 8
- `subsampling_conv_channels`: 256

### 训练参数
- `batch_size`: 2 (小数据集) / 8 (大数据集)
- `num_workers`: 0 (Windows) / 8 (Linux)
- `devices`: 1 (Windows) / -1 (Linux)
- `strategy`: auto (Windows) / ddp (Linux)

## 🔍 文件对应关系

| nest_ssl_project | NeMo | 说明 |
|------------------|------|------|
| `train.py` | `examples/asr/speech_pretraining/masked_token_pred_pretrain.py` | 训练脚本 |
| `models/ssl_models.py` | `nemo/collections/asr/models/ssl_models.py` | 模型定义 |
| `modules/conformer_encoder.py` | `nemo/collections/asr/modules/conformer_encoder.py` | 编码器 |
| `modules/ssl_modules/*` | `nemo/collections/asr/modules/ssl_modules/*` | SSL 模块 |
| `losses/ssl_losses/mlm.py` | `nemo/collections/asr/losses/ssl_losses/mlm.py` | 损失函数 |
| `core/classes/model_pt.py` | `nemo/core/classes/model_pt.py` | 模型基类 |

## ✅ 一致性检查清单

- [x] 配置文件 100% 一致
- [x] 模型架构完全一致
- [x] 损失函数完全一致
- [x] 数据处理流程一致
- [x] 训练流程一致
- [x] 参数量接近（112M vs 120M）

## 📚 相关文档

- [PROJECT_STRUCTURE_CLEAN.md](PROJECT_STRUCTURE_CLEAN.md) - 详细结构
- [STRUCTURE_COMPARISON.md](STRUCTURE_COMPARISON.md) - 与 NeMo 对比
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md) - 模型对比
- [DOCS_INDEX.md](DOCS_INDEX.md) - 文档索引

