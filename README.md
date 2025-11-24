# NEST SSL Project

这是一个从 NeMo 项目中提取的简化版本，专门用于运行 `masked_token_pred_pretrain.py` 脚本。

## 项目结构

```
nest_ssl_project/
├── models/          # 模型定义
├── data/            # 数据集相关
├── modules/         # 神经网络模块
├── losses/          # 损失函数
├── config/          # 配置文件
├── utils/           # 工具函数
└── train.py         # 主训练脚本
```

## 使用方法

```bash
python train.py \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```

## 依赖

需要安装 NeMo 和相关依赖。本项目只包含运行脚本所需的最小代码集合。

