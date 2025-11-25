# 下一步工作

## 当前状态

已创建大部分核心模块和辅助功能。项目结构基本完整，但还需要完成以下工作：

## 需要完成的任务

### 1. 更新从 NeMo 复制的文件的导入路径

以下文件已从 NeMo 复制，但需要更新所有导入路径：

- `modules/audio_preprocessing.py`
- `modules/conformer_encoder.py`
- `modules/spectrogram_augmentation.py`
- `data/audio_to_text.py`
- `parts/preprocessing/features.py`

**操作步骤：**
1. 打开每个文件
2. 将所有 `from nemo.*` 导入替换为相应的本地导入
3. 确保所有依赖都能正确解析

### 2. 创建缺失的辅助模块

可能需要创建以下模块（如果从 NeMo 复制的文件需要它们）：

- `core/classes/neural_module.py` - 如果还没有
- `core/classes/common.py` - 包含 `PretrainedModelInfo`, `typecheck` 等
- 其他被引用的工具函数

### 3. 修复循环导入问题

检查并修复任何循环导入问题。

### 4. 测试导入

运行以下命令测试所有导入是否正确：

```python
# 在项目根目录运行
python -c "from models.ssl_models import EncDecDenoiseMaskedTokenPredModel; print('Import successful')"
```

### 5. 更新配置文件

确保 `config/nest_fast-conformer.yaml` 中的所有 `_target_` 路径都指向正确的本地模块。

### 6. 测试训练脚本

尝试运行训练脚本，修复遇到的任何错误：

```bash
python train.py --config-path=config --config-name=nest_fast-conformer
```

## 快速检查清单

- [ ] 所有 `nemo.*` 导入已替换
- [ ] 所有模块可以正确导入
- [ ] 配置文件路径正确
- [ ] 训练脚本可以运行（至少到模型初始化）
- [ ] 没有循环导入错误

## 常见问题

### Q: 如果遇到 "ModuleNotFoundError" 怎么办？

A: 
1. 检查导入路径是否正确
2. 确保所有必要的 `__init__.py` 文件存在
3. 检查 Python 路径设置

### Q: 如果遇到循环导入怎么办？

A:
1. 使用延迟导入（在函数内部导入）
2. 重构代码结构
3. 合并相关模块

### Q: 某些功能在简化版本中不可用怎么办？

A:
- 对于可选功能（如 Lhotse、DALI），可以暂时禁用或显示警告
- 对于必需功能，需要实现简化版本或从 NeMo 复制相关代码

## 建议的工作流程

1. **先修复导入错误** - 确保所有模块可以导入
2. **再修复运行时错误** - 确保模型可以实例化
3. **最后测试训练流程** - 确保整个训练流程可以运行

## 需要帮助？

如果遇到问题，可以：
1. 检查 `PROGRESS.md` 了解已完成的工作
2. 查看 NeMo 原始代码作为参考
3. 逐步添加缺失的功能
