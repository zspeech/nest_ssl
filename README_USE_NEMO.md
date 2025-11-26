# 如何使用 NeMo 的 ConformerEncoder

如果你想直接使用 NeMo 的 ConformerEncoder（而不是我们的本地实现），有两种方法：

## 方法 1: 使用环境变量（推荐）

设置环境变量 `USE_NEMO_CONFORMER=true`：

```bash
# Windows PowerShell
$env:USE_NEMO_CONFORMER="true"
python train.py

# Windows CMD
set USE_NEMO_CONFORMER=true
python train.py

# Linux/Mac
export USE_NEMO_CONFORMER=true
python train.py
```

## 方法 2: 修改 serialization.py

在 `nest_ssl_project/core/classes/serialization.py` 中，将：

```python
USE_NEMO_CONFORMER = os.getenv('USE_NEMO_CONFORMER', 'false').lower() == 'true'
```

改为：

```python
USE_NEMO_CONFORMER = True  # 直接使用 NeMo 的 ConformerEncoder
```

## 注意事项

1. **依赖**: 使用 NeMo 的 ConformerEncoder 需要安装完整的 NeMo 库
2. **兼容性**: NeMo 的 ConformerEncoder 应该与我们的代码兼容，因为它也继承自 `NeuralModule`
3. **参数量**: NeMo 的 ConformerEncoder 应该正好是 120M 参数

## 对比参数量

运行对比脚本：

```bash
cd nest_ssl_project
python tools/compare_parameters.py
```

这会比较我们的实现和 NeMo 的实现，并显示详细的参数差异。

