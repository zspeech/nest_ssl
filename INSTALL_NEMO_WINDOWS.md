# Windows 上安装 NeMo 用于对比

## 问题说明

在 Windows 上安装 NeMo 时，以下包需要 C++ 编译器，可能会失败：
- `ctc_segmentation` - CTC 对齐工具（可选）
- `texterrors` - 文本错误计算（可选）
- `megatron_core` - 大模型训练（可选）

**好消息**：这些包对于运行对比脚本 `compare_with_nemo.py` 不是必需的！

## 方法 1: 使用自动安装脚本（最简单，推荐）

### 选项 A: 使用 Python 脚本（跨平台）

```bash
cd nest_ssl_project
python install_nemo_minimal.py
```

### 选项 B: 使用批处理脚本（Windows）

```bash
cd nest_ssl_project
quick_install_nemo.bat
```

脚本会自动：
1. 安装核心依赖
2. 检查 NeMo 路径
3. 安装 NeMo 依赖（跳过需要编译的包）
4. 验证安装

## 方法 2: 手动跳过可选依赖安装

### 步骤 1: 安装基础依赖

```bash
# 先安装基础包
pip install torch torchaudio lightning hydra-core omegaconf
```

### 步骤 2: 从源码安装 NeMo（跳过可选依赖）

```bash
cd NeMo

# 方法 A: 只安装核心部分，跳过 asr 的可选依赖
pip install -e ".[asr-only]" --no-deps

# 然后手动安装依赖（跳过 ctc_segmentation 和 texterrors）
pip install braceexpand editdistance einops jiwer kaldi-python-io librosa marshmallow optuna packaging pyannote.core pyannote.metrics pydub pyloudnorm resampy ruamel.yaml scipy soundfile sox kaldialign whisper_normalizer diskcache

# 安装 NeMo 的核心依赖
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_lightning.txt
pip install -r requirements/requirements_common.txt
```

### 步骤 3: 验证安装

```bash
python -c "from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder; print('✓ NeMo 安装成功')"
```

## 方法 3: 使用 conda（推荐，如果有 conda）

Conda 通常有预编译的包，避免编译问题。

### 步骤 1: 安装 conda（如果还没有）

从 https://www.anaconda.com/download 下载安装

### 步骤 2: 创建新环境

```bash
conda create -n nemo python=3.10
conda activate nemo
```

### 步骤 3: 安装 NeMo

```bash
# 安装 PyTorch（conda 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 NeMo（conda-forge 可能有预编译版本）
conda install -c conda-forge nemo_toolkit

# 或者从源码安装
cd NeMo
pip install -e ".[asr]"
```

## 方法 4: 只安装对比所需的模块（最小安装）

如果只需要运行 `compare_with_nemo.py`，可以只安装必要的模块：

### 步骤 1: 安装基础依赖

```bash
pip install torch torchaudio lightning hydra-core omegaconf
```

### 步骤 2: 设置 PYTHONPATH

```bash
# Windows PowerShell
$env:PYTHONPATH = "C:\Users\zhile\Desktop\Nemo_nest\NeMo;$env:PYTHONPATH"

# Windows CMD
set PYTHONPATH=C:\Users\zhile\Desktop\Nemo_nest\NeMo;%PYTHONPATH%
```

### 步骤 3: 安装最小依赖

```bash
pip install einops packaging braceexpand ruamel.yaml
```

### 步骤 4: 测试导入

```bash
python -c "import sys; sys.path.insert(0, r'C:\Users\zhile\Desktop\Nemo_nest\NeMo'); from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder; print('✓ 导入成功')"
```

## 方法 5: 安装 Visual C++ Build Tools（完整安装）

如果你想完整安装所有依赖：

### 步骤 1: 下载并安装 Visual C++ Build Tools

1. 访问：https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下载 "Build Tools for Visual Studio"
3. 安装时选择 "C++ build tools" 工作负载

### 步骤 2: 重新安装 NeMo

```bash
pip install nemo_toolkit[asr]
```

## 验证对比脚本能否运行

安装完成后，测试对比脚本：

```bash
cd nest_ssl_project
python tools/compare_with_nemo.py
```

如果遇到导入错误，检查：
1. NeMo 路径是否正确添加到 PYTHONPATH
2. 是否安装了必要的依赖

## 常见问题

### Q: 为什么 `ctc_segmentation` 安装失败？

A: 这个包需要 C++ 编译器。对于对比脚本，它不是必需的。可以跳过。

### Q: 对比脚本需要哪些 NeMo 模块？

A: 只需要：
- `nemo.collections.asr.modules.conformer_encoder.ConformerEncoder`
- `nemo.collections.asr.losses.ssl_losses.mlm.MLMLoss`
- `nemo.core.classes.model_pt.ModelPT`（用于加载配置）

### Q: 可以使用 WSL（Windows Subsystem for Linux）吗？

A: 可以！在 WSL 中安装 NeMo 通常更简单，因为 Linux 环境编译更容易。

```bash
# 在 WSL 中
pip install nemo_toolkit[asr]
```

## 推荐方案

**对于快速对比（推荐）**：
1. **首选**：使用方法 1（自动安装脚本）- 最简单
2. **备选**：使用方法 4（最小安装）+ 设置 PYTHONPATH

**对于完整功能**：
1. **首选**：使用方法 3（conda）- 最稳定
2. **备选**：使用方法 5（安装 Build Tools）- 完整功能

