# 安装指南

## 系统要求

- Python >= 3.8
- CUDA >= 11.0 (如果使用 GPU)
- 足够的磁盘空间用于数据集和模型检查点

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n nest_ssl python=3.10
conda activate nest_ssl

# 或使用 venv
python -m venv nest_ssl_env
source nest_ssl_env/bin/activate  # Linux/Mac
# 或
nest_ssl_env\Scripts\activate  # Windows
```

### 2. 安装 PyTorch

根据你的 CUDA 版本安装 PyTorch：

```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchaudio
```

### 3. 安装项目依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或安装开发依赖（包括测试工具）
pip install -r requirements-dev.txt
```

### 4. 验证安装

```bash
# 测试 PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 测试 Lightning
python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"

# 测试音频处理库
python -c "import librosa; import soundfile; print('Audio libraries OK')"

# 测试配置管理
python -c "from omegaconf import DictConfig; from hydra import compose, initialize; print('Config libraries OK')"
```

## 常见问题

### Q: 安装 PyTorch 时出错

A: 
1. 确保 Python 版本 >= 3.8
2. 检查 CUDA 版本是否匹配
3. 尝试使用 conda 安装：`conda install pytorch torchaudio -c pytorch`

### Q: librosa 安装失败

A:
1. 确保已安装系统依赖（如 libsndfile）
   - Ubuntu/Debian: `sudo apt-get install libsndfile1`
   - macOS: `brew install libsndfile`
   - Windows: 通常会自动安装
2. 如果仍有问题，尝试：`pip install librosa --no-cache-dir`

### Q: CUDA 不可用

A:
1. 检查 CUDA 是否正确安装：`nvidia-smi`
2. 确保 PyTorch 版本与 CUDA 版本匹配
3. 可以先用 CPU 版本测试：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Q: 内存不足

A:
1. 减少批次大小（batch_size）
2. 使用梯度累积
3. 使用混合精度训练（fp16）

## 可选依赖

某些功能可能需要额外的依赖：

- **Lhotse** (可选): 用于高级数据集处理
  ```bash
  pip install lhotse>=1.31.1
  ```

- **DALI** (可选): 用于加速数据加载
  ```bash
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
  ```

- **Wandb** (可选): 用于实验跟踪
  ```bash
  pip install wandb
  ```

## 下一步

安装完成后，请查看：
- `README.md` - 项目概述
- `QUICK_START.md` - 快速开始指南
- `config/nest_fast-conformer.yaml` - 配置文件示例

