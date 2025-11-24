# Git 仓库设置指南

## 已完成的步骤

✅ Git 仓库已初始化
✅ 已创建 .gitignore 文件
✅ 已添加所有项目文件
✅ 已创建初始提交

## 下一步：连接到远程仓库

### 方法 1: 在 GitHub 上创建新仓库

1. **在 GitHub 上创建新仓库**
   - 访问 https://github.com/new
   - 输入仓库名称（例如：`nest-ssl-project`）
   - **不要**初始化 README、.gitignore 或 license（我们已经有了）
   - 点击 "Create repository"

2. **连接到远程仓库并推送**

```bash
cd nest_ssl_project

# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为你的实际值）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用 SSH（如果你配置了 SSH 密钥）
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 重命名分支为 main（可选，GitHub 默认使用 main）
git branch -M main

# 推送代码到远程仓库
git push -u origin main
```

### 方法 2: 使用现有的远程仓库

如果你已经有一个远程仓库：

```bash
cd nest_ssl_project

# 添加远程仓库
git remote add origin <你的仓库URL>

# 重命名分支为 main（如果需要）
git branch -M main

# 推送代码
git push -u origin main
```

### 方法 3: 使用 GitLab 或其他 Git 服务

步骤类似，只需要将 URL 替换为相应的服务地址。

## 常用 Git 命令

### 查看状态
```bash
git status
```

### 查看远程仓库
```bash
git remote -v
```

### 添加文件
```bash
git add <文件名>
git add .  # 添加所有更改
```

### 提交更改
```bash
git commit -m "提交信息"
```

### 推送到远程
```bash
git push
git push origin main  # 指定分支
```

### 拉取更新
```bash
git pull
```

### 查看提交历史
```bash
git log
git log --oneline  # 简洁版本
```

## 分支管理

### 创建新分支
```bash
git branch <分支名>
git checkout <分支名>
# 或者
git checkout -b <分支名>  # 创建并切换
```

### 切换分支
```bash
git checkout <分支名>
# 或者（Git 2.23+）
git switch <分支名>
```

### 合并分支
```bash
git checkout main
git merge <分支名>
```

## 注意事项

1. **不要提交大文件**：确保 .gitignore 正确配置，避免提交模型文件、数据文件等
2. **定期提交**：保持提交信息清晰，便于追踪更改
3. **保护敏感信息**：不要提交包含 API 密钥、密码等敏感信息的文件

## 如果遇到问题

### 推送被拒绝
如果远程仓库已有内容，可能需要先拉取：
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### 更改远程仓库 URL
```bash
git remote set-url origin <新的URL>
```

### 查看当前分支
```bash
git branch
```

