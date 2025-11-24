# 快速上手指南：上传到 Git 仓库

## ✅ 已完成

- Git 仓库已初始化
- 所有文件已添加并提交
- 已创建初始提交

## 🚀 下一步：上传到远程仓库

### 步骤 1: 在 GitHub 创建新仓库

1. 访问 https://github.com/new
2. 输入仓库名称（例如：`nest-ssl-project`）
3. **重要**：不要勾选 "Initialize this repository with a README"（我们已经有了）
4. 点击 "Create repository"

### 步骤 2: 连接并推送

在项目目录下运行以下命令（替换 `YOUR_USERNAME` 和 `REPO_NAME`）：

```bash
# 进入项目目录
cd nest_ssl_project

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 重命名分支为 main（GitHub 默认使用 main）
git branch -M main

# 推送代码
git push -u origin main
```

### 如果使用 SSH（推荐）

如果你配置了 SSH 密钥，可以使用：

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## 📝 示例

假设你的 GitHub 用户名是 `zhile`，仓库名是 `nest-ssl-project`：

```bash
cd nest_ssl_project
git remote add origin https://github.com/zhile/nest-ssl-project.git
git branch -M main
git push -u origin main
```

## ⚠️ 如果遇到问题

### 问题 1: 远程仓库已存在内容

如果远程仓库已经有文件，需要先拉取：

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### 问题 2: 需要身份验证

GitHub 现在要求使用 Personal Access Token 而不是密码：

1. 访问 https://github.com/settings/tokens
2. 生成新 token（选择 `repo` 权限）
3. 使用 token 作为密码

或者配置 SSH 密钥（更安全）。

## 📚 更多信息

查看 `GIT_SETUP.md` 获取详细的 Git 使用指南。

