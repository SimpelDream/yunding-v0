#!/bin/bash

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 启用错误检查
set -euo pipefail

# ANSI颜色代码
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 DVC
if ! command -v dvc &> /dev/null; then
    echo -e "\033[31m❌ 未找到 DVC，请先安装: pip install dvc\033[0m"
    exit 1
fi
echo -e "\033[32m✅ 已安装 $(dvc --version)\033[0m"

# 检查 Git
if ! command -v git &> /dev/null; then
    echo "错误: Git 未安装，请先安装 Git"
    exit 1
fi

# 检查数据集目录
if [ ! -d "datasets/raw" ]; then
    echo -e "${RED}❌ datasets/raw 目录不存在，请先录制数据${NC}"
    exit 1
fi

# 检查 DVC 是否初始化
if [ ! -d .dvc ]; then
    echo "📦 正在初始化DVC..."
    if ! dvc init; then
        echo -e "${RED}❌ DVC初始化失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ DVC初始化成功${NC}"
fi

# 拉取数据
echo "拉取数据..."
dvc pull || {
    echo "错误: DVC pull 失败"
    exit 1
}

# 更新数据
echo "更新数据..."
dvc repro || {
    echo "错误: DVC repro 失败"
    exit 1
}

# 询问Google Drive remote ID
read -p "请输入Google Drive remote ID (可选): " gdrive_id

# 添加Google Drive remote
if [ ! -z "$gdrive_id" ]; then
    echo "🔗 正在添加Google Drive remote..."
    if ! dvc remote add -d myremote gdrive://$gdrive_id; then
        echo -e "${RED}❌ 添加remote失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 添加remote成功${NC}"
fi

# 添加数据集
echo "📁 正在添加数据集..."
if ! dvc add datasets/raw; then
    echo -e "${RED}❌ 添加数据集失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 添加数据集成功${NC}"

# 提交更改
echo "📝 正在提交更改..."
git add .dvc/config datasets/raw.dvc || {
    echo "错误: Git add 失败"
    exit 1
}

git commit -m "feat: add datasets/raw" || {
    echo "错误: Git commit 失败"
    exit 1
}

# 推送更改
if [ ! -z "$gdrive_id" ]; then
    echo "⬆️ 正在推送到remote..."
    if ! dvc push; then
        echo -e "${RED}❌ 推送失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 推送成功${NC}"
fi

git push || {
    echo "错误: Git push 失败"
    exit 1
}

echo -e "${GREEN}✅ DVC流程完成${NC}" 