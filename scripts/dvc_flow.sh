#!/bin/bash

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 设置颜色
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

# 检查数据集目录
if [ ! -d "datasets/raw" ]; then
    echo -e "\033[31m❌ 未找到 datasets/raw 目录\033[0m"
    exit 1
fi

# 询问 Google Drive ID
read -p "请输入 Google Drive remote ID (可选，直接回车跳过): " drive_id

# 初始化 DVC
echo -e "\033[33m📦 正在初始化 DVC...\033[0m"
dvc init

# 添加 remote（如果提供了 ID）
if [ ! -z "$drive_id" ]; then
    dvc remote add -d myremote "gdrive://$drive_id"
    echo -e "\033[32m✅ 已添加 Google Drive remote\033[0m"
fi

# 添加数据集
echo -e "\033[33m📦 正在添加数据集...\033[0m"
dvc add datasets/raw

# 提交到 Git
echo -e "\033[33m📝 正在提交到 Git...\033[0m"
git add .dvc/config datasets/raw.dvc
git commit -m "add raw data"

# 推送到 remote（如果设置了）
if [ ! -z "$drive_id" ]; then
    echo -e "\033[33m📤 正在推送到 remote...\033[0m"
    dvc push
fi

echo -e "\033[32m✅ DVC 流程完成\033[0m" 