#!/bin/bash

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 DVC 是否安装
if ! command -v dvc &> /dev/null; then
    echo -e "${RED}❌ DVC 未安装，请运行: pip install dvc[gdrive]${NC}"
    exit 1
fi

# 检查数据集目录
if [ ! -d "datasets/raw" ]; then
    echo -e "${RED}❌ 数据集目录 datasets/raw 不存在${NC}"
    exit 1
fi

# 获取远程存储 ID
read -p "请输入 Google Drive remote ID (例如: gdrive://1xxx，直接回车跳过): " remoteId
if [ -z "$remoteId" ]; then
    echo -e "${YELLOW}跳过远程存储配置${NC}"
fi

# 检查是否已初始化
if [ ! -d .dvc ]; then
    echo -e "${YELLOW}初始化 DVC...${NC}"
    dvc init
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ DVC 初始化失败${NC}"
        exit 1
    fi
fi

# 添加远程存储（如果提供了 ID）
if [ -n "$remoteId" ]; then
    dvc remote add -d storage "$remoteId"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 添加远程存储失败${NC}"
        exit 1
    fi
fi

# 添加数据集
echo -e "${YELLOW}添加数据集...${NC}"
dvc add datasets/raw
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 添加数据集失败${NC}"
    exit 1
fi

# Git 提交
echo -e "${YELLOW}提交更改...${NC}"
git add .dvc/config datasets/raw.dvc
git commit -m "add raw data"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Git 提交失败${NC}"
    exit 1
fi

# DVC 推送（如果配置了远程存储）
if [ -n "$remoteId" ]; then
    echo -e "${YELLOW}推送数据...${NC}"
    dvc push
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ DVC 推送失败${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ DVC 工作流完成${NC}" 