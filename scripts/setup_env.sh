#!/bin/bash

# ANSI颜色代码
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 检查依赖
check_dep() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 未安装${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 已安装${NC}"
        return 0
    fi
}

# 检查所有依赖
failed=0
for dep in git python poetry pre-commit; do
    check_dep $dep || failed=1
done

if [ $failed -eq 1 ]; then
    echo -e "${RED}❌ 依赖检查失败${NC}"
    exit 1
fi

# 检查Python版本
python_version=$(python3 --version 2>&1)
if ! echo "$python_version" | grep -qE "Python 3\.(10|11)"; then
    echo -e "${RED}❌ Python版本必须 >= 3.10${NC}"
    exit 1
fi

# 安装依赖
echo "📦 正在安装依赖..."
if ! poetry install --with dev; then
    echo -e "${RED}❌ 依赖安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 依赖安装成功${NC}"

# 安装pre-commit钩子
echo "🔧 正在安装pre-commit钩子..."
if ! poetry run pre-commit install; then
    echo -e "${RED}❌ Pre-commit钩子安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Pre-commit钩子安装成功${NC}"

# 运行测试
echo "🧪 正在运行测试..."
if ! poetry run pytest; then
    echo -e "${RED}❌ 测试失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 测试通过${NC}"

echo -e "${GREEN}✅ 环境设置完成${NC}" 