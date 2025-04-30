#!/bin/bash

# 检查依赖
check_dependency() {
    if command -v $1 &> /dev/null; then
        echo -e "\033[32m✅ $1 已安装\033[0m"
        return 0
    else
        echo -e "\033[31m❌ $1 未安装\033[0m"
        return 1
    fi
}

dependencies=("git" "python" "poetry" "pre-commit")
missing=()

for dep in "${dependencies[@]}"; do
    if ! check_dependency $dep; then
        missing+=($dep)
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo -e "\033[31m请先安装缺失的依赖: ${missing[*]}\033[0m"
    exit 1
fi

# 安装项目依赖
if ! poetry install --with dev; then
    echo -e "\033[31m❌ 依赖安装失败\033[0m"
    exit 1
fi
echo -e "\033[32m✅ 依赖安装成功\033[0m"

# 安装 pre-commit 钩子
if ! pre-commit install; then
    echo -e "\033[31m❌ pre-commit 安装失败\033[0m"
    exit 1
fi
echo -e "\033[32m✅ pre-commit 安装成功\033[0m"

# 运行测试
if ! poetry run pytest; then
    echo -e "\033[31m❌ 测试失败\033[0m"
    exit 1
fi
echo -e "\033[32m✅ 测试通过\033[0m"

echo -e "\033[32m✅ 环境设置完成\033[0m" 