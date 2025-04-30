#!/bin/bash

# 运行测试脚本
# 用法: ./run_tests.sh [--coverage] [--verbose]

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 解析参数
COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查 poetry 是否安装
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ poetry 未安装${NC}"
    exit 1
fi

# 构建测试命令
TEST_CMD="poetry run pytest"
if [ "$COVERAGE" = true ]; then
    TEST_CMD+=" --cov=tft_assist --cov-report=html"
fi
if [ "$VERBOSE" = true ]; then
    TEST_CMD+=" -v"
fi

# 运行测试
if $TEST_CMD; then
    echo -e "${GREEN}✅ 测试通过${NC}"
else
    echo -e "${RED}❌ 测试失败${NC}"
    exit 1
fi 