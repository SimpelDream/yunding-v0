#requires -version 5.0

try {
    # 检查 Python 版本
    $pythonVersion = python --version
    if (-not $?) {
        Write-Host "❌ 未找到 Python" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ 已安装 $pythonVersion" -ForegroundColor Green

    # 检查 Poetry
    $poetryVersion = poetry --version
    if (-not $?) {
        Write-Host "❌ 未找到 Poetry" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ 已安装 $poetryVersion" -ForegroundColor Green

    # 安装依赖
    Write-Host "📦 正在安装依赖..." -ForegroundColor Yellow
    poetry install --with dev

    # 检查环境
    Write-Host "🔍 正在检查环境..." -ForegroundColor Yellow
    python scripts/setup_env_check.py

    Write-Host "✅ 环境配置完成" -ForegroundColor Green
}
catch {
    Write-Host "❌ 发生错误：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 