# 运行测试脚本
param(
    [switch]$Coverage,
    [switch]$Verbose
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# 检查 poetry 是否安装
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Error "❌ poetry 未安装"
    exit 1
}

# 构建测试命令
$testCommand = "poetry run pytest"
if ($Coverage) {
    $testCommand += " --cov=tft_assist --cov-report=html"
}
if ($Verbose) {
    $testCommand += " -v"
}

# 运行测试
try {
    Invoke-Expression $testCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ 测试失败"
        exit 1
    }
    Write-Host "✅ 测试通过" -ForegroundColor Green
} catch {
    Write-Error "❌ 测试执行出错: $_"
    exit 1
} 