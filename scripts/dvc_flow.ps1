#requires -version 5.0

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 设置颜色
$ErrorActionPreference = "Stop"

# 检查 DVC 是否安装
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Error "❌ DVC 未安装，请运行: pip install dvc[gdrive]"
    exit 1
}

# 检查数据集目录
if (-not (Test-Path datasets/raw)) {
    Write-Error "❌ 数据集目录 datasets/raw 不存在"
    exit 1
}

# 获取远程存储 ID
$remoteId = Read-Host "请输入 Google Drive remote ID (例如: gdrive://1xxx，直接回车跳过)"
if (-not $remoteId) {
    Write-Host "跳过远程存储配置" -ForegroundColor Yellow
}

# 检查是否已初始化
if (-not (Test-Path .dvc)) {
    Write-Host "初始化 DVC..." -ForegroundColor Yellow
    dvc init
    if ($LASTEXITCODE -ne 0) { throw "DVC 初始化失败" }
}

# 添加远程存储（如果提供了 ID）
if ($remoteId) {
    dvc remote add -d storage $remoteId
    if ($LASTEXITCODE -ne 0) { throw "添加远程存储失败" }
}

# 添加数据集
Write-Host "添加数据集..." -ForegroundColor Yellow
dvc add datasets/raw
if ($LASTEXITCODE -ne 0) { throw "添加数据集失败" }

# Git 提交
Write-Host "提交更改..." -ForegroundColor Yellow
git add .dvc/config datasets/raw.dvc
git commit -m "add raw data"
if ($LASTEXITCODE -ne 0) { throw "Git 提交失败" }

# DVC 推送（如果配置了远程存储）
if ($remoteId) {
    Write-Host "推送数据..." -ForegroundColor Yellow
    dvc push
    if ($LASTEXITCODE -ne 0) { throw "DVC 推送失败" }
}

Write-Host "✅ DVC 工作流完成" -ForegroundColor Green 