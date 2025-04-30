#requires -version 5.0

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 设置颜色
$ErrorActionPreference = "Stop"

# 检查 DVC 是否安装
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Error "❌ DVC 未安装"
    exit 1
}

# 检查是否已初始化
if (-not (Test-Path .dvc)) {
    Write-Host "初始化 DVC..." -ForegroundColor Yellow
    dvc init
    if ($LASTEXITCODE -ne 0) { throw "DVC 初始化失败" }
}

# 获取远程存储 ID
$remoteId = Read-Host "请输入远程存储 ID (例如: gdrive://1xxx)"
if (-not $remoteId) {
    Write-Error "❌ 远程存储 ID 不能为空"
    exit 1
}

# 添加远程存储
dvc remote add -d storage $remoteId
if ($LASTEXITCODE -ne 0) { throw "添加远程存储失败" }

# 添加数据集
Write-Host "添加数据集..." -ForegroundColor Yellow
dvc add datasets/raw
if ($LASTEXITCODE -ne 0) { throw "添加数据集失败" }

# Git 提交
Write-Host "提交更改..." -ForegroundColor Yellow
git add .dvc datasets/raw.dvc
git commit -m "feat: 添加原始数据集"
if ($LASTEXITCODE -ne 0) { throw "Git 提交失败" }

# DVC 推送
Write-Host "推送数据..." -ForegroundColor Yellow
dvc push
if ($LASTEXITCODE -ne 0) { throw "DVC 推送失败" }

Write-Host "✅ DVC 工作流完成" -ForegroundColor Green 