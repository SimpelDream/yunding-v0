#requires -version 5.0

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

# 检查 DVC 是否安装
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Error "DVC 未安装，请先安装 DVC"
    exit 1
}

# 检查 Git 是否安装
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git 未安装，请先安装 Git"
    exit 1
}

# 检查数据集目录
if (-not (Test-Path "datasets/raw")) {
    Write-Host -ForegroundColor Red "❌ datasets/raw 目录不存在，请先录制数据"
    exit 1
}

# 询问Google Drive remote ID
$gdrive_id = Read-Host "请输入Google Drive remote ID (可选)"

# 初始化DVC
if (-not (Test-Path ".dvc")) {
    Write-Host "📦 正在初始化DVC..."
    dvc init
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Red "❌ DVC初始化失败"
        exit 1
    }
    Write-Host -ForegroundColor Green "✅ DVC初始化成功"
}

# 添加Google Drive remote
if ($gdrive_id) {
    Write-Host "🔗 正在添加Google Drive remote..."
    dvc remote add -d myremote gdrive://$gdrive_id
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Red "❌ 添加remote失败"
        exit 1
    }
    Write-Host -ForegroundColor Green "✅ 添加remote成功"
}

# 添加数据集
Write-Host "📁 正在添加数据集..."
dvc add datasets/raw
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "❌ 添加数据集失败"
    exit 1
}
Write-Host -ForegroundColor Green "✅ 添加数据集成功"

# Git操作
Write-Host "📝 正在提交更改..."
git add datasets/raw.dvc .dvc/config
git commit -m "feat: add datasets/raw"
if ($LASTEXITCODE -ne 0) {
    Write-Host -ForegroundColor Red "❌ Git提交失败"
    exit 1
}
Write-Host -ForegroundColor Green "✅ Git提交成功"

# 推送到remote
if ($gdrive_id) {
    Write-Host "⬆️ 正在推送到remote..."
    dvc push
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Red "❌ 推送失败"
        exit 1
    }
    Write-Host -ForegroundColor Green "✅ 推送成功"
}

Write-Host -ForegroundColor Green "✅ DVC流程完成" 