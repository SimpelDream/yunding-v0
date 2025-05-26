#requires -version 5.0

# DVC 工作流脚本
# 执行 DVC 初始化、添加数据集和推送操作

try {
    # 检查 DVC
    $dvcVersion = dvc --version
    if (-not $?) {
        Write-Host "❌ 未找到 DVC，请先安装: pip install dvc" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ 已安装 $dvcVersion" -ForegroundColor Green

    # 检查数据集目录
    if (-not (Test-Path "datasets/raw")) {
        Write-Host "❌ 未找到 datasets/raw 目录" -ForegroundColor Red
        exit 1
    }

    # 询问 Google Drive ID
    $driveId = Read-Host "请输入 Google Drive remote ID (可选，直接回车跳过)"
    
    # 初始化 DVC
    Write-Host "📦 正在初始化 DVC..." -ForegroundColor Yellow
    dvc init

    # 添加 remote（如果提供了 ID）
    if ($driveId) {
        dvc remote add -d myremote gdrive://$driveId
        Write-Host "✅ 已添加 Google Drive remote" -ForegroundColor Green
    }

    # 添加数据集
    Write-Host "📦 正在添加数据集..." -ForegroundColor Yellow
    dvc add datasets/raw

    # 提交到 Git
    Write-Host "📝 正在提交到 Git..." -ForegroundColor Yellow
    git add .dvc/config datasets/raw.dvc
    git commit -m "add raw data"

    # 推送到 remote（如果设置了）
    if ($driveId) {
        Write-Host "📤 正在推送到 remote..." -ForegroundColor Yellow
        dvc push
    }

    Write-Host "✅ DVC 流程完成" -ForegroundColor Green
}
catch {
    Write-Host "❌ 发生错误：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 