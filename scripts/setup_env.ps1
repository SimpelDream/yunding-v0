# 检查依赖
$dependencies = @{
    "git" = "git --version"
    "python" = "python --version"
    "poetry" = "poetry --version"
    "pre-commit" = "pre-commit --version"
}

$missing = @()
foreach ($dep in $dependencies.Keys) {
    try {
        Invoke-Expression $dependencies[$dep] | Out-Null
        Write-Host -ForegroundColor Green "✅ $dep 已安装"
    } catch {
        $missing += $dep
        Write-Host -ForegroundColor Red "❌ $dep 未安装"
    }
}

if ($missing.Count -gt 0) {
    Write-Host -ForegroundColor Red "请先安装缺失的依赖: $($missing -join ', ')"
    exit 1
}

# 安装项目依赖
try {
    poetry install --with dev
    if ($LASTEXITCODE -ne 0) { throw "poetry install 失败" }
    Write-Host -ForegroundColor Green "✅ 依赖安装成功"
} catch {
    Write-Host -ForegroundColor Red "❌ 依赖安装失败: $_"
    exit 1
}

# 安装 pre-commit 钩子
try {
    pre-commit install
    if ($LASTEXITCODE -ne 0) { throw "pre-commit install 失败" }
    Write-Host -ForegroundColor Green "✅ pre-commit 安装成功"
} catch {
    Write-Host -ForegroundColor Red "❌ pre-commit 安装失败: $_"
    exit 1
}

# 运行测试
try {
    poetry run pytest
    if ($LASTEXITCODE -ne 0) { throw "测试失败" }
    Write-Host -ForegroundColor Green "✅ 测试通过"
} catch {
    Write-Host -ForegroundColor Red "❌ 测试失败: $_"
    exit 1
}

Write-Host -ForegroundColor Green "✅ 环境设置完成" 