#requires -version 5.0

# 检查依赖
$deps = @{
    "git" = "git --version"
    "python" = "python --version"
    "poetry" = "poetry --version"
    "pre-commit" = "pre-commit --version"
}

$failed = $false
foreach ($dep in $deps.GetEnumerator()) {
    try {
        Invoke-Expression $dep.Value | Out-Null
        Write-Host -ForegroundColor Green "✅ $($dep.Key) 已安装"
    } catch {
        Write-Host -ForegroundColor Red "❌ $($dep.Key) 未安装"
        $failed = $true
    }
}

if ($failed) {
    Write-Host -ForegroundColor Red "❌ 依赖检查失败"
    exit 1
}

# 检查Python版本
$python_version = python --version 2>&1
if ($python_version -notmatch "Python 3\.(10|11)") {
    Write-Host -ForegroundColor Red "❌ Python版本必须 >= 3.10"
    exit 1
}

# 安装依赖
try {
    poetry install --with dev
    if ($LASTEXITCODE -ne 0) {
        throw "Poetry安装失败"
    }
    Write-Host -ForegroundColor Green "✅ 依赖安装成功"
} catch {
    Write-Host -ForegroundColor Red "❌ 依赖安装失败: $_"
    exit 1
}

# 安装pre-commit钩子
try {
    poetry run pre-commit install
    if ($LASTEXITCODE -ne 0) {
        throw "Pre-commit安装失败"
    }
    Write-Host -ForegroundColor Green "✅ Pre-commit钩子安装成功"
} catch {
    Write-Host -ForegroundColor Red "❌ Pre-commit钩子安装失败: $_"
    exit 1
}

# 运行测试
try {
    poetry run pytest
    if ($LASTEXITCODE -ne 0) {
        throw "测试失败"
    }
    Write-Host -ForegroundColor Green "✅ 测试通过"
} catch {
    Write-Host -ForegroundColor Red "❌ 测试失败: $_"
    exit 1
}

Write-Host -ForegroundColor Green "✅ 环境设置完成" 