# 与远程仓库合并 - 保留本地所有修复
# 用法: .\merge_with_remote.ps1
# 前置: 确保能访问 GitHub（VPN/代理已开启）

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

Write-Host "`n=== 与远程仓库合并（保留本地修复）===" -ForegroundColor Cyan
Write-Host ""

# 1. 暂存本地修改
Write-Host "[1/5] 暂存本地修改..." -ForegroundColor Yellow
Set-Location $projectRoot
git stash push -m "local_fixes_before_merge" -- app.py retrain_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "stash 失败（可能无修改）" -ForegroundColor Gray
}

# 2. 拉取远程
Write-Host "[2/5] 从 GitHub 拉取最新..." -ForegroundColor Yellow
$fetchOk = $false
try {
    git fetch origin
    if ($LASTEXITCODE -eq 0) { $fetchOk = $true }
} catch {
    Write-Host "fetch 失败，请检查网络与代理" -ForegroundColor Red
    Write-Host "  - 若用代理: git config https.proxy http://127.0.0.1:17890" -ForegroundColor Gray
    git stash pop 2>$null
    exit 1
}

git pull origin master
if ($LASTEXITCODE -ne 0) {
    Write-Host "pull 失败" -ForegroundColor Red
    git stash pop 2>$null
    exit 1
}

# 3. 恢复本地修改
Write-Host "[3/5] 恢复本地修改..." -ForegroundColor Yellow
git stash pop
if ($LASTEXITCODE -ne 0) {
    Write-Host "存在冲突，请手动解决后运行: git add . ; git commit -m 'merge'" -ForegroundColor Red
    exit 1
}

# 4. 语法检查
Write-Host "[4/5] 语法检查..." -ForegroundColor Yellow
python -m py_compile app.py retrain_all.py 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  通过" -ForegroundColor Green
} else {
    Write-Host "  失败，请检查 app.py retrain_all.py" -ForegroundColor Red
}

# 5. 提示
Write-Host "[5/5] 完成" -ForegroundColor Green
Write-Host ""
Write-Host "建议提交并推送:" -ForegroundColor Cyan
Write-Host '  git add app.py retrain_all.py'
Write-Host '  git commit -m "fix: Streamlit 兼容 + XGBoost 内存回退"'
Write-Host '  git push origin master'
Write-Host ""
