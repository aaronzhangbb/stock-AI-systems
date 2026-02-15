# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    QuantX 定时任务管理脚本
.DESCRIPTION
    注册/注销/查询 Windows 计划任务
    - 收盘任务：每工作日16:15执行（数据更新+AI扫描+邮件）
    - 盘中监控：每工作日9:25启动（实时价格+止损止盈+预警邮件）
.PARAMETER Action
    register   - 注册全部定时任务
    unregister - 注销全部定时任务
    status     - 查询任务状态
    run        - 立即执行收盘任务
.PARAMETER Time
    收盘任务执行时间，默认 "16:15"
.EXAMPLE
    .\setup_scheduler.ps1 -Action register
    .\setup_scheduler.ps1 -Action register -Time "16:30"
    .\setup_scheduler.ps1 -Action unregister
    .\setup_scheduler.ps1 -Action status
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("register", "unregister", "status", "run")]
    [string]$Action,
    
    [string]$Time = "16:15"
)

$TaskNameDaily = "QuantX_DailyJob"
$TaskNameIntraday = "QuantX_IntradayMonitor"
$TaskDescription = "QuantX量化交易系统 - 每日收盘后自动更新数据并运行AI策略扫描"
$IntradayDescription = "QuantX量化交易系统 - 盘中实时持仓监控（止损止盈预警）"
$ProjectDir = "F:\project\my finance"
$BatPath = Join-Path $ProjectDir "run_daily.bat"
$IntradayBatPath = Join-Path $ProjectDir "run_intraday.bat"
$LogDir = Join-Path $ProjectDir "data"

# 兼容旧变量名
$TaskName = $TaskNameDaily

# 确保日志目录存在
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Register-Task {
    param([string]$RunTime)
    
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  注册 QuantX 定时任务（收盘 + 盘中监控）" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    
    # 检查BAT脚本是否存在
    if (-not (Test-Path $BatPath)) {
        Write-Host "[错误] 找不到执行脚本: $BatPath" -ForegroundColor Red
        return $false
    }
    if (-not (Test-Path $IntradayBatPath)) {
        Write-Host "[警告] 找不到盘中监控脚本: $IntradayBatPath (将跳过)" -ForegroundColor Yellow
    }
    
    # ====== 1. 注册收盘任务 ======
    $existing = Get-ScheduledTask -TaskName $TaskNameDaily -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  发现旧收盘任务，正在更新..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskNameDaily -Confirm:$false
    }
    
    $hour, $minute = $RunTime.Split(":")
    
    $trigger = New-ScheduledTaskTrigger `
        -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
        -At "${hour}:${minute}"
    
    $action = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument "/c `"$BatPath`"" `
        -WorkingDirectory $ProjectDir
    
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType Interactive
    
    Register-ScheduledTask `
        -TaskName $TaskNameDaily `
        -Description $TaskDescription `
        -Trigger $trigger `
        -Action $action `
        -Settings $settings `
        -Principal $principal `
        -Force
    
    Write-Host ""
    Write-Host "  ✅ 收盘任务注册成功！" -ForegroundColor Green
    Write-Host "  任务名称: $TaskNameDaily" -ForegroundColor White
    Write-Host "  执行时间: 每周一到周五 $RunTime" -ForegroundColor White
    
    # ====== 2. 注册盘中监控任务 ======
    if (Test-Path $IntradayBatPath) {
        $existingIntra = Get-ScheduledTask -TaskName $TaskNameIntraday -ErrorAction SilentlyContinue
        if ($existingIntra) {
            Write-Host "  发现旧盘中监控任务，正在更新..." -ForegroundColor Yellow
            Unregister-ScheduledTask -TaskName $TaskNameIntraday -Confirm:$false
        }
        
        $intraTrigger = New-ScheduledTaskTrigger `
            -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
            -At "09:25"
        
        $intraAction = New-ScheduledTaskAction `
            -Execute "cmd.exe" `
            -Argument "/c `"$IntradayBatPath`"" `
            -WorkingDirectory $ProjectDir
        
        # 盘中监控最长运行6小时（覆盖整个交易日）
        $intraSettings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
            -RestartCount 2 `
            -RestartInterval (New-TimeSpan -Minutes 3)
        
        Register-ScheduledTask `
            -TaskName $TaskNameIntraday `
            -Description $IntradayDescription `
            -Trigger $intraTrigger `
            -Action $intraAction `
            -Settings $intraSettings `
            -Principal $principal `
            -Force
        
        Write-Host "  ✅ 盘中监控任务注册成功！" -ForegroundColor Green
        Write-Host "  任务名称: $TaskNameIntraday" -ForegroundColor White
        Write-Host "  执行时间: 每周一到周五 09:25（运行到收盘自动退出）" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "  提示: 确保电脑在交易时段不要关机/休眠" -ForegroundColor Yellow
    Write-Host "  可在「任务计划程序」中查看和管理这些任务" -ForegroundColor Gray
    
    # 记录注册信息到JSON
    $info = @{
        daily_task = $TaskNameDaily
        daily_time = $RunTime
        intraday_task = $TaskNameIntraday
        intraday_time = "09:25"
        bat_path = $BatPath
        intraday_bat_path = $IntradayBatPath
        registered_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        status = "active"
    } | ConvertTo-Json
    
    $info | Out-File -FilePath (Join-Path $LogDir "scheduler_config.json") -Encoding utf8
    
    return $true
}

function Unregister-Task {
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  注销 QuantX 全部定时任务" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    
    # 注销收盘任务
    $existing = Get-ScheduledTask -TaskName $TaskNameDaily -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskNameDaily -Confirm:$false
        Write-Host "  ✅ 收盘任务已注销" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️ 未找到收盘任务 $TaskNameDaily" -ForegroundColor Yellow
    }
    
    # 注销盘中监控任务
    $existingIntra = Get-ScheduledTask -TaskName $TaskNameIntraday -ErrorAction SilentlyContinue
    if ($existingIntra) {
        Unregister-ScheduledTask -TaskName $TaskNameIntraday -Confirm:$false
        Write-Host "  ✅ 盘中监控任务已注销" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️ 未找到盘中监控任务 $TaskNameIntraday" -ForegroundColor Yellow
    }
    
    # 更新配置
    $configPath = Join-Path $LogDir "scheduler_config.json"
    if (Test-Path $configPath) {
        $cfg = Get-Content $configPath | ConvertFrom-Json
        $cfg.status = "inactive"
        $cfg | ConvertTo-Json | Out-File -FilePath $configPath -Encoding utf8
    }
}

function Get-TaskStatus {
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  QuantX 定时任务状态" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    
    $statusData = @{}
    
    # 查询收盘任务
    Write-Host ""
    Write-Host "  【收盘任务 $TaskNameDaily】" -ForegroundColor White
    $existing = Get-ScheduledTask -TaskName $TaskNameDaily -ErrorAction SilentlyContinue
    if ($existing) {
        $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskNameDaily
        Write-Host "  状态:     $($existing.State)" -ForegroundColor Green
        Write-Host "  上次运行: $($taskInfo.LastRunTime)" -ForegroundColor White
        Write-Host "  上次结果: $($taskInfo.LastTaskResult)" -ForegroundColor White
        Write-Host "  下次运行: $($taskInfo.NextRunTime)" -ForegroundColor White
        $statusData["daily"] = @{
            exists = $true; state = $existing.State.ToString()
            last_run = $taskInfo.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss")
            last_result = $taskInfo.LastTaskResult
            next_run = $taskInfo.NextRunTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
    } else {
        Write-Host "  ❌ 未注册" -ForegroundColor Red
        $statusData["daily"] = @{ exists = $false; state = "NotRegistered" }
    }
    
    # 查询盘中监控任务
    Write-Host ""
    Write-Host "  【盘中监控 $TaskNameIntraday】" -ForegroundColor White
    $existingIntra = Get-ScheduledTask -TaskName $TaskNameIntraday -ErrorAction SilentlyContinue
    if ($existingIntra) {
        $intraInfo = Get-ScheduledTaskInfo -TaskName $TaskNameIntraday
        Write-Host "  状态:     $($existingIntra.State)" -ForegroundColor Green
        Write-Host "  上次运行: $($intraInfo.LastRunTime)" -ForegroundColor White
        Write-Host "  上次结果: $($intraInfo.LastTaskResult)" -ForegroundColor White
        Write-Host "  下次运行: $($intraInfo.NextRunTime)" -ForegroundColor White
        $statusData["intraday"] = @{
            exists = $true; state = $existingIntra.State.ToString()
            last_run = $intraInfo.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss")
            last_result = $intraInfo.LastTaskResult
            next_run = $intraInfo.NextRunTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
    } else {
        Write-Host "  ❌ 未注册" -ForegroundColor Red
        $statusData["intraday"] = @{ exists = $false; state = "NotRegistered" }
    }
    
    $statusData | ConvertTo-Json -Depth 3 | Out-File -FilePath (Join-Path $LogDir "scheduler_status.json") -Encoding utf8
}

function Run-TaskNow {
    Write-Host "  手动触发运行..." -ForegroundColor Yellow
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "  ✅ 已触发执行" -ForegroundColor Green
    } else {
        Write-Host "  直接运行BAT脚本..." -ForegroundColor Yellow
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$BatPath`"" -WorkingDirectory $ProjectDir
    }
}

# 执行
switch ($Action) {
    "register"   { Register-Task -RunTime $Time }
    "unregister" { Unregister-Task }
    "status"     { Get-TaskStatus }
    "run"        { Run-TaskNow }
}
