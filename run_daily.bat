@echo off
chcp 65001 >nul
echo ============================================
echo   QuantX 每日收盘任务
echo   %date% %time%
echo ============================================

cd /d "F:\project\my finance"

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 确保日志目录存在
if not exist "data\logs" mkdir "data\logs"

:: 运行每日任务
echo [开始] 运行每日收盘任务...
python daily_job.py 2>&1 >> "data\logs\daily_%date:~0,4%%date:~5,2%%date:~8,2%.log"

:: 记录运行日志
echo [完成] %date% %time% >> data\scheduler_log.txt

:: 如果出错，记录
if %errorlevel% neq 0 (
    echo [错误] 任务执行失败，错误码: %errorlevel%
    echo [错误] %date% %time% 错误码:%errorlevel% >> data\scheduler_log.txt
)
