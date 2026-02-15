@echo off
chcp 65001 >nul
echo ============================================
echo   QuantX 盘中实时监控
echo   %date% %time%
echo ============================================

cd /d "F:\project\my finance"

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 确保日志目录存在
if not exist "data\logs" mkdir "data\logs"

:: 运行盘中监控（会自动循环到收盘）
echo [开始] 启动盘中实时监控...
python intraday_monitor.py

echo [完成] 盘中监控结束 %date% %time%
echo [盘中] %date% %time% 监控结束 >> data\scheduler_log.txt
