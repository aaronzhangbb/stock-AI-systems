@echo off
chcp 65001 >nul
echo ========================================
echo   A股量化交易辅助系统 启动中...
echo ========================================
echo.

cd /d "%~dp0"

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 启动 Streamlit
echo 正在启动 Web 界面...
echo 浏览器将自动打开 http://localhost:8501
echo.
echo 按 Ctrl+C 停止服务
echo.
streamlit run app.py --server.port 8501 --server.headless false

pause

