@echo off
cd /d D:\video_clip

:: 停止已有的 Streamlit 进程
taskkill /f /im streamlit.exe >nul 2>&1
ping 127.0.0.1 -n 2 >nul

:: 后台静默启动（无任何命令框弹出）
wscript //nologo "%~dp0start_silent.vbs"

:: 等待服务就绪
ping 127.0.0.1 -n 6 >nul

:: 自动打开浏览器
start "" "http://localhost:8501"
