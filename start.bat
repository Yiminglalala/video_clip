@echo off
chcp 65001 >nul
title 演唱会视频智能切片 - 启动服务

cd /d D:\video_clip

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_service.ps1" -Port 8501 -OpenBrowser

if errorlevel 1 (
    echo.
    echo 启动失败，请查看 D:\video_clip\logs\streamlit_8501.err.log
    pause
)
