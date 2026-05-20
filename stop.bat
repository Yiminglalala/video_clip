@echo off
chcp 65001 >nul
title 演唱会视频智能切片 - 停止服务

cd /d D:\video_clip

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\stop_service.ps1" -Port 8501

echo.
pause
