@echo off
chcp 65001 >nul
title Streamlit - 演唱会字幕生成器
cd /d D:\video_clip
call D:\video_clip\SongFormer_install\venv_gpu\Scripts\activate.bat
D:\video_clip\SongFormer_install\venv_gpu\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true
pause
