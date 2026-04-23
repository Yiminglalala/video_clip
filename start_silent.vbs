Dim ps
Set ps = CreateObject("WScript.Shell")
' 用 venv_gpu 的 Python 启动 streamlit（torch 2.12.0 CUDA128，支持 RTX 5060 Ti）
ps.Run "powershell -WindowStyle Hidden -Command ""Start-Process -FilePath 'D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe' -ArgumentList '-m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false' -WindowStyle Hidden -WorkingDirectory 'D:\video_clip'""", 0, False
Set ps = Nothing
