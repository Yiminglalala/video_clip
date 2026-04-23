@echo off
echo ========================================
echo   Live Concert Cutter - 内网共享模式
echo ========================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 需要管理员权限来配置防火墙
    echo [*] 正在请求管理员权限...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: 添加防火墙规则
echo [1/3] 配置防火墙...
netsh advfirewall firewall show rule name="Streamlit-8501" >nul 2>&1
if %errorlevel% neq 0 (
    netsh advfirewall firewall add rule name="Streamlit-8501" dir=in action=allow protocol=tcp localport=8501 >nul
    echo     [OK] 已开放端口 8501
) else (
    echo     [SKIP] 端口 8501 已开放
)

:: 获取本机 IP
echo.
echo [2/3] 获取本机 IP...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr "IPv4"') do (
    set IP=%%a
    goto :gotip
)
:gotip
set IP=%IP: =%
echo     本机 IP: %IP%

:: 启动 Streamlit
echo.
echo [3/3] 启动 Streamlit 服务...
echo.
echo ========================================
echo   同事访问地址: http://%IP%:8501
echo ========================================
echo.
echo [按 Ctrl+C 停止服务]
echo.

cd /d D:\video_clip
call SongFormer_install\venv_gpu\Scripts\activate.bat
streamlit run app.py
