# 服务运行手册

**适用项目**: `D:\video_clip`  
**默认服务地址**: [http://localhost:8501](http://localhost:8501)  
**更新时间**: 2026-04-23

## 1. 标准目标

后续所有本地启动和重启都按这份手册执行，不再临时试命令。

验收标准只有三条：
- `http://localhost:8501` 返回 `200`
- 页面能打开
- 变更后的代码已经生效

不要只看 Streamlit 日志里打印了 URL。  
这个项目当前在 Windows 上存在“日志显示已启动，但端口没有真正监听”的情况。

## 2. 推荐启动命令

优先使用 GPU 环境的 Python：

```powershell
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -m streamlit run D:\video_clip\app.py --server.port 8501 --server.headless true --server.fileWatcherType none
```

说明：
- `venv_gpu` 是项目约定环境
- `8501` 是当前固定本地端口
- `--server.fileWatcherType none` 避免文件监听带来的不稳定行为

## 3. 正确重启步骤

### Step 1. 先清理旧进程

```powershell
$procs = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and (
    $_.CommandLine -like '*streamlit*app.py*8501*' -or
    $_.CommandLine -like '*--server.port 8501*'
  )
}
foreach($p in $procs){
  try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
}
```

### Step 2. 再启动新进程

```powershell
$logDir='D:\video_clip\tmp\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$out=Join-Path $logDir 'streamlit_8501.out.log'
$err=Join-Path $logDir 'streamlit_8501.err.log'

if(Test-Path $out){ Remove-Item $out -Force }
if(Test-Path $err){ Remove-Item $err -Force }

Start-Process -FilePath 'D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe' `
  -ArgumentList @(
    '-m','streamlit','run','D:\video_clip\app.py',
    '--server.port','8501',
    '--server.headless','true',
    '--server.fileWatcherType','none'
  ) `
  -RedirectStandardOutput $out `
  -RedirectStandardError $err `
  -WindowStyle Hidden
```

### Step 3. 必做验证

```powershell
Invoke-WebRequest -UseBasicParsing 'http://localhost:8501'
```

期望结果：
- HTTP `200`

如果不是 `200`，启动不算成功。

## 4. 正确验证方式

### A. 验证页面是否可访问

```powershell
try {
  (Invoke-WebRequest -UseBasicParsing 'http://localhost:8501').StatusCode
} catch {
  'ERR: ' + $_.Exception.Message
}
```

### B. 验证端口是否真的监听

```powershell
Get-NetTCPConnection -LocalPort 8501 -State Listen
```

### C. 验证是谁占用了 8501

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and (
    $_.CommandLine -like '*streamlit*app.py*8501*' -or
    $_.CommandLine -like '*--server.port 8501*'
  )
} | Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

### D. 看启动日志

```powershell
Get-Content -LiteralPath 'D:\video_clip\tmp\logs\streamlit_8501.out.log' -Tail 40
Get-Content -LiteralPath 'D:\video_clip\tmp\logs\streamlit_8501.err.log' -Tail 40
```

## 5. 当前已知异常

### 异常 1：日志显示已启动，但端口没有监听

表现：
- `out.log` 里有 `Local URL: http://localhost:8501`
- 但 `Invoke-WebRequest` 连不上
- `Get-NetTCPConnection` 查不到 `8501`

处理原则：
- 以 HTTP `200` 和端口监听为准
- 不以日志打印 URL 为准

### 异常 2：`venv_gpu` 父进程拉起了系统 Python 子进程

当前观察到的现象：
- 父进程是  
  `D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe`
- 实际监听 `8501` 的子进程可能变成  
  `C:\Users\YIMING\AppData\Local\Programs\Python\Python310\python.exe`

这会带来两个风险：
- 页面虽然能开，但运行环境不稳定
- CUDA / 依赖判断可能偏离预期

当前处理原则：
- 先保证页面可访问
- 后续单独治理启动链路，直到 `8501` 监听者稳定收敛到 `venv_gpu`

## 6. 当前执行口径

以后每次“启动服务”或“重启服务”都按以下口径执行：

1. 清旧进程  
2. 用 `venv_gpu` 命令启动  
3. 用 `Invoke-WebRequest` 验证 `200`  
4. 必要时查端口与进程归属  
5. 没有 `200` 就不能对用户说“服务已启动”

## 7. 后续待办

这份手册先解决“流程统一”和“验证统一”，但还有一个未收口的问题：
- 彻底修复 `venv_gpu` 启动后派生系统 Python 子进程的问题

在这个问题收掉之前，所有服务启动都必须保留“HTTP 200 + 端口监听 + 进程归属检查”。
