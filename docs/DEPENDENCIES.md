# 依赖与运行环境说明

更新时间：2026-05-06

## 环境分层

当前项目依赖分为三层：

| 层级 | 用途 | 说明 |
|---|---|---|
| 最小 Web 依赖 | 打开 Streamlit 页面、字幕页基础流程 | 对应 `requirements.txt` |
| 完整切片依赖 | SongFormer、Demucs、MSAF、GPU 推理 | 当前以 `SongFormer_install\venv_gpu` 为准 |
| 系统依赖 | FFmpeg、CUDA、NVIDIA 驱动 | 不由 pip 管理 |

## 当前推荐 Python 环境

```powershell
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe
```

启动服务：

```powershell
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -m streamlit run D:\video_clip\app.py --server.port 8501 --server.headless true --server.fileWatcherType none
```

## 最小依赖

`requirements.txt` 当前只覆盖基础 Web/字幕能力：

```text
streamlit
numpy
scipy
librosa
soundfile
requests
```

这不代表完整切片环境。完整切片还需要 PyTorch、SongFormer 相关包、Demucs、MSAF 等依赖。

## SongFormer 依赖预检

切片页在开始分析前会调用：

```python
SongFormerAnalyzer.check_runtime_dependencies()
```

如果缺依赖，页面会提示需要安装的 pip 包。当前策略是：

- `strict_songformer=True`
- SongFormer 不可用时直接阻断切片
- 不再静默降级为手工分类

## GPU 依赖

GPU 相关能力：

| 能力 | GPU 路径 | CPU fallback |
|---|---|---|
| SongFormer | PyTorch CUDA | 严格模式下不可用会阻断 |
| Demucs | PyTorch CUDA | 可 CPU，但速度明显下降 |
| 视频编码 | FFmpeg `h264_nvenc` | `libx264` |

检查 CUDA：

```powershell
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

检查 FFmpeg：

```powershell
ffmpeg -version
ffmpeg -encoders | Select-String h264_nvenc
```

## 待治理项

- 将完整 GPU 依赖冻结成单独文件，例如 `requirements-gpu.txt`。
- 将 SongFormer 外部仓库依赖和模型下载步骤写成可重复安装脚本。
- 将 FFmpeg/CUDA/驱动版本纳入 `run_summary.json`。
