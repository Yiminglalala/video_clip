# 本地代码审查门禁

本项目默认使用本地审查门禁替代第三方 Code Review CLI。目标是把提交前常用检查固定下来，避免遗漏。

## 入口命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\video_clip\tools\review_gate.ps1
```

脚本会自动使用：

```text
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe
```

如果该 Python 不存在，则回退到系统 `python`。

## 默认检查项

1. `git status --short --branch`
2. `python -m compileall -q app.py src tests`
3. `python -m unittest discover -s tests -v`
4. `git diff --check`
5. `git diff --cached --check`
6. diff + 未跟踪文本文件敏感信息扫描

检查结果会写入：

```text
D:\video_clip\logs\review_gate_YYYYMMDD_HHMMSS.json
```

`logs/` 已在 `.gitignore` 中，不会进入 Git。

## 常用参数

检查当前分支相对远端主分支的差异：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\video_clip\tools\review_gate.ps1 -BaseRef origin/main
```

要求工作区必须干净：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\video_clip\tools\review_gate.ps1 -RequireClean
```

只做快速静态门禁，跳过单测：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\video_clip\tools\review_gate.ps1 -SkipUnitTests
```

## 使用约定

- 小改动：提交前至少跑一次默认门禁。
- 中大型改动：开发分支合并前跑默认门禁，并补充实际 Web/视频业务回归。
- 涉及凭证、路径、FFmpeg、缓存、GPU、导出目录的改动：不能跳过敏感信息扫描。
- 新增文件尚未 `git add` 时，脚本也会扫描常见文本扩展名的未跟踪文件；大文件和二进制文件不会扫描。
- 门禁通过不等于业务验收通过；切片边界、字幕显示、导出编码仍需要按具体任务做样本验证。
