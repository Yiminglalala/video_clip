# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260423-001] best_practice

**Logged**: 2026-04-23T13:30:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
`D:\video_clip` 的本地 Streamlit 服务必须按统一手册启动和验证，不能只看日志里的 URL。

### Details
在 Windows 环境下，`D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -m streamlit run ...` 存在不稳定现象：日志可能打印 `Local URL: http://localhost:8501`，但 `8501` 并未真正监听；同时还观察到 `venv_gpu` 父进程派生系统 Python 子进程监听端口的异常链路。单纯看到日志并不能证明服务可用。

### Suggested Action
统一使用 `docs/SERVICE_RUNBOOK.md` 中的流程：
- 先清理旧进程
- 再用 `venv_gpu` 命令启动
- 必须用 `Invoke-WebRequest http://localhost:8501` 验证 `200`
- 必要时再检查端口监听和进程归属

### Metadata
- Source: error
- Related Files: docs/SERVICE_RUNBOOK.md
- Tags: streamlit, service, startup, restart, windows

---
