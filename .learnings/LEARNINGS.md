# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260515-001] best_practice

**Logged**: 2026-05-15T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: process

### Summary
业务例外必须显式留痕，避免后续代码审查把有意取舍误判为缺陷。

### Details
例如切片链路里，`min_duration` 是目标约束；当歌词完整性与最短时长冲突时，可以保留被歌词保护的短片段。这类行为必须在代码注释、测试断言、日志/元数据和治理文档中说明原因。

### Suggested Action
后续新增或调整类似例外时，同步更新 `docs/governance/BUSINESS_EXCEPTIONS.md` 或对应 TECH_SPEC，并在测试里覆盖该例外场景。

### Metadata
- Source: user_feedback
- Related Files: docs/governance/BUSINESS_EXCEPTIONS.md, src/segment_postprocess.py, tests/test_segment_postprocess.py
- Tags: review, business-exception, documentation

---

## [LRN-20260509-001] correction

**Logged**: 2026-05-09T10:30:00+08:00
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
切片边界排查应优先使用当前流程已有的豆包 ASR 缓存，不应优先重新调用豆包 API。

### Details
用户指出本轮问题已有豆包返回的歌词/句子数据。正确流程是先读取 `LiveVideoProcessor._cached_asr_results` 或落盘缓存，用同一份 ASR 时间轴复核切点；只有缓存不存在或需要复现时才重新跑 ASR。

### Suggested Action
将 ASR 缓存持久化到输出目录，并让后处理、调试和回归测试优先消费该缓存，避免额外 API 成本和诊断数据不一致。

### Metadata
- Source: user_feedback
- Related Files: src/processor.py, src/segment_postprocess.py
- Tags: doubao, asr-cache, boundary-debug

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
