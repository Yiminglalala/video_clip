# Legacy 模块清单

更新时间：2026-05-06

## 目标

明确哪些代码不是当前主路径，避免后续开发继续在旧入口上叠功能。

## 当前主路径

| 功能 | 主入口 |
|---|---|
| Web 应用 | `D:\video_clip\app.py` |
| 视频切片 | `app.py -> render_slicing_mode() -> src.processor.LiveVideoProcessor` |
| 字幕生成 | `app.py -> render_subtitle_mode() -> src.doubao_api.DoubaoASR` |
| 输出规格 | `src.output_spec` |

## Legacy / 非主路径模块

| 模块 | 当前状态 | 处理策略 |
|---|---|---|
| `src/ui.py` | 旧版独立 Streamlit 原型 | 已标记 legacy；不要继续新增正式功能 |
| `src/preview_editor.py` | 旧编辑器服务层 | 保留，等待标签库接入时评估复用 |
| `src/auto_optimizer.py` | 早期自动优化雏形 | 保留，后续标签库校准时升级 |
| `src/sample_library.py` | 样本库基础 SQLite 实现 | 保留，后续接入当前 Step 3 |
| `src/lyric_subtitle.py` 中 ACR/Shazam 识曲 | 旧识曲/歌词链路 | 默认禁用，需显式环境变量才可启用 |

## 识曲开关

旧听歌识曲现在默认关闭。若确实需要临时测试 legacy 识曲，可设置：

```powershell
$env:ENABLE_LEGACY_SONG_IDENTIFY = "1"
```

默认不启用的原因：

- 避免 ACRCloud/Shazam 网络调用卡住主流程。
- 避免产生额外成本。
- 当前产品策略已收敛为“稳定切片 + 手动歌名/标签编辑 + 标签库校准”。

## 后续清理原则

1. 不直接删除 legacy 模块，除非确认没有导入依赖。
2. 新功能必须接入 `app.py` 当前双 Tab 主流程。
3. 如果 legacy 模块要复用，先迁移成服务函数，再接入主流程。
4. 删除前必须有 Git 备份和最小回归测试。
