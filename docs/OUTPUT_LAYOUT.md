# 项目文件与输出目录规范

## 目标

让项目根目录只保留入口、源码、正式文档和必要配置；运行产物按用途进入固定目录，避免测试视频、截图、缓存和旧脚本混在一起。

## 标准目录

- `src/`：正式业务代码。
- `tests/`：稳定回归测试，发布前默认执行这里的测试。
- `docs/`：正式文档。
- `docs/archive/`：历史方案、阶段文档和不再作为当前入口的资料。
- `tools/dev/`：开发辅助脚本。
- `tools/dev/legacy_root_scripts/`：从根目录归档的历史调试脚本和临时验证脚本。
- `G:\video_output`：切片页最终导出视频。
- `output/subtitles/`：字幕页最终成片。
- `output/cache/`：歌词、字幕、ASR 等可再生成缓存。
- `output/qa/`：人工验证、Playwright 截图、抽帧检查和历史测试产物。
- `temp/`：运行时上传、预览和临时处理中间文件。
- `logs/`：本地服务日志。
- `backups/`：本地备份，不提交 GitHub。

## 输出规则

- 新的切片导出默认写入 `G:\video_output`。
- 新的字幕生成默认写入 `output/subtitles/`。
- 新的字幕可见性抽帧默认写入 `output/qa/subtitle_probe/`。
- 新的浏览器测试截图默认写入 `output/qa/playwright/`。
- 新的缓存默认写入 `output/cache/` 的子目录。

如需临时覆盖切片导出目录，可设置环境变量 `VIDEO_CLIP_VIDEO_OUTPUT_DIR`；未设置时默认使用 `G:\video_output`。

## 不再推荐

- 不在根目录放新的 `test_*.py`、`debug_*.py`、`diagnostic_*.py`。
- 不在 `output/` 顶层直接放新的测试视频、截图或日志。
- 不提交 `output/`、`temp/`、`logs/`、`backups/`、`local_config.json`、`slice_config.json`。
