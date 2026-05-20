# 本地运行状态与缓存规范

## 目标

避免把本机运行状态、缓存文件、个人路径或 UI 上次选择提交到 GitHub。

## 不进入版本库的文件

- `slice_config.json`：Streamlit UI 上一次选择的切片参数、歌手名、演唱会名、分辨率选择等本机状态。
- `.features_msaf_tmp.json`：音频结构分析产生的临时特征缓存，可能包含本机临时路径且体积较大。
- `local_config.json`：本机私有配置，可能包含 API 凭证。
- `G:\video_output`：切片页最终导出视频。
- `output/subtitles/`：字幕页最终成片。
- `output/cache/`：ASR、歌词、字幕等可再生成缓存。
- `output/qa/`：截图、抽帧、历史测试产物和人工验证材料。
- `output/` 整体不进入版本库。
- `logs/`：本地 Streamlit 服务日志。
- `backups/`：本地备份包和历史代码副本。

## 配置原则

- 团队默认值写入代码默认配置或示例配置文件。
- 用户私有状态只保留在本机。
- 生成缓存可以复用，但不得作为源码提交。
- 如果发现运行态文件已被 Git 跟踪，应使用 `git rm --cached <file>` 从版本库移除，并加入 `.gitignore`。
