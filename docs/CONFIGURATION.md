# 本地配置说明

更新时间：2026-05-06

## 目标

统一运行时配置来源，避免把 API 凭证、个人路径、临时测试参数写死在代码里。

## 配置优先级

运行时配置按以下顺序读取：

1. 环境变量
2. 项目根目录 `local_config.json`
3. 代码默认值

当前已接入该策略的配置：

| 配置 | 环境变量 | `local_config.json` 字段 | 说明 |
|---|---|---|---|
| 豆包 AppID | `DOUBAO_APPID` | `doubao.appid` 或 `doubao_appid` | 字幕识别必需 |
| 豆包 Access Token | `DOUBAO_ACCESS_TOKEN` | `doubao.access_token` 或 `doubao_access_token` | 字幕识别必需 |
| 私有配置路径 | `VIDEO_CLIP_LOCAL_CONFIG` | 不适用 | 覆盖默认 `D:\video_clip\local_config.json` |

## 推荐配置方式

复制示例文件：

```powershell
Copy-Item D:\video_clip\local_config.example.json D:\video_clip\local_config.json
```

填写：

```json
{
  "doubao": {
    "appid": "你的 AppID",
    "access_token": "你的 Access Token"
  }
}
```

`local_config.json` 已加入 `.gitignore`，不会提交到 GitHub。

## 切片配置

`slice_config.json` 保存 UI 上一次选择的切片参数，属于用户状态配置，不适合存放密钥。

当前字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `min_dur` | number | 最短切片时长 |
| `max_dur` | number | 最长切片时长 |
| `cut_mode` | string | UI 兼容参数，当前导出统一重编码 |
| `enable_subtitle` | boolean | 切片页是否生成字幕 |
| `singer_name` | string | 歌手名，可辅助歌词匹配 |
| `concert_name` | string | 演唱会名 |
| `landscape_resolution_choice` | string | 横屏输入输出分辨率选择 |

## 禁止事项

- 不要在 `app.py`、`processor.py` 或测试脚本中写入真实 API 凭证。
- 不要把 `local_config.json`、`.env`、个人测试视频路径作为通用配置提交。
- 不要把 `slice_config.json` 当作密钥配置文件使用。
