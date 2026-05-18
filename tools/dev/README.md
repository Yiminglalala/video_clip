# 开发脚本归档说明

根目录历史 `debug_*.py`、`diagnostic_*.py`、`run_*.py` 和临时 `test_*.py` 脚本已迁移到 `tools/dev/legacy_root_scripts/`。

这些脚本只作为历史排查入口保留，不参与发布门禁。稳定回归测试以 `tests/` 目录为准。

后续清理规则：

- 稳定回归测试迁移到 `tests/`，优先使用 `unittest`。
- 临时排查脚本迁移到 `tools/dev/`，文件名保留原始用途说明。
- 无法复现价值或依赖已废弃链路的脚本，先登记到 `docs/LEGACY_MODULES.md`，再删除。
- 发布门禁只认 `tests/` 下的稳定测试，不认根目录临时脚本。
