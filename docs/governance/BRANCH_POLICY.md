# BRANCH_POLICY（分支开发与合并规范）

**文档版本**: v1.0
**创建时间**: 2026-05-15
**最后更新**: 2026-05-15
**负责人**: AI
**变更日志**:
- 2026-05-15: 初始创建，规范新分支、worktree、测试、合并、删除分支流程。

---

## 1. 目标

- 保证 `main` 始终尽量可运行，避免开发中的实验影响正在使用的 8501 服务。
- 明确什么时候直接在当前分支做，什么时候必须新建分支或独立 worktree。
- 明确合并、推送、删除分支的门禁，避免丢失未提交代码。
- 让每次改动都能追踪：需求、技术方案、测试结论、回滚方式。

---

## 2. 分支类型

| 类型 | 命名 | 用途 | 生命周期 |
|---|---|---|---|
| 主分支 | `main` | 当前稳定开发基线，默认运行 8501 | 长期保留 |
| 功能分支 | `codex/feature-name` | 新功能、结构调整、UI 改造 | 合并后删除 |
| 修复分支 | `codex/fix-name` | 明确 bug 修复 | 合并后删除 |
| 实验分支 | `codex/experiment-name` | 算法、模型、阈值、样本库等不确定方案 | 验证后合并或废弃 |
| 长期隔离分支 | `codex/topic-name` + worktree | 不希望影响主线的功能，例如样本库校准 | 完成前保留 |

命名规则：
- 使用 `codex/` 前缀。
- 使用英文小写、数字和短横线。
- 名称要表达目标，例如 `codex/word-boundary-ownership`、`codex/sample-library-calibration`。

---

## 3. 什么时候必须新建分支

出现以下任一情况，必须新建分支：

- 改动会影响切片主流程、字幕烧录、导出编码、分辨率、GPU 调度等核心链路。
- 需要跑较长时间测试，或需要多轮试验阈值、算法、模型。
- 用户正在使用 `main` 上的 8501 服务，不能被开发过程打断。
- 改动可能引入数据库、样本库、配置迁移、依赖安装、文件结构调整。
- 需要并行推进两个方向，例如主线 bug 修复和样本库功能开发。
- 不确定是否会合并，例如实验性优化、候选方案 A/B 测试。

推荐使用独立 worktree 的情况：

- `main` 需要继续开着服务给用户测试。
- 分支需要启动自己的 Streamlit 服务。
- 分支会产生大量临时输出、缓存或测试数据。

示例：

```powershell
git fetch origin
git worktree add D:\video_clip_worktrees\word-boundary -b codex/word-boundary origin/main
```

---

## 4. 什么时候可以不新建分支

以下低风险改动可以直接在当前 `main` 上做，但仍要测试：

- 文档修正、注释修正、README 或治理文档更新。
- 明确的一行或少量配置修复，且用户当前没有依赖正在运行的服务。
- 已定位根因的小 bug 修复，改动范围小，能立即跑完相关测试。
- 临时本地排查，不会提交，也不会影响服务。

限制：
- 如果 `main` 已经有大量未提交改动，先整理改动来源，再决定是否继续直接开发。
- 如果涉及真实输出效果，至少要跑相关单元测试或一次手工 smoke。

---

## 5. 开发前检查

开始任何分支工作前，先执行：

```powershell
git status --short --branch
git fetch --all --prune
git branch -vv
git worktree list
```

检查项：
- 当前分支是否正确。
- 是否有未提交改动；如果有，确认属于谁、属于哪个任务。
- 远程 `origin/main` 是否已同步。
- 是否已有可复用的任务分支或 worktree。
- 是否有运行中的 Streamlit 服务需要保留。

不允许：
- 在未确认未提交改动来源时切分支、合并或删除分支。
- 用 `git reset --hard`、`git checkout -- .` 处理未知改动。
- 把 `output/`、`backups/`、`.features_msaf_tmp.json`、真实凭证、个人测试视频提交到 GitHub。

---

## 6. 开发中的隔离规则

主线服务规则：
- `main` 默认使用 `http://localhost:8501`。
- 如果用户正在测试 8501，不要重启主线服务，除非用户明确要求。
- 分支 worktree 测试应使用其他端口，例如 `8502`、`8506`。

代码隔离规则：
- 功能分支只改本任务需要的文件。
- 不在同一分支混入无关功能、临时清理、格式化大改。
- 长时间实验产生的脚本、输出、缓存必须进入 `tools/dev/`、`output/` 或被 `.gitignore` 忽略。

测试记录规则：
- 每次关键测试记录命令、输入视频、结论。
- 如果是完整流程测试，产出或记录 `run_summary`。
- 如果失败，记录确定根因和修复方案，不只记录现象。

---

## 7. 合并前门禁

合并到 `main` 前必须满足：

- `git status --short` 清晰：只包含本任务应合并文件。
- 分支相对 `main` 的 diff 已审查。
- 相关单元测试通过。
- 至少一次 smoke 测试通过；核心链路改动需要端到端测试。
- `python -m compileall -q app.py src tests` 通过。
- `git diff --check` 对本任务文件通过。
- 不包含真实 API Token、个人隐私路径、临时视频产物。
- 文档已更新：涉及流程、配置、输出规范时更新治理文档。
- 有明确回滚方式：通常是 `git revert <merge_commit_or_commit>`。

推荐测试命令：

```powershell
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -m unittest discover -s tests -v
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe -m compileall -q app.py src tests
git diff --check -- <本任务文件列表>
```

---

## 8. 合并方案

### 8.1 小型修复

适用：单点 bug、小范围逻辑修复、文档改动。

流程：

```powershell
git checkout main
git pull --ff-only origin main
git merge --ff-only <branch>
```

如果不能 fast-forward，说明分支历史不线性，改用普通 merge 或 squash。

### 8.2 常规功能分支

适用：有多个提交、需要保留上下文的功能。

流程：

```powershell
git checkout main
git pull --ff-only origin main
git merge --no-ff <branch>
```

合并后：

```powershell
git push origin main
```

### 8.3 提交较乱的实验分支

适用：多次试错、提交粒度混乱，但最终改动有效。

流程：

```powershell
git checkout main
git pull --ff-only origin main
git merge --squash <branch>
git commit -m "feat: summarize final change"
git push origin main
```

### 8.4 worktree 分支

适用：独立目录开发，例如 `D:\video_clip_worktrees\sample-library-calibration`。

流程：

```powershell
cd D:\video_clip_worktrees\<name>
git status --short
git add <files>
git commit -m "<message>"

cd D:\video_clip
git checkout main
git pull --ff-only origin main
git merge --no-ff codex/<name>
git push origin main
```

合并并确认不再需要后，才能移除 worktree：

```powershell
git worktree remove D:\video_clip_worktrees\<name>
git branch -d codex/<name>
```

---

## 9. 什么时候可以删除分支

可以删除的条件：

- 分支已经合并到 `main`。
- `git log main..<branch>` 没有输出。
- 该分支没有 worktree，或 worktree 已确认可删除。
- 分支工作区没有未提交改动。
- 远程分支不再需要。

检查命令：

```powershell
git branch --merged main
git log --oneline main..<branch>
git worktree list
```

删除本地分支：

```powershell
git branch -d <branch>
```

删除远程分支：

```powershell
git push origin --delete <branch>
```

禁止删除：
- 有未提交改动的 worktree 分支。
- 用户明确说“暂时不合并”的功能分支。
- 不确定是否已备份的实验分支。

---

## 10. GitHub 同步规则

不会自动同步 GitHub。只有满足以下条件才推送：

- 用户明确要求“同步到 GitHub / 备份 / push”。
- 本地测试通过。
- 已确认提交内容不包含临时产物和敏感信息。
- 当前分支与远程关系明确。

常规同步：

```powershell
git status --short --branch
git add <files>
git commit -m "<message>"
git push origin main
```

如果是功能分支：

```powershell
git push -u origin codex/<name>
```

---

## 11. 回滚规则

优先使用 `git revert`，不要用破坏性回滚。

单提交回滚：

```powershell
git revert <commit>
```

合并提交回滚：

```powershell
git revert -m 1 <merge_commit>
```

回滚后必须：
- 说明触发原因。
- 跑 smoke 测试。
- 记录回滚结果。

---

## 12. 本项目推荐工作流

### 普通 bug 修复

1. 在 `main` 确认问题。
2. 如用户正在测试 8501，新建 `codex/fix-*` 分支或 worktree。
3. 修复并跑相关测试。
4. 合并到 `main`。
5. 用户确认后推送 GitHub。
6. 删除已合并分支。

### 复杂功能开发

1. 新建 `codex/feature-*` 分支。
2. 如需不影响 8501，使用 worktree。
3. 在分支端口测试，例如 8502/8506。
4. 完成单元测试、smoke、必要端到端测试。
5. 代码审查。
6. 合并到 `main`。
7. 重启主线服务并验证。
8. 推送 GitHub。
9. 删除分支和 worktree。

### 暂不合并的功能

1. 保留分支和 worktree。
2. 不合并到 `main`。
3. 在文档或任务记录中说明保留原因。
4. 定期检查是否继续、废弃或合并。

---

## 13. 验收标准

- 能明确判断一个任务是否需要新分支或 worktree。
- 合并前有固定测试门禁。
- 删除分支前有固定检查命令。
- 不会删除带未提交改动的 worktree。
- `main` 上的 8501 服务不会被无关开发任务打断。
