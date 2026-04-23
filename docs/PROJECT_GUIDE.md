# 项目治理总入口（Project Guide）

**文档版本**: v1.0
**创建时间**: 2026-04-15
**最后更新**: 2026-04-18
**责任人**: AI
**变更日志**:
- 2026-04-15: 初始创建
- 2026-04-18: 添加GLOSSARY.md链接，统一文档头

---

## 0. 术语与定义

所有术语定义请参考：[GLOSSARY.md](./GLOSSARY.md)

---

## 1. 文档目标
- 统一本项目的需求、设计、开发、测试、发布与变更流程。
- 让后续所有改动可追溯、可复盘、可验收。

---

## 2. 治理价值主张

### 2.1 为什么需要这套规范？
**问题1：无规范开发的痛点**
- 需求不明确，反复修改
- 代码改动无记录，出问题无法追溯
- 测试不完整，发布后频繁bug
- 团队协作混乱，职责不清
- 新人上手慢，没有统一流程

**问题2：我们的实际经历**
- 合唱识别问题：需求明确，但缺乏测试验证，反复调优
- AED策略问题：阈值调整无记录，忘记为什么这么调
- 文档问题：之前没有统一文档，新人不知道从哪里开始

**问题3：这套规范解决什么？**
- ✅ 需求明确：PRD模板+验收标准，避免反复修改
- ✅ 可追溯：所有变更有记录，出问题可以查
- ✅ 质量保障：测试策略+发布清单，减少bug
- ✅ 职责清晰：角色职责明确，避免推诿
- ✅ 新人友好：30分钟上手路径，统一流程

### 2.2 这套规范带来什么价值？
**价值1：提高开发效率**
- 减少反复修改：需求一次写清楚
- 减少排查时间：所有变更有记录
- 减少发布问题：发布前有完整检查

**价值2：提升代码质量**
- 测试保障：每个改动都有测试
- 可维护性：配置统一，文档完整
- 可扩展性：架构清晰，易于扩展

**价值3：改善团队协作**
- 职责清晰：谁做什么一目了然
- 沟通高效：用统一的术语和流程
- 知识传承：新人可以快速上手

### 2.3 什么时候用这套规范？
- ✅ 新功能开发：必须走完整流程
- ✅ Bug修复：至少走Issue+测试
- ✅ 参数调整：至少走CONFIG_POLICY+测试
- ⚠️ 紧急Hotfix：可以跳过部分流程，但24小时内补齐文档
- ❌ 注释/文档修改：可以简化，但需要记录

---

## 3. 角色职责

### 3.1 产品负责人（Product Owner）
**核心职责**：
- 撰写和维护PRD（产品需求文档）
- 明确需求的验收标准
- 优先级排序和需求管理
- 验收开发成果

**具体工作**：
- 收到需求后，填写PRD模板
- 与技术负责人一起评审需求
- 明确验收标准，必须可量化
- 发布前验收功能是否符合预期

**对应文档**：PRD.md

---

### 3.2 技术负责人（Tech Lead）
**核心职责**：
- 撰写和维护TECH_SPEC（技术规范）
- 技术方案设计和评审
- L2级别变更审批
- L3级别变更组织团队评审
- 代码质量把关

**具体工作**：
- 收到PRD后，撰写TECH_SPEC
- 评估技术方案的可行性和风险
- 审批L2级别的变更
- 组织L3级别的团队评审
- 检查代码质量和架构设计

**对应文档**：TECH_SPEC.md、CHANGE_POLICY.md

---

### 3.3 开发人员（Developer）
**核心职责**：
- 按照规范进行编码实现
- 编写和执行单元测试
- 记录变更日志
- 产出run_summary.json

**具体工作**：
- 根据TECH_SPEC进行编码
- 遵循代码规范和最佳实践
- 为核心函数编写单元测试
- 执行Smoke测试和Regression测试
- 记录变更日志到CHANGE_POLICY
- 产出run_summary.json测试报告

**对应文档**：PIPELINE_SPEC.md、TEST_STRATEGY.md、CHANGE_POLICY.md

---

### 3.4 测试人员（Tester）
**核心职责**：
- 执行Regression测试和Performance测试
- 维护测试样本集
- 生成和对比测试报告
- 验证发布质量

**具体工作**：
- 执行每周Regression测试
- 执行Performance测试
- 维护和更新测试样本集
- 对比当前结果与基线
- 发布前做最终验证

**对应文档**：TEST_STRATEGY.md

---

### 3.5 发布负责人（Release Manager）
**核心职责**：
- 检查RELEASE_CHECKLIST（发布清单）
- 执行发布流程
- 发布后验证
- 回滚决策和执行

**具体工作**：
- 发布前逐项检查RELEASE_CHECKLIST
- 确认所有必检项都通过
- 执行发布操作
- 发布后执行验证
- 出现问题时决策是否回滚
- 执行回滚流程（如果需要）

**对应文档**：RELEASE_CHECKLIST.md

---

## 4. 适用范围
- 适用于 `D:\video_clip` 项目全部主链路开发工作。
- 当前主关注链路：视频切片。
- 字幕链路要求：不回退现有可用能力。

## 3. 输入与输出
- 输入：新需求、缺陷、优化项、线上问题、实验结论。
- 输出：
  - 需求产物：PRD 变更条目 + 验收标准。
  - 技术产物：TECH_SPEC 变更条目 + 配置变更说明。
  - 验收产物：`run_summary.json` + 对比结论（达标/不达标）。

## 4. 治理文档索引
- [PRD](./governance/PRD.md)
- [TECH_SPEC](./governance/TECH_SPEC.md)
- [PIPELINE_SPEC](./governance/PIPELINE_SPEC.md)
- [CONFIG_POLICY](./governance/CONFIG_POLICY.md)
- [TEST_STRATEGY](./governance/TEST_STRATEGY.md)
- [RELEASE_CHECKLIST](./governance/RELEASE_CHECKLIST.md)
- [ISSUE_TEMPLATE](./governance/ISSUE_TEMPLATE.md)
- [CHANGE_POLICY](./governance/CHANGE_POLICY.md)
- [ROADMAP_EXECUTION_PLAN](./ROADMAP_EXECUTION_PLAN.md)

## 5. 统一执行流（必须）
1. 需求进入：先更新 `PRD` 条目与验收口径。
2. 开发前：先更新 `TECH_SPEC` 变更段与影响面。
3. 开发中：按任务清单执行并记录关键日志。
4. 开发后：按 `TEST_STRATEGY` 产出 `run_summary.json` 与对比结论。
5. 合入前：逐项通过 `RELEASE_CHECKLIST`。

## 6. 状态标签（统一）
- `planned`
- `in_progress`
- `blocked`
- `verified`
- `done`

## 7. 30 分钟上手路径
1. 阅读本文件 + `ROADMAP_EXECUTION_PLAN.md`。
2. 在 `PRD` 新增需求条目。
3. 在 `TECH_SPEC` 写改动方案。
4. 按 `TEST_STRATEGY` 跑一轮 smoke 并产出 `run_summary.json`。
5. 用 `RELEASE_CHECKLIST` 自检。

## 8. 验收标准
- 本文件中引用的治理文档全部存在且可访问。
- 任一需求能在 30 分钟内从“需求输入”进入“可执行任务清单”。
- 任一代码改动能按“三类最小产物”完成闭环。
