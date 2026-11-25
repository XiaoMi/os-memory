# ToolSAGE

ToolSAGE 是 SAGE 慢路径的第二步。OS-Memory 的 SAGE 依次执行：

1. 边界巩固（`hypermem/online/consolidation.py`）：裁决 DRED 的待定归属、拆分话题、缝合跨段事件、生成事件卡片；
2. ToolSAGE（`hypermem/online/sage_tools.py`）：逐事件以工具调用细化事件；
3. LongMemory 巩固（`hypermem/online/sage.py`）：把多个事件提炼为长期记忆。

三步只有这一种实现，没有可切换的模式。DRED 归属、原子观测接口和 QA 检索接口不受 ToolSAGE 影响。

## 三个工具

| 工具 | 行为与约束 |
| --- | --- |
| `SplitEvent` | `inspect` 按时间顺序进行语义单遍聚类；`commit` 由 LLM 将所有 cluster ID 恰好分配一次至至少两个子事件。先拆分，再对子事件生成语义。 |
| `SummarizeEvent` | 生成标题、摘要、事件类型和锚点；实体必须来自该事件允许的键集。提供时间邻近的话题锚点帮助复用，底层仍沿用原有向量匹配。 |
| `ExtractFacts` | 摘要成功后抽取事实；每条事实必须引用当前事件内非空的观测 ID 集合。可以明确返回空事实集。 |

LLM 每轮只能提议一次工具调用。执行器校验参数、证据归属、事件版本和拆分成员守恒，再在事务中写入；错误以工具反馈返回模型。成功写入与调用日志一起提交，恢复时优先执行已经收到的回复，不重复执行已完成任务。接口请求前中断可能重试请求，但不会重复提交其写入。

拆分保留不可变的历史父事件。选择 `retain_parent` 后，可视化展示父事件与 `part-of` 边；原始观测仍由叶事件唯一持有，父节点不重复拥有事实证据，也不进入叶事件检索。子事件刷新边界巩固生成的事件卡片骨架，SAGE 标题和摘要存于 `sage_events`。

## 使用

`HyperMemory.build()` 会在 DRED 之后自动运行 SAGE 全部三步。评测场景下，`eval/` 目录里的 `python -m pmm_bench.final run … --stage sage`（或 `--stage all`）在 `dred` 之后运行全部三步：边界巩固未完成时先完成它，然后运行 ToolSAGE 和 LongMemory，最后写出 `frozen.json`。ToolSAGE 拒绝在边界巩固未完成的数据库上运行。

模型使用 `sage_config.json` 中的 `llm` 配置及对应 API key 环境变量，接口需要支持标准 `tools` / `tool_calls` / `tool` 消息。编码器沿用 DRED 的配置，不需要新增模型。

工具预算采用 `sage_tools.py` 中 `DEFAULT_TOOLS` 的默认值：

```json
{
  "max_rounds": 12,
  "max_calls": 12,
  "max_context_chars": 60000,
  "max_clusters": 32,
  "max_split_depth": 2,
  "split_similarity": 0.65,
  "topic_window_hours": 168,
  "max_topic_candidates": 16
}
```

同一运行不能修改预算后静默重置已耗尽的任务。预算计数跨重启保留。

工具执行器遵守 `sage_config.json` 的 `event.concurrency`：默认值为 1，保持串行；大于 1 时按该上限分批，并行等待独立事件的模型回复。同一事件仍逐轮调用工具；所有回复落盘、工具执行、证据与版本校验、嵌入请求及数据库提交仍在拥有连接的主线程完成。响应按到达顺序保存，工具按批内事件顺序应用。未完成任务的预算跨恢复保留，过期回复不会写入派生层。

并发批次的新事件上下文读取批次开始时的话题锚点快照，**不保证与串行逐条读取不断更新的锚点产生相同的模型文本输出**；候选锚点只是提示，写入时仍校验原始观测、实体键并对当前锚点去重。相同合法工具建议的串行/并行投影等价有回归测试，但这不等于真实 LLM 输出逐字等价。实际执行模式记录在数据库 `sage_tool_execution` 及 `sage_report.json.execution` 中。PMM-Bench 构建使用 8 路并发。

`sage` 阶段结束后，如仍有耗尽重试次数的任务，可显式使用 `--stage sage --retry-unresolved` 恢复。每次授权只给已耗尽且未完成的任务增加一次机会，并将授权写入任务日志；不会清零计数、重做成功任务或跳过失败任务。该参数只用于 `sage` 和 `all`。恢复前应确认原进程已结束；若使用了新源码快照，仍须按原有版本校验流程记录变更原因。

恢复时，Event Card 仅有生命周期展示变化（例如休眠转封存）不会创建新任务：沿用原任务 ID、原请求上下文、响应和重试计数。候选、证据文本、成员版本及语义指纹的变化仍会形成不同任务，不会因为此优化复用过期证据。

## 完成判定与边界

- `sage_tool_jobs` / `sage_tool_jobs.json` 保存上下文、回复、错误反馈、调用计数及失效派生数据，便于审计和恢复。日志包含观测内容，需按用户数据管理。
- 失败处理（`hypermem/online/failures.py`）：每个模型任务先在预算内重试，仍失败则跳过，构建继续。被跳过的事件保留 DRED 归属和边界巩固的事件卡片，不参与 LongMemory。`sage_report.json` 的状态记为 `complete_with_skipped_failures`，并在 `skipped_failures` 和 `completion_gate` 中列出跳过的任务与覆盖率；评测的 `contract.json` 记录跳过数量。
- 模型服务本身失败（`ModelServiceError`：连接不上、超时、过载，或拒绝所配置的密钥 / 地址 / 模型）不算任务失败：不消耗任何任务的重试预算，当前阶段立即停止，同一批中已收到的其他回复照常落盘。服务恢复后重跑同一阶段即从任务日志续跑。
- 结构审计失败、证据失配或仍有待处理任务时标记 `needs_review`，QA 不会运行。
- 上下文能够容纳时提供所有成员文本。过大的事件只允许拆分或保留待处理，不会用前几个片段冒充全量事实抽取；暂未实现超大事件的分页证据工具。
- LLM 拆分沿用成员守恒、唯一归属、版本与提交记录检查，不要求拆分满足最小内聚增益。结构合法不代表语义正确，仍需用 QA 和人工抽样评估质量。
- 此实现是离线/闲置阶段的有界工具执行器，运行时持有数据库写锁；不宣称支持手机交互路径中的并发实时写入。
- 单元测试中的本地编码器与模型替身只验证执行契约，不代表用户数据上的准确率。全量运行是否完成，以该运行目录的构建审计与逐题评测记录为准。

## 评测

PMM-Bench 评测统一用仓库根目录的 `python -m pmm_bench`，见 `docs/EVALUATION.md`。
