# OS-Memory 使用文档

本文档面向两类读者：

1. **想把 OS-Memory 当记忆插件接入自己 agent 的开发者** —— 见 [一、接入我们的系统](#一接入我们的系统)。
2. **想在 PMM-Bench 上复现/对比评测的研究者** —— 见 [二、在 PMM-Bench 上测评](#二在-pmm-bench-上测评)。

## 仓库结构

```text
os-memory/
├── README.md                            # 使用说明书（安装、数据接入、接口、评测）；README.zh.md 为中文版
├── AGENTS.md                            # 给编码 agent 读的安装指引（读完可自动安装）与改代码须知
├── pyproject.toml                       # pip 安装：name = hyperos-mem；可选依赖 [mcp]
├── hypermem/                            # 我们的记忆系统（发布核心）：Python 包，对外接口在 api.py
│   ├── api.py                           #   HyperMemory 公开接口
│   ├── mcp_server.py                    #   MCP server（add / build / search / read / status 工具）
│   ├── online/                          #   DRED 引擎 + SAGE 慢路径 + 检索
│   ├── memory/ schema/                  #   事件记忆存储与数据模型
│   └── ...
├── tests/                               # 核心包的单元测试
├── docs/
│   ├── USAGE.md                         # 本文档
│   ├── OS-MEMORY.md                   # 方法概述、PMM-Bench 与结果
│   └── VERSION.md SAGE_TOOLS.md ...     # 版本声明与 ToolSAGE 说明
└── eval/                                # 评测（PMM-Bench + baselines），与系统解耦
    ├── pmm_bench/                       # 评测 runner、judge、报告；final.py 编排入口
    ├── systems/<baseline>/              # 各对比系统源码（固定上游 commit）
    ├── configs/ run.py examples/ tests/
    └── docs/EVALUATION.md               # 评测详解
```

系统与评测完全解耦：核心包不依赖评测代码，评测通过安装好的 `hypermem` 包（或上一级的仓库根目录）调用系统。

---

## 一、接入我们的系统

OS-Memory 对外是一个 Python 库，核心是 `HyperMemory` 类：

```python
mem.add(records)             # 保存记忆：批量写入一批记录（列表或生成器）
mem.build()                  # 构建记忆图（DRED + SAGE），写完后执行一次
mem.search(query)            # 查找相关记忆：自然语言语义检索
mem.read(...)                # 读取记忆：按事件 ID / 时间 / 实体 / anchor 结构化浏览
```

| 能力 | 方法 | 需要模型服务 |
| --- | --- | --- |
| 保存记忆（批量写入） | `add(records)` | 否 |
| 构建记忆图 | `build()` | LLM + 嵌入 |
| 查找相关记忆 | `search(query)` | 嵌入 |
| 读取记忆 | `read(...)` | 否 |

`add` 收的是**内存中的记录列表**（也可以是生成器）：数据从哪里读、原来是什么格式由调用方决定，整理成统一的记录结构后直接传入，不需要先写成文件。每条记录外层只有 `type`、`timestamp`、`data` 三个字段，其余信息都放在 `data` 里；系统逐条校验、逐条落盘，某条不合法会报错（指出是第几条）并整批回滚。记录格式见第 2 节和第二章。

### 1. 安装

提供两种安装方式，按用途选：

| 方式 | 适合 | 得到什么 |
| --- | --- | --- |
| **pip** | 在自己的 Python 代码 / agent 里调用 | `from hypermem import HyperMemory`，可用 `add` / `build` / `search` / `read` |
| **MCP** | 在 Claude Desktop、Cursor、Claude Code 等 MCP 客户端里直接用 | 与库同名的工具 `add` / `build` / `search` / `read`，外加 `status` |

也可以让 agent 代为安装：见本节末「让 agent 自动安装」。

**环境要求**：Python ≥ 3.10，git。支持 Linux 和 macOS。

> **Windows 限制**：数据库的写锁依赖 POSIX 的 `fcntl`，Windows 上无法打开文件型记忆库（报 `No module named 'fcntl'`），因此无法在 Windows 上建库或提供 MCP 服务。Windows 用户请在 WSL 中安装使用。

#### 方式一：pip

```bash
git clone https://github.com/XiaoMi/os-memory.git
cd os-memory
python -m venv .venv
source .venv/bin/activate                      # Windows（未用 WSL 时）：.venv\Scripts\activate
pip install -e .
```

验证：

```bash
python -c "from hypermem import HyperMemory; print('hypermem ok')"
```

输出 `hypermem ok` 即安装成功。之后按第 3～5 节配置模型服务、写入并构建记忆。

#### 方式二：MCP

MCP server 的工具与库的同名方法一一对应，参数相同：

| 工具 | 对应库方法 | 说明 |
| --- | --- | --- |
| `add(records=None, jsonl_path=None, user=None)` | `add` | 写入记录，返回记录数与观测数。`records`（记录数组）和 `jsonl_path`（JSONL 文件，每行一条记录，须为 server 所在机器上的**绝对路径**，适合大批量）二选一 |
| `build()` | `build` | 构建记忆图。**后台执行、立即返回** `{"status": "running", ...}`，进度和结果用 `status` 查看 |
| `search(query, top_k=20, as_text=True)` | `search` | 语义查找 |
| `read(...)` | `read` | 结构化读取，参数同库 |
| `status()` | `is_started` / `is_built` / `count()` | 库是否存在、建库是否已开始 / 已完成、各类计数、已配置的服务，以及后台 build 的状态 |

`build` 要调 LLM、可能耗时数分钟，超过 MCP 客户端等待一次工具调用的时限，所以在后台跑。`status` 返回的 `build` 字段为 `{"state": "running", "progress": ...}`、`{"state": "done", "report": ...}`（`report` 即库 `build()` 的返回值）或 `{"state": "failed", "error": ...}`。build 进行中，除 `status` 外的工具都会报错并提示稍后重试。`error` 以 `ModelServiceError` 开头时，表示 LLM 或嵌入服务出了故障：修好服务后再调一次 `build` 即从中断处继续（见第 9 节 `build`）。

典型流程：`add` → `build` → 反复 `status` 直到 `done` → `search` / `read`。也可以照旧用库离线建好库，MCP 只用来查询。

**① 安装**（带 `[mcp]` 可选依赖，会额外装上 MCP SDK）：

```bash
pip install -e ".[mcp]"
python -c "from hypermem.mcp_server import build_server; build_server(); print('mcp ok')"
```

输出 `mcp ok` 即可。安装后还有一个等价的启动命令 `hyperos-mem-mcp`（位于虚拟环境的 `bin/` 下），可代替 `python -m hypermem.mcp_server`。

**② 准备记忆库路径**：`HYPERMEM_DB` 指向的文件不存在时，第一次 `add` 会新建它（所在目录须已存在）；其他工具遇到不存在的路径会报错，不会新建空库。已有用库建好的数据库，直接填它的绝对路径即可。

**③ 在客户端注册 server**。Claude Desktop 和 Cursor 用同一段配置：

```json
{
  "mcpServers": {
    "hypermemory": {
      "command": "/absolute/path/to/os-memory/.venv/bin/python",
      "args": ["-m", "hypermem.mcp_server"],
      "env": {
        "HYPERMEM_DB": "/absolute/path/to/memory.db",
        "HYPERMEM_EMBEDDING_URL": "https://your-gateway/v1/embeddings",
        "HYPERMEM_EMBEDDING_MODEL": "bge-m3",
        "HYPERMEM_EMBEDDING_DIMENSION": "1024",
        "HYPERMEM_EMBEDDING_API_KEY": "sk-...",
        "HYPERMEM_LLM_BASE_URL": "https://your-gateway/v1",
        "HYPERMEM_LLM_MODEL": "qwen2.5-7b",
        "HYPERMEM_LLM_API_KEY": "sk-..."
      }
    }
  }
}
```

只用 MCP 查询、不在 MCP 里建库时，可以省掉三个 `HYPERMEM_LLM_*`。

| 客户端 | 配置文件位置 |
| --- | --- |
| Claude Desktop（macOS） | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop（Windows） | `%APPDATA%\Claude\claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json`（全局）或项目下 `.cursor/mcp.json` |

Claude Code 用命令注册：

```bash
claude mcp add hypermemory \
  -e HYPERMEM_DB=/absolute/path/to/memory.db \
  -e HYPERMEM_EMBEDDING_URL=https://your-gateway/v1/embeddings \
  -e HYPERMEM_EMBEDDING_MODEL=bge-m3 \
  -e HYPERMEM_EMBEDDING_DIMENSION=1024 \
  -e HYPERMEM_LLM_BASE_URL=https://your-gateway/v1 \
  -e HYPERMEM_LLM_MODEL=qwen2.5-7b \
  -- /absolute/path/to/os-memory/.venv/bin/python -m hypermem.mcp_server
```

改完配置后重启客户端，工具列表里出现 `add`、`build`、`search`、`read`、`status` 即接入成功。

**环境变量说明**：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `HYPERMEM_DB` | 是 | 记忆库文件的**绝对路径**；不存在时由 `add` 新建，其他工具报错 |
| `HYPERMEM_EMBEDDING_URL` | `search`、`build` 需要 | OpenAI 兼容的嵌入地址，以 `/v1/embeddings` 结尾 |
| `HYPERMEM_EMBEDDING_MODEL` | `search`、`build` 需要 | 嵌入模型名 |
| `HYPERMEM_EMBEDDING_DIMENSION` | `search`、`build` 需要 | 嵌入向量维度（整数） |
| `HYPERMEM_EMBEDDING_API_KEY` | 服务需要时 | 嵌入服务的 API key |
| `HYPERMEM_LLM_BASE_URL` | `build` 需要 | OpenAI 兼容的 LLM 地址，以 `/v1` 结尾 |
| `HYPERMEM_LLM_MODEL` | `build` 需要 | LLM 模型名 |
| `HYPERMEM_LLM_API_KEY` | 服务需要时 | LLM 服务的 API key |

可选调优项，与第 3 节库配置的同名字段一一对应：`HYPERMEM_LLM_PATH`、`HYPERMEM_LLM_MAX_TOKENS`、`HYPERMEM_LLM_TIMEOUT`、`HYPERMEM_LLM_RETRIES`、`HYPERMEM_EMBEDDING_BATCH_SIZE`、`HYPERMEM_EMBEDDING_PROTOCOL`、`HYPERMEM_EMBEDDING_TIMEOUT`、`HYPERMEM_EMBEDDING_RETRIES`。

几点注意：

- 嵌入配置必须和**建库时用的完全一致**（同一服务、同一模型、同一维度），否则检索结果无意义。
- `add`、`read`、`status` 只需要 `HYPERMEM_DB`；只有 `build` 调用 LLM。
- `command`、`HYPERMEM_DB` 和 `add` 的 `jsonl_path` 都要用**绝对路径**：客户端以子进程方式启动 server，不会继承你激活的虚拟环境和当前目录。
- 一个库只建一次：`build` 开始后 `add` 会报错，建完后 `build` 会报错；要加入新记忆请换一个 `HYPERMEM_DB` 重建。
- 已兼容 MCP SDK 1.x 与 2.x（实测 1.30 与 2.3）。

#### 让 agent 自动安装

仓库根目录的 [`AGENTS.md`](../AGENTS.md) 是写给编码 agent（Claude Code、Codex、Cursor 等）的安装指引：环境检查、clone、建虚拟环境、pip 安装、验证、建库、注册 MCP，每一步都有可执行命令和预期输出。对你的 agent 说「读一下 hypermemory 仓库的 AGENTS.md，帮我把它装上」即可。agent 会向你询问模型服务地址和密钥，密钥只放在环境变量或客户端配置里，不会写进仓库文件。

### 2. 记录格式

每条记录外层只有三个字段：`type`、`timestamp`、`data`。`data` 里放所有类型共用的**通用字段**和该类型的**专属字段**。

```python
records = [
    {"type": "screen", "timestamp": "2026-05-03T09:00:00+08:00",
     "data": {"from_app": "铁路12306",
              "action_description": "在 12306 查询 5月8日 北京南→上海虹桥 G15 二等座余票"}},
    {"type": "sms", "timestamp": 1777770360000,
     "data": {"from_app": "短信", "content": "【铁路12306】订单E2026050301已支付", "sender": "12306",
              "anchors": {"order_id": "E2026050301"}}},
    {"type": "chat", "timestamp": "2026-05-03T11:46:40+08:00",
     "data": {"from_app": "MyAssistant", "query": "我不吃香菜，帮我找家附近的川菜馆",
              "answer": "为你找到蜀味轩，已避开香菜菜品"}},
]
```

- **外层**：`type`（10 种之一）、`timestamp`（整数 Unix 毫秒，或 ISO 8601 字符串；入库统一存为毫秒）、`data`，全部必填，不能有别的字段。
- **`data` 里的通用字段**（都可选）：`from_app`（来源应用）、`record_id`（调用方的记录 ID，同库不可重复）、`anchors`（订单号等强标识）、`entities`（联系人、商家等实体）、`latitude` / `longitude`（事情发生时设备所在的位置）。
- 各类型的专属字段见第二章；完整说明与每种类型的示例见 [README.md](../README.md) 第 4 章。

### 3. 配置模型服务

`build` 和 `search` 需要一个 OpenAI 兼容的 LLM 端点和一个嵌入端点。密钥放环境变量，配置里只写变量名。

```python
import os
from hypermem import HyperMemory

os.environ["OPENAI_API_KEY"] = "sk-..."

config = {
    "llm": {
        "base_url": "https://your-gateway/v1",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "embedding": {
        "url": "https://your-gateway/v1/embeddings",
        "model": "bge-m3",
        "dimension": 1024,
        "api_key_env": "OPENAI_API_KEY",
    },
}
```

### 4. 工作模型：先喂、再建、后查

OS-Memory 是 **DRED（在线、不调模型的事件归属）+ SAGE（空闲时用 LLM 把事件消化成事实与长期记忆）**。使用分三步：

```text
add(记录)  ──►  build()  ──►  search / read
   本地、无网络      DRED + SAGE       查找 / 读取
                   需要 LLM+嵌入      search 需要嵌入
```

- `add` 只把记录落到本地库，不发网络请求，调用顺序随意（建库时按时间排序）。
- `build()` 跑一次，构建可检索的事件图（调用 LLM 和嵌入服务）。模型服务出故障时它会报错停下，修好服务后再调一次即可接着建。
- 建完后可任意多次 `search` / `read`。
- 一个数据库建一次图；`build()` 开始后就不能再 `add`，要再记一批新内容，开一个新库。

### 5. 最小示例

```python
from hypermem import HyperMemory

mem = HyperMemory("./user_memory.db", config=config)

# 批量写入：传列表；数据量大时传生成器，边读边写
result = mem.add(records, user="U01")
print(result)   # {'records': 3, 'observations': 4}（带 answer 的 chat 存为两条观测）

# 一次性建图：DRED + SAGE
report = mem.build()
print(report)   # {'status': 'built', 'observations': 4, 'events': ...}

# 查询
result = mem.search("我最近的火车票订单？")
print(result["context"])    # 可直接拼进 prompt 的文本

mem.close()
```

已经整理成 JSONL 文件时，自己逐行读取即可：

```python
import json

with open("records.jsonl", encoding="utf-8") as f, HyperMemory("./user_memory.db", config=config) as mem:
    mem.add((json.loads(line) for line in f if line.strip()), user="U01")
    mem.build()
```

### 6. 图片 / 多模态数据怎么进系统

OS-Memory 从不打开任何媒体文件。图片、语音、视频在进入系统之前，由上游先转成文本——OCR、ASR、VLM 描述——填进 `data` 的对应字段。媒体文件路径只作为 `image_path`、`audio_path` 保留，用于溯源，系统不读取。

```python
{"type": "gallery", "timestamp": "2026-05-03T09:16:40+08:00",
 "data": {"from_app": "相册", "image_description": "一张高铁车票照片",
          "ocr": "G15 北京南 08:00 上海虹桥 二等座", "image_path": "/sdcard/DCIM/ticket.jpg"}}
```

没有 `image_description` 的相册记录会被拒收——转文本是接入方的责任，系统只认文本。

### 7. 聊天数据怎么进系统

手机 IM（微信 / QQ）的聊天内容在真实数据里是以截图经 VLM 概括的描述进来的，使用 `screen` 类型。

用户与 agent 的对话用 `chat` 类型，**一轮问答 = 一条记录**，`query` 必填、`answer` 可选。系统在文字前标明说话人（"用户：……""助手：……"），构建时据此区分用户本人的话和助手的回答。带同一个 `session_id` 的多轮对话会优先归入同一个事件。

```python
{"type": "chat", "timestamp": "2026-05-03T11:46:40+08:00",
 "data": {"from_app": "MyAssistant", "session_id": "s-20260503-01",
          "query": "我不吃香菜，帮我找家附近的川菜馆", "answer": "为你找到蜀味轩，已避开香菜菜品"}}
```

### 8. 录音怎么进系统

`recording` 的 `timestamp` 是录音开始的时间，`data.transcript` 是一个列表，每一段结构相同：`text`、`offset_ms`（距录音开始的毫秒数）、`speaker`（`user` 为用户本人，其他值为对方，未知为 `null`）。整段转写就是只有一段的列表。

```python
{"type": "recording", "timestamp": "2026-05-03T09:30:00+08:00",
 "data": {"from_app": "通话录音",
          "transcript": [{"text": "下周三去杭州出差", "offset_ms": 0, "speaker": "user"},
                         {"text": "好的我帮你订高铁", "offset_ms": 5000, "speaker": "张三"}]}}
```

每一段存成一条观测，系统在文字前标明说话人。

### 9. API 参考

#### `HyperMemory(db_path=":memory:", *, config=None, source_id=None)`
打开或新建一个记忆库。`db_path` 为 SQLite 文件路径，或 `":memory:"` 做临时库。`config` 见上；只有 `build` / `search` 需要它，`add` 没有也能用。

#### `add(records, *, user=None) -> dict`
批量写入记忆，返回 `{"records": 记录数, "observations": 观测数}`。本地操作，不调模型。
- `records`：记录列表，或任意可迭代对象（如生成器）；每个元素是一条 `type` + `timestamp` + `data` 结构的记录。
- 逐条校验、逐条落盘入库；传生成器时边读边写，数据量不受内存限制。
- 每条记录先被翻译成系统内部的格式再参与构建，同时**原样存入数据库的 `raw_records` 表**（`timestamp` 统一转成毫秒），作为溯源与审计的权威来源；解析出的观测存入 `observations` 表，通过 `raw_id` 指回原始记录。
- 某条记录不合法会报 `ValueError`（如 `records[12]: type 'sms' requires data.content`），同时整批回滚；误传文件路径等非记录列表时报 `TypeError`。
- 带 `answer` 的 `chat`、多段的 `recording`、同时有经纬度和 `ssid` 的 `position` 会产生多条观测，所以观测数可能大于记录数。
- 带 `record_id` 的记录，同库内重复会报错；`build()` 开始后再调用也会报错。

#### `build(*, progress=None) -> dict`
构建记忆图：DRED 快路径 → SAGE 边界巩固 → ToolSAGE + 长期记忆。返回简报 `{"status": "built", "resumed": ..., ...}`。未配置模型、未加任何记录、或已建完都会报错。

**模型服务故障时停下，恢复后续建。** LLM 或嵌入服务连不上、超时、过载（408 / 429 / 5xx，重试后仍失败），或拒绝所配的密钥 / 地址 / 模型（401 / 403 / 404）时，`build()` 抛出 `ModelServiceError`（`from hypermem import ModelServiceError`），错误信息写明是哪个服务地址出的错。此时：

- 已完成的工作都已落盘，**任何任务的重试次数都不会因为这次故障而消耗**；
- 库处于"已开始、未完成"状态：`is_started` 为 `True`、`is_built` 为 `False`，`search` 会报错提示先完成构建；
- 修好服务后再调一次 `build()`，它从中断处继续，返回的简报里 `"resumed": True`。进程崩溃或被中断后同样可以这样续建。

```python
from hypermem import HyperMemory, ModelServiceError

try:
    mem.build()
except ModelServiceError as exc:
    print("模型服务故障，构建已暂停：", exc)   # 修好服务后再调一次 mem.build()
```

模型**回复了但内容无效**（格式不对、证据不符等）不算服务故障：该任务在自己的重试次数内重试，仍失败就跳过并记录，构建继续，`sage.status` 为 `complete_with_skipped_failures`，仍算建完。

`progress` 可选，是一个回调，以 `progress(info)` 的形式被调用：`info` 是一个 dict，`info["stage"]` 为 `"dred"`、`"sage_boundary"` 或 `"sage_event_tools"`，其余键是该阶段的计数。

#### `search(query, *, top_k=20, as_text=True) -> dict`

按自然语言查询检索。返回一个**完整 dict**（不是单条文本），其中 `events` 和 `long_memories` 都是**列表**——多条召回体现在这两个列表里。

> **召回的单元不是你写入的那条记录。** 你用 `add` 写入的记录，入库后是一条原子观测。建库后，DRED 把多条观测归并成**事件（event）**，SAGE 再消化成**事实（fact）**和跨事件的**长期记忆（long_memory）**。所以检索召回的是这些高层单元：一个 event 通常聚合了好几条原始观测，原始观测只作为每条 fact 的 `support` 溯源 ID 挂在下面。层次是 `长期记忆 → 事件 → 事实 → 支撑观测`。

返回结构：

```python
{
    "query": "我去杭州的高铁订好了吗？",
    "events": [                       # 召回的事件：按得分取前 top_k 个，再按时间排序
        {
            "event_id": "E...",
            "score": 0.82,            # 该事件的检索分
            "title": "预订杭州高铁",
            "summary": "……",          # SAGE 写的事件摘要
            "time_span": ["2026-05-03T01:33:20+00:00", "2026-05-03T01:33:20+00:00"],
            "facts": [                # 从该事件抽取的事实
                {"text": "订好了下周三去杭州的高铁",
                 "support": ["O-abc...", "O-def..."]}  # 支撑该事实的原始观测 ID
            ]
        },
        # ... 更多事件
    ],
    "long_memories": [                # 召回的多个长期记忆
        {"memory_id": "M...", "score": 0.75, "text": "用户近期有杭州出行计划",
         "evidence": ["O-abc..."]}    # 支撑它的观测 ID
    ],
    "scores": {"E...": 0.82, "M...": 0.75},   # 各召回节点的分数表
    "context": "……"                   # 便利字段：把以上召回拼成一段 prompt-ready 文本
}
```

- `top_k`：最多返回多少个事件（默认 20，须为正整数）。召回的事件多于 `top_k` 时保留得分最高的 `top_k` 个，再按时间排序返回。
- `as_text`：默认 `True` 时额外生成 `context`——把所有召回的事件/长期记忆拼成一段现成文本，适合直接塞进 LLM 的 prompt。设 `False` 则不生成，只给结构化字段。

**两种取法**：要文本直接拼进 prompt，就用 `result["context"]`；要逐条处理（带分数、带溯源观测 ID），就遍历 `result["events"]` / `result["long_memories"]`：

```python
result = mem.search("我去杭州的高铁订好了吗？")

# 取法 A：直接要一段 prompt 文本
prompt_context = result["context"]

# 取法 B：要结构化、逐条处理
for ev in result["events"]:
    print(ev["score"], ev["title"])
    for fact in ev["facts"]:
        print("  ", fact["text"], "←", fact["support"])   # 溯源到原始观测
```

#### `read(*, event_id=None, start=None, end=None, entity=None, anchor=None, limit=20, include_facts=True, include_observations=False, include_raw=False, include_long_memories=False) -> dict`

按**事件 ID、时间、实体、anchor** 等条件结构化读取记忆，用于给人展示、浏览。与 `search` 不同：`search` 是语义检索（给自然语言问题、按向量相似度召回，需要嵌入服务、建库后才能用）；`read` 是结构化浏览（按 ID / 时间 / 实体直接查，**不调任何模型、不需要嵌入**）。

**记忆的层次结构**——理解参数前先看记忆是怎么分层的：

```text
长期记忆 (long_memory)         ← 跨多个事件的总结
    └─ 事件 (event)            ← 一次完整的事（查票+付款+出行）
         ├─ 事实 (fact)        ← SAGE 从事件里抽的一句话结论
         └─ 观测 (observation) ← 解析后的单条记录（文本/anchor/时间）
              └─ 原始记录 (raw) ← 你 add 提交的那条记录
```

`read` 默认返回**事件层**；下面几个 `include_*` 参数决定要不要把上下各层一并带出来。

**筛选参数**（决定返回哪些事件）：

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `event_id` | str | 按 ID 精确读一个事件；给了它就忽略其余筛选条件 |
| `start` | str / datetime / date | 只保留开始时间 ≥ 该时间的事件，如 `"2026-05-03"`、`"2026-05-03 08:30"`、`"2026-05-03T08:30+08:00"` |
| `end` | str / datetime / date | 只保留开始时间 ≤ 该时间的事件，写法同 `start`；只写日期时包含当天全天 |
| `entity` | str | 只保留实体列表含该字符串的事件（大小写模糊匹配） |
| `anchor` | str | 只保留 anchors 里含该值的事件（如 `"E2026050301"` 找某订单相关的一切） |
| `limit` | int | 最多返回多少个事件（默认 20） |

> `start`/`end` 可只给一个或都给；没写时区的时间按本机时区理解（MCP 下为服务器所在机器的时区），写法无法识别时报错。`entity` 和 `anchor` 可同时给，取交集。

**层级参数**（决定每个事件带出哪些层，从"消化后"到"最原始"）：

| 参数 | 默认 | 带出哪一层 | 说明 |
| --- | --- | --- | --- |
| `include_facts` | `True` | **事实**：SAGE 从事件抽出的一句话结论 + 支撑它的观测 ID | 展示"这个事件得出了什么" |
| `include_observations` | `False` | **观测**：解析后的单条记录（文本、anchors、来源 app、时间） | 展示"事件由哪些记录组成" |
| `include_raw` | `False` | **原始记录**：你 `add` 提交的原始记录 | 展示"用户原封不动提交了什么"；按 `raw_id` 去重（一条 recording 的多段只带一份原文） |
| `include_long_memories` | `False` | **长期记忆**：与所返回事件相关的跨事件总结 | 展示"用户的长期画像 / 习惯" |

> `include_facts` 和 `include_long_memories` 的内容由 SAGE 产出，**建库（`build()`）前为空**；`include_observations` / `include_raw` 建库前也能读到。开启越多层，查询开销越大、返回体越大，按展示需要选。

**返回结构**：

```python
{
    "events": [
        {
            "event_id": "E-...",
            "time_span": ["2026-05-03T01:06:00+00:00", "..."],
            "state": "transaction",          # DRED 事件类型
            "entities": ["铁路12306", "G15"],
            "anchors": {"order_id": ["E2026050301"]},
            "confidence": 0.92,
            "title": "预订北京到上海高铁",       # SAGE 产出，建库前为 None
            "summary": "...",
            "facts": [{"fact_id": "...", "text": "...", "support": ["O-..."]}],  # include_facts
            "observations": [                # include_observations=True 才有
                {"observation_id": "O-...", "timestamp": "...", "source": "系统", "anchors": {...}}
            ],
            "raw_records": [                 # include_raw=True 才有
                {"raw_id": "R-...", "record": {"type": "sms", "content": "...", ...}}
            ]
        }
    ],
    "long_memories": [...],                  # include_long_memories=True 才有
    "total": 5                               # 符合条件的事件总数（limit 截断前）
}
```

示例：

```python
# 读某订单相关的全部事件，并带出原始提交记录
res = mem.read(anchor="E2026050301", include_raw=True)
for ev in res["events"]:
    print(ev["title"], ev["time_span"])
    for rr in ev["raw_records"]:
        print("  原始提交：", rr["record"])

# 读某一天的记忆（时间窗）
res = mem.read(start="2026-05-03", end="2026-05-03")
print(f"共 {res['total']} 个事件")
```

#### `is_started` / `is_built` / `count()` / `close()`
建库是否已开始（开始后不能再 `add`）/ 是否已建完 / 各类计数 / 关闭连接。

### 10. 作为 agent 工具接入

如果你的 agent 运行在支持 MCP 的客户端里（Claude Desktop、Cursor、Claude Code 等），直接用第 1 节的 **MCP 方式**即可，无需自己封装：写入、建库、查询都有对应工具。下面是在自己的 agent 框架里手动封装的写法。

把两个方法包成 agent 的工具即可。建库在会话外离线完成，查询在会话内实时调用：

```python
mem = HyperMemory("./user_memory.db", config=config)  # 已 build 过的库

def search(query: str) -> str:
    """检索与问题相关的用户长期记忆，返回可直接放入上下文的文本。"""
    return mem.search(query)["context"]

# 注册 search 为你的 agent 工具；需要新增记忆时，用一个离线任务跑
# add + build 重建库，再热切换。
```

> 注意：`build` 会调用 LLM 做事件消化，是一次相对重的离线过程，不适合放在实时对话里。实时路径只用 `search`。

---

## 二、数据类型与字段总表

每条记录外层只有 `type`、`timestamp`、`data`。完整说明与每种类型的示例见 [README.md](../README.md) 第 4 章。

### 外层字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `type` | 是 | 下表 10 种之一 |
| `timestamp` | 是 | 这件事在设备上发生的时间：整数 Unix 毫秒，或 ISO 8601 字符串（不带时区按本机时区）；入库统一存为毫秒 |
| `data` | 是 | 通用字段 + 该类型的专属字段 |

外层不能有其他字段。归属用户（`user`）是 `add` 的参数，不写进记录。

### `data` 里的通用字段（所有类型都可带，都可选）

| 字段 | 说明 |
| --- | --- |
| `from_app` | 来源应用 |
| `record_id` | 调用方的记录 ID，同库不可重复 |
| `anchors` | 强标识：`order_id`、`transaction_id`、`tracking_id`、`reservation_id`、`phone`、`email`、`url`（DRED 优先按它匹配，跨 app 关联靠它） |
| `entities` | 实体：`contact`、`merchant`、`product`、`place`、`organizer`（二级匹配键） |
| `latitude` / `longitude` | 事情发生时设备所在的位置，成对出现 |

### `data` 里的专属字段（按类型）

| type | 中文 | 必填 | 选填 |
| --- | --- | --- | --- |
| `screen` | 屏幕内容 | `action_description`（VLM 描述的用户动作） | `typed_text`（用户输入的文字）、`title`、`ocr`、`image_path` |
| `gallery` | 相册 | `image_description`（VLM 对画面的描述） | `ocr`、`image_path`、`captured_at`（拍摄时间）、`location`（拍摄地点） |
| `sms` | 短信 | `content` | `sender`（发送方号码）、`sender_name` |
| `push` | 推送通知 | `action`（`receive` / `open`），以及 `title` 或 `content`（至少一个） | — |
| `calendar` | 日历 | `title`（标题）或 `content`（备注），至少一个 | `start_time`、`end_time`、`remind_time`、`calendar_type` |
| `clipboard` | 剪贴板 | `content` | `action`（`copy` / `cut`） |
| `position` | 定位 | `latitude` + `longitude`，或 `ssid` | `location`、`accuracy` |
| `chat` | 对话（仅 user↔agent） | `query` | `answer`、`session_id` |
| `recording` | 录音转文本 | `transcript`（列表，每段 `text`、`offset_ms`、`speaker` 都要写） | `audio_path` |
| `note` | 笔记 | `action`（`create` / `edit` / `delete`），以及 `title` 或 `content`（至少一个） | — |

### 几类需要说明的字段

**时间与位置**：`timestamp` 和经纬度说的是**这条记录在设备上发生的时间和位置**，决定时间线顺序；`gallery` 的 `captured_at` / `location`、`calendar` 的 `start_time` / `end_time` / `remind_time` 是内容本身的时间或地点，不参与排序，只写进可检索的文字。

**screen 与 gallery**：`screen` 是"用户在做什么"，`gallery` 是"用户看到或得到了什么图"。两者的描述都要由上游 VLM 生成。

**push / clipboard / note 的 `action`**：系统会把动作写进可检索的文字，如"用户打开了这条通知""用户编辑了这条笔记"。

**note**：同一篇笔记一次编辑只提交最后一版，由接入方在 `add` 之前合并。

**position**：系统合成一句可检索的文字（如 `位置：上海人民广场`、`连接 WiFi：Home_5G`），同时保留原始经纬度供定位提取。

---

## 三、在 PMM-Bench 上测评

评测代码全部在 `eval/`，一条命令对任意一组系统和用户建库并答题。完整说明见 [eval/docs/EVALUATION.md](../eval/docs/EVALUATION.md)，这里给最短路径。

### 1. 安装

```bash
cd eval
pip install -e .                                        # PMM-Bench 工具包
pip install -e ..                                       # 被测的 OS-Memory 核心（仓库根目录）
```

runner 会自动通过上一级的仓库根目录找到核心包；如需指定别处，设 `PMM_HYPERMEM_SOURCE`。

### 2. 配置模型网关

```bash
export OPENAI_BASE_URL=https://your-gateway/v1
export OPENAI_API_KEY=...
export PMM_EMBEDDING_URL=https://your-embedder/v1/embeddings
export PMM_EMBEDDING_MODEL=...
export PMM_EMBEDDING_API_KEY=...        # 可选
```

### 3. 跑评测

```bash
cd eval
python -m pmm_bench users --data /path/to/PMM-Bench           # 列用户
python -m pmm_bench run   --data /path/to/PMM-Bench --systems hyperos-mem --users U01 --out runs/one
python -m pmm_bench run   --data /path/to/PMM-Bench --systems hyperos-mem,mem0,simplemem --users all --jobs 4 --out runs/all
python -m pmm_bench report --out runs/all                     # 结果表
```

每个 (system, user) 对把断点续跑的逐题记录写到 `runs/<name>/results/<system>/<user>/`；重跑同一命令会跳过已完成的工作。`report.md` 给出分类准确率、Judge（五类均值）和 F1。

OS-Memory 始终通过 `pmm_bench.final` 运行（DRED + 完整 ToolSAGE）。各 baseline 在各自的 Python 环境里跑（`PMM_PYTHON_MEM0=/path/to/env/bin/python` 等）。

### 4. 离线自测（无需真实模型）

仓库自带一个合成样例和 mock 服务，可在完全离线下验证整条流水线：

```bash
cd eval
python tests/mock_openai_server.py --port 8765 --dimension 64 &
export OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_API_KEY=offline \
       PMM_EMBEDDING_URL=http://127.0.0.1:8765/v1/embeddings PMM_EMBEDDING_MODEL=offline
echo '{"embedding": {"dimension": 64}}' > /tmp/offline.json
python -m pmm_bench run --data examples/pmm_sample --systems hyperos-mem,mock --users all \
       --out runs/offline --config /tmp/offline.json
```

mock 服务返回占位内容，分数无意义，只用于证明流水线能跑通。

### 5. 验证一次完整构建

```bash
cd eval
python -m pmm_bench.final check --run runs/one/runs/hyperos-mem/U01/build
```

读取只读校验一次已完成的构建，不调用任何模型。

---

## 常见问题

**`import hypermem` 会不会很重？** 不会。裸导入只加载数据层，不碰 engine/SAGE。只有在你真正 `from hypermem import HyperMemory` 时才惰性加载在线模块。

**为什么必须 `build` 之后才能查询？** OS-Memory 的查询依赖 SAGE 产出的事件/事实/长期记忆图，而 DRED 的语料中心在建库时冻结。这是 DRED+SAGE 方法的固有设计，没有跳过建图的检索模式。

**能逐条在线更新吗？** 当前为成批建图：一个库建一次。要加入新记忆，新开一个库重建，再热切换。

**建库途中模型服务挂了怎么办？** `build()` 会抛出 `ModelServiceError` 并停下，已完成的部分和各任务的重试次数都保留。修好服务后对同一个库再调一次 `build()` 即可续建，不需要新开库。
