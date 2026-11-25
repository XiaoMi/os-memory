# 更新日志

本项目的重要变更记录在此文件中。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

## [1.0.0]

首个公开版本，方法为 DRED + SAGE。

### 新增

- Python API：`HyperMemory.add / build / search / read / count / close`，以及 `is_started`、`is_built`、`format_memories`、`ModelServiceError`。
- MCP 服务器（`hyperos-mem-mcp`）：`add / build / search / read / status` 工具，`build` 在后台执行；只配置嵌入服务即可检索和浏览。
- 统一记录格式 `{type, timestamp, data}`，支持 10 类手机数据。
- 模型服务故障后可续建，续建时可修改服务地址、密钥等配置。
- 评测工具 `eval/`，入口 `python -m pmm_bench`，附合成样例与离线 mock 服务；对比系统用 `python eval/scripts/fetch_baselines.py` 按固定的上游 commit 拉取并打补丁，默认关闭第三方使用统计上报。
- 新建的数据库文件权限为 `0600`。
- 文档：README（中英文）、数据与隐私说明（README 1.5 节）、`SECURITY.md`、`CONTRIBUTING.md`。

[未发布]: https://github.com/XiaoMi/os-memory/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/XiaoMi/os-memory/releases/tag/v1.0.0
