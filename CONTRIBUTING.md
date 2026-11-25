# 贡献指南

感谢你愿意改进 OS-Memory。提交前请花几分钟读完本文。

## 开发环境

- Linux 或 macOS（Windows 请使用 WSL），Python 3.10 及以上。
- 仓库分两部分：根目录是核心库 `hypermem`，`eval/` 是评测工具 PMM-Bench。

```bash
git clone https://github.com/XiaoMi/os-memory.git
cd os-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"            # 核心库 + MCP
pip install -e eval                # 需要改评测工具时再装
```

## 运行测试

```bash
python -m unittest discover -s tests -t tests      # 核心库，全部通过后再提交
```

改动评测工具时，再按 README 6.7 节用 mock 服务跑一次离线评测。

## 提交流程

1. 先开 issue 说明要解决的问题，较大的改动先讨论方案。
2. 从默认分支新建分支，一个 PR 只做一件事。
3. 提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 格式，例如 `fix: ...`、`feat: ...`、`docs: ...`；破坏性变更在类型后加 `!`。
4. 修改了行为或接口时，同步更新 README 与 `CHANGELOG.md` 的“未发布”一节，并补充测试。

需要额外评审的改动：DRED/SAGE 方法代码（`hypermem/online/`）、公开 API 与返回结构、SQLite schema、评测的 prompt 与评分代码、`eval/systems/` 下的第三方代码。

## 禁止提交的内容

- 真实用户数据（包括脱敏后的样本）、PMM-Bench 私有数据；
- API 密钥、私有模型服务地址、内网链接、本机绝对路径；
- 数据库、日志、评测输出、模型权重等运行产物。

示例和测试只使用明确标注为虚构的合成数据。

## 第三方代码

引入或更新第三方代码时，在 PR 中写明上游地址、确切的 commit、许可证，并保留原有的版权与许可证文件；修改过的文件需注明修改。

## 许可

提交贡献即表示你同意以 [Apache License 2.0](LICENSE) 授权你的贡献，并确认你有权这样做。

## 安全问题

请按 [SECURITY.md](SECURITY.md) 私下报告，不要提交公开 issue。
