#!/usr/bin/env python
# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

# =============================================================================
# HyperMemory Eval 统一运行入口 (跨平台 / CPU / 无需 GPU)
# -----------------------------------------------------------------------------
# 这个脚本做的事:
#   1. 读 .env 拿到 LLM 网关的 key 和地址
#   2. 读对应系统的配置文件 configs/<system>.json (每个系统一个, 完整独立)
#   3. 从 config 的 _data 拿本次数据路径, 占位符 {ROOT}/{OUT_DIR} 替换成实际路径, 设好环境变量
#   4. 调 run_benchmark 跑该系统在其数据集上的评测, 出指标
#
# 统一入口: 所有系统 (LoCoMo 对话类 mem0/amem/lightmem/simplemem/plugmem/lightrag,
#          以及其他事件类数据集) 都用这一个 run.py, 用法完全一致。
#
# 用法 (先激活对应系统的 conda 环境, 再运行本脚本):
#     conda activate mb_lightmem
#     python run.py lightmem smoke      # smoke = 快速小样本验证
#     python run.py lightmem full       # full  = 全量正式评测
#
#   第 1 个参数 = 系统名 (即 configs/<系统>.json): mem0 / amem / lightmem / simplemem /
#                plugmem / lightrag
#   第 2 个参数 = smoke 或 full (不填默认 smoke)
#   第 3 个参数 = (可选) 临时覆盖数据路径, 不改 config 试跑另一个数据文件
#
# 断点续跑 (只补跑上次失败/未跑的样本, 已成功样本从上次目录结转, 不重跑):
#     python run.py lightrag full --resume runs/lightrag_locomo_full_<上次时间戳>
#     python run.py lightrag full --resume <上次目录> --retry 2   # 失败样本重试 2 轮
#   本次仍是新时间戳目录, 但会自包含"迄今所有成功样本", metrics 从并集重算。
#   失败样本列在本次 report.md / sample_status.json, 下次 --resume 指向本目录即可继续补。
#
# 配置怎么改: 每个系统的全部配置都在 configs/<system>.json 里 (共用项也各写一遍,
#            所以每个系统能独立设不同的 LLM/嵌入/参数/数据)。改配置只改那个 json, 不动本脚本。
#
# 配置文件里的约定:
#   - 占位符 {ROOT}    -> 本仓库根目录 (本文件所在目录)。路径都写相对, 保证可移植。
#   - 占位符 {OUT_DIR} -> 本次运行的输出目录 (带时间戳)。用于向量库等运行期存储。
#   - api_key_env / base_url_env: 只写环境变量名, 真实 key 从 .env 读, 不写进 json (防泄露)。
#   - 以 _ 开头的字段是给 run.py 用的元信息, 不会传给记忆系统适配器:
#       _dataset      数据集名 (决定用哪个数据集 loader + 输出目录的 run_tag)
#       _data         数据路径, 按档位 {"smoke": "...", "full": "..."} (用 {ROOT} 占位)
#       _sample_size  采样量, 按档位 {"smoke": N, "full": null}; null=全部
#       _pythonpath   该系统源码目录 (加入 import 搜索路径)
#       _extra_env    运行前要额外设置的环境变量 (如 simplemem 的 EMBEDDING_MODEL、
#                     某些系统的额外运行环境变量等)
#
# 依赖: 每个系统一个独立 conda 环境, 依赖清单见 requirements/<system>.txt
# 结果: runs/<system>_<数据集tag>_<smoke|full>_<时间戳>/  (metrics.json / report.md / run.log)
#       数据集tag 来自数据集 loader 的 run_tag (例如 locomo=locomo)
# =============================================================================
import os as _os
# --- 环境前置: 必须在 torch / chromadb 等被导入之前执行 (对不需要的系统是无害 no-op) ---
# 1) 强制 CPU-only: 某些环境 (CloudML orion vGPU hook) 在首次 torch.cuda.is_available()
#    时会杀进程, 即便我们只想用 CPU。隐藏 CUDA 设备并 patch is_available -> False。
#    对本就在 CPU 上跑的系统 (sentence-transformers 等) 无影响。
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
try:
    import torch as _torch  # type: ignore

    _torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    if hasattr(_torch.cuda, "device_count"):
        _torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
except Exception:
    pass
# 2) chromadb 要 sqlite>=3.35; 部分平台系统 sqlite 太旧。若装了 pysqlite3 就换上,
#    没装则 no-op (Windows / 新系统的 sqlite 通常已达标, 不受影响)。
try:  # noqa: SIM105
    import pysqlite3  # type: ignore
    import sys as _sys

    _sys.modules["sqlite3"] = pysqlite3
    _sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except ImportError:
    pass

import os
import sys
import json
import time
from pathlib import Path

# ROOT = 本脚本所在目录 (即 HyperMemory Eval 根)。所有相对路径都基于它, 保证可移植。
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def load_env():
    """把根目录 .env 里的 KEY=VALUE 读进环境变量，并覆盖父进程同名变量。
    .env 至少要有 OPENAI_API_KEY 和 OPENAI_BASE_URL, 供 LLM 网关调用。"""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 跳过空行和注释行, 只处理 K=V
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def _subst(obj, mapping):
    """递归地把配置里字符串的占位符 ({ROOT}/{OUT_DIR}) 替换成实际路径。"""
    if isinstance(obj, str):
        for key, val in mapping.items():
            obj = obj.replace(key, val)
        return obj
    if isinstance(obj, dict):
        return {k: _subst(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, mapping) for v in obj]
    return obj


def _peek_meta(system):
    """在建输出目录之前, 从 configs/<system>.json 读取数据相关的元字段。

    返回 (dataset, data_map, sample_map):
      - dataset    : _dataset (默认 locomo), 决定用哪个数据集 loader。
      - data_map   : _data, 形如 {"smoke": "...", "full": "..."} 的数据路径 (可空);
                     {ROOT} 占位符在此替换成仓库根。数据路径写在 config, 不走命令行。
      - sample_map : _sample_size, 形如 {"smoke": 200, "full": null} 的采样量 (可空)。
    只读不替换 {OUT_DIR} (此时输出目录还没建); 只把 {ROOT} 替进 _data。
    """
    cfg_file = ROOT / "configs" / f"{system}.json"
    if not cfg_file.exists():
        sys.exit(f"[FATAL] 配置文件不存在: {cfg_file} "
                 f"(系统名应为 configs/ 下某个 <系统>.json)")
    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    root_str = str(ROOT).replace("\\", "/")
    data_map = raw.get("_data") or {}
    data_map = {k: str(v).replace("{ROOT}", root_str) for k, v in data_map.items()}
    return raw.get("_dataset", "locomo"), data_map, (raw.get("_sample_size") or {})


def load_config(system, out_dir):
    """读 configs/<system>.json, 替换占位符, 拆出元字段。

    返回 (config, dataset, pythonpath, extra_env, adapter_system):
      - config         : 传给适配器的系统配置 (已去掉 _ 开头的元字段)
      - dataset        : 数据集名
      - pythonpath     : 该系统源码目录
      - extra_env      : 运行前要设的额外环境变量
      - adapter_system : 用哪个已注册适配器 (元字段 _system, 缺省=位置参数 system)。
                         用于"同一适配器 + 不同数据集/配置"的配置变体: 例如
                         配置变体可以通过 "_system" 复用已注册的适配器，
                         无需新建 systems/ 目录。
    """
    cfg_file = ROOT / "configs" / f"{system}.json"
    if not cfg_file.exists():
        sys.exit(f"[FATAL] 配置文件不存在: {cfg_file} "
                 f"(系统名应为 mem0 / amem / lightmem / simplemem 之一)")

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    # 占位符 -> 实际路径 (统一用正斜杠, 跨平台安全)
    mapping = {
        "{ROOT}": str(ROOT).replace("\\", "/"),
        "{OUT_DIR}": str(out_dir).replace("\\", "/"),
    }
    raw = _subst(raw, mapping)

    dataset = raw.get("_dataset", "locomo")
    pythonpath = raw.get("_pythonpath")
    extra_env = dict(raw.get("_extra_env", {}))
    adapter_system = raw.get("_system", system)
    # 去掉所有 _ 开头的元字段, 剩下的才是真正传给适配器的配置
    config = {k: v for k, v in raw.items() if not k.startswith("_")}
    return config, dataset, pythonpath, extra_env, adapter_system


class Tee:
    """把 print 的内容同时写到终端和日志文件 run.log。"""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
    def flush(self):
        self.stdout.flush()
        self.f.flush()


def main():
    # ---- 1. 解析参数: 系统名 + smoke/full [+ --resume <目录>] [+ --retry <N>] ----
    argv = sys.argv[1:]
    # 先抽出可选的 --resume / --retry (带值), 余下按位置解析系统名和 smoke/full。
    resume_from = None
    retry_rounds = 1
    positional = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--resume", "--resume-from", "--resume_from"):
            resume_from = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
        elif tok in ("--retry", "--retry-rounds"):
            retry_rounds = int(argv[i + 1]) if i + 1 < len(argv) else 1
            i += 2
        else:
            positional.append(tok)
            i += 1
    if not positional:
        sys.exit("用法: python run.py <系统> [smoke|full] [数据路径] [--resume <上次run目录>] [--retry <N>]")
    system = positional[0].lower()
    subcmd = positional[1].lower() if len(positional) > 1 else "smoke"
    # 第 3 个位置参数 = 可选数据路径 (给需要临时覆盖数据路径的系统使用)。
    data_arg = positional[2] if len(positional) > 2 else None

    # ---- 2. 读 .env, 校验 LLM 网关凭据 ----
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[FATAL] OPENAI_API_KEY 缺失 (请检查 .env)")
    if not os.environ.get("OPENAI_BASE_URL"):
        sys.exit("[FATAL] OPENAI_BASE_URL 缺失 (请检查 .env)")

    # ---- 3. 离线模式 (不联网下模型) ----
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if subcmd not in ("smoke", "full"):
        sys.exit("子命令只能是 smoke 或 full")

    # ---- 4. 数据来源: 完全由 config 声明 (每个系统一致) ----
    # 数据路径写在 configs/<系统>.json 的 _data[subcmd], 采样量写在 _sample_size[subcmd]。
    # 所有系统 (LoCoMo 与 os_events) 都走这一套, run.py 不写死任何数据集路径。
    # 命令行第 3 参数仅作临时覆盖 (换个 db 试跑, 不改 config)。数据集名 _dataset 决定用
    # 哪个 loader 解析文件, 以及 run_tag (输出目录名)。
    dataset, data_map, sample_map = _peek_meta(system)
    from hypereval.registry import create_dataset
    try:
        loader = create_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[FATAL] 未知数据集 {dataset!r} (系统 {system} 的 _dataset): {exc}")

    run_tag = getattr(loader, "run_tag", dataset)
    # 数据路径: 命令行覆盖 > config._data[subcmd]。
    data_str = data_arg or data_map.get(subcmd)
    if not data_str:
        sys.exit(
            f"[FATAL] configs/{system}.json 未配置数据路径: 请在 "
            f'_data["{subcmd}"] 写上数据文件路径 (用 {{ROOT}} 占位)。'
        )
    data = Path(data_str).expanduser().resolve()
    if not data.exists():
        sys.exit(f"[FATAL] 数据缺失: {data}")
    # 采样量: config._sample_size[subcmd] (缺省则 smoke=1样本 / full=全部)。
    sample = sample_map[subcmd] if subcmd in sample_map else (1 if subcmd == "smoke" else None)

    # ---- 5. 建本次运行的输出目录 (带时间戳, 目录名用数据集的 run_tag) ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runs" / f"{system}_{run_tag}_{subcmd}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 6. 读该系统的配置文件, 应用它要求的额外环境变量 ----
    config, dataset, pythonpath, extra_env, adapter_system = load_config(system, out_dir)
    if pythonpath:
        sys.path.insert(0, pythonpath)          # 把该系统源码目录加到 import 搜索路径
        os.environ["PYTHONPATH"] = pythonpath
    for k, v in extra_env.items():
        os.environ[k] = v

    # 把最终 config (占位符已替换) 落盘, 适配器通过 system_config 读它
    cfg_path = out_dir / f"{system}.system.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. 终端输出同时写 run.log ----
    tee = Tee(out_dir / "run.log")
    sys.stdout = tee
    sys.stderr = tee
    print(f"===== [run.py] {system}-{dataset} start {time.strftime('%F %T')} subcmd={subcmd} =====")
    print(f"[cfg] config=configs/{system}.json  data={data}  sample={sample}")
    if adapter_system != system:
        print(f"[cfg] adapter=create_adapter({adapter_system!r})  (via config _system)")
    print(f"[cfg] out={out_dir}")
    if resume_from:
        print(f"[cfg] resume_from={resume_from}  retry_rounds={retry_rounds}")
    print("########## RUN ##########")

    # ---- 8. 跑评测 (延迟到这里再 import, 确保 PYTHONPATH/env 已设好) ----
    from hypereval.runner import run_benchmark
    run_benchmark(
        # system_name 决定用哪个适配器 (create_adapter); 配置变体用 _system 指向已注册适配器,
        # 缺省即位置参数名。输出目录/配置路径仍用位置参数 system (见上), 目录名清晰可辨。
        system_name=adapter_system,
        dataset_name=dataset,
        sample_size=sample,
        output_dir=str(out_dir),
        top_k=int(os.environ.get("TOP_K", "10")),
        # system_context 传给适配器: system_config 指向刚生成的配置; dataset 供适配器按数据集分派;
        # run_dir 供需要运行期存储目录的适配器使用
        system_context={"system_config": str(cfg_path), "dataset": dataset, "run_dir": str(out_dir)},
        data_path=str(data),
        # 断点续跑: 结转上次目录的已成功样本, 只补跑失败/未跑的; retry 为失败样本重试轮数
        resume_from=resume_from,
        retry_rounds=retry_rounds,
    )

    # ---- 9. 打印结果 ----
    print("########## SUMMARY ##########")
    metrics = out_dir / "metrics.json"
    report = out_dir / "report.md"
    if metrics.exists():
        print(metrics.read_text(encoding="utf-8"))
    if report.exists():
        print(report.read_text(encoding="utf-8"))
    print(f"===== [run.py] {system}-{dataset} finish {time.strftime('%F %T')} out={out_dir} =====")


if __name__ == "__main__":
    main()
