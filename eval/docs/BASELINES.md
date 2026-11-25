# Baselines

| System | Group in the paper | Status here | Upstream |
| --- | --- | --- | --- |
| Long context (128K) | Reference | integration pending | – |
| Flat dense retrieval | Reference | integration pending | – |
| SimpleMem | Compact-entry memory | adapter ready | [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem) @ `db80b6a` |
| LightMem | Compact-entry memory | adapter ready | [zjunlp/LightMem](https://github.com/zjunlp/LightMem) |
| Mem0 | Compact-entry memory | adapter ready | [mem0ai/mem0](https://github.com/mem0ai/mem0) @ `b357a5a` |
| A-Mem | Relational memory | adapter ready | [WujiangXu/A-mem-sys](https://github.com/WujiangXu/A-mem-sys) @ `f303dfc` |
| StructMem | Relational memory | integration pending | – |
| CompassMem | Relational memory | integration pending | – |
| PlugMem | not in the main table | adapter ready | [TIMAN-group/PlugMem](https://github.com/TIMAN-group/PlugMem) @ `3b2ce75` |

"Integration pending" systems are registered in `configs/pmm/default.json` with `"kind": "pending"`; the runner lists them and skips them. Their PMM-Bench runners will be added to this repository.

## Adding a system

1. Implement `hypereval.adapters.base.MemoryAdapter`: `initialize`, `reset`, `ingest`, `finalize_ingest`, `retrieve(query, user_id, session_id, top_k) -> list[MemoryRecord]`.
2. Register it in `hypereval/registry.py` and put its source under `systems/<name>/` with a `manifest.json`.
3. Add its native construction config as `configs/pmm/<name>.json`. Placeholders `{ROOT}`, `{BUILD_DIR}`, `{CONSTRUCTION_MODEL}` and `{EMBED_DIM}` are filled at run time; use the encoder name `api-embedder` to get the shared embedding endpoint.
4. Add `"<name>": {"kind": "adapter", "config": "configs/pmm/<name>.json"}` under `systems` in `configs/pmm/default.json`.
5. Check the wiring offline: `PMM_MOCK_LLM=1 PMM_MOCK_EMBEDDING=1 python -m pmm_bench run --data examples/pmm_sample --systems <name> --users U01 --out runs/check`.

The shared answer and judge prompts (`pmm_bench/prompts.py`) and the scoring code are identical for every system, OS-Memory included.
