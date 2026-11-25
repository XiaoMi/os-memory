# Third-party code

The repository does not include the source code of the baseline systems. Each
`systems/<name>/manifest.json` names the upstream repository and the exact
commit that was evaluated; `python scripts/fetch_baselines.py` clones it into
`systems/<name>/` and applies our patch, if any. The fetched copies are ignored
by git and keep their original licenses.

| System | Project | License | Upstream commit | Our patch |
| --- | --- | --- | --- | --- |
| `amem` | [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) | MIT | `f303dfc71e07bdc787f4bc135d4cea328ae30e99` | none |
| `lightmem` | [LightMem](https://github.com/zjunlp/LightMem) | MIT | `579ee76cede1cf7ee298716c9a8f1645a5b385bb` | `systems/lightmem/patches/pmm-bench.patch` |
| `lightrag` | [LightRAG](https://github.com/HKUDS/LightRAG) | MIT | `e66173cec29e33d093bbd89427861068c2268621` | `systems/lightrag/patches/pmm-bench.patch` |
| `mem0` | [Mem0](https://github.com/mem0ai/mem0) | Apache-2.0 | `b357a5a1b03c299ec8229c268e63cfac0f7c6566` | none |
| `plugmem` | [PlugMem](https://github.com/TIMAN-group/PlugMem) | Apache-2.0 | `3b2ce75257d40bca8fac3f78e54e22ea41d92529` | `systems/plugmem/patches/pmm-bench.patch` |
| `simplemem` | [SimpleMem](https://github.com/aiming-lab/SimpleMem) | MIT | `db80b6a7c591e0ea730a058e9f5fc4eb06572299` | `systems/simplemem/patches/pmm-bench.patch` |

Upstream repositories contain further components under their own licenses,
for example VLM2Vec (Apache-2.0) inside LightMem and OmniSimpleMem
(Apache-2.0) inside SimpleMem. Check each upstream repository for the full
terms before redistributing a fetched copy.

## Patches

The patches under `systems/<name>/patches/` are our changes to the upstream
files they list. Each patch is distributed under the license of the project it
modifies, and its header names the upstream commit and every modified file.
In summary:

- LightMem: Pydantic v1/v2 compatibility, tolerate responses without `usage`,
  extraction concurrency set by `LIGHTMEM_EXTRACT_MAX_WORKERS` (default 1).
- LightRAG: retry atomic file replacement on Windows.
- PlugMem: run the reward and state LLM calls in parallel, batch embedding
  requests (`PLUGMEM_EMBEDDING_BATCH_SIZE`).
- SimpleMem: UTF-8 file reads; fall back to the top-ranked entries when no
  entry scores above zero; recall with question images in Omni-SimpleMem.

Read the patch files for the exact changes.

## Python dependencies

`requirements/<name>.txt` lists the packages each baseline environment needs.
They are installed from PyPI by the user and are not distributed with this
repository.
