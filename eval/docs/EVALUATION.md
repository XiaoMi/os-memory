# Evaluating on PMM-Bench

`python -m pmm_bench` is the single entry point. It runs any set of systems on one user, a list of users, or all users, then aggregates the results.

## 1. Data layout

Point `--data` at a PMM-Bench package:

```text
PMM-Bench/
├── dataset_manifest.json        # per user: cleaned file, record count, sha256, months
├── checksums.json               # sha256 + bytes of every file
└── users/<user_id>/
    ├── qa.json                  # questions, reference answers, evidence IDs, category
    ├── observations.jsonl       # raw stream, one complete original record per line
    └── data/cleaned_observations.jsonl   # cleaned stream (default build input)
```

Every directory under `users/` that contains `qa.json` is a user. `examples/pmm_sample/` is a synthetic package with this exact layout; `examples/make_pmm_sample.py` regenerates it.

Construction reads observations only. Reference answers and evidence are used for scoring.

### A small package for a first run

`mini` cuts one user (or several) down to a quick run: questions unchanged, every observation cited as evidence kept, plus a fixed number of randomly sampled other observations as noise.

```bash
python -m pmm_bench mini --data PMM-Bench --users <user_id> --out PMM-mini --noise 300 --alias demo
python -m pmm_bench run  --data PMM-mini --systems hyperos-mem --users demo --out runs/mini
```

Scores on a mini package are far higher than on the full history and are not comparable to the paper; use it to check the setup. A package cut from real users contains their records; keep it private.

## 2. Environment

```bash
export OPENAI_BASE_URL=https://your-gateway/v1          # chat completions, OpenAI-compatible
export OPENAI_API_KEY=...
export PMM_EMBEDDING_URL=https://your-embedder/v1/embeddings
export PMM_EMBEDDING_MODEL=...
export PMM_EMBEDDING_API_KEY=...                        # optional; sent by every system, OS-Memory included
```

All systems share this one embedding endpoint. Baselines that load an encoder through sentence-transformers receive a thin client of it, so no local embedding model is needed.

OS-Memory needs Python ≥ 3.10 and NumPy. Install the evaluation toolkit from
this `eval/` directory and the memory system from the repository root, one
level up:

```bash
pip install -e .                                  # from eval/ (PMM-Bench toolkit)
pip install -e ..                                 # OS-Memory core (repository root)
```

The runner locates the core automatically in the repository root; set
`PMM_HYPERMEM_SOURCE` to override that path.

Baseline source code is not part of the repository. Fetch each baseline at the commit that was evaluated, with our patches applied (needs git and network access to GitHub):

```bash
python scripts/fetch_baselines.py                 # all baselines; --systems mem0,amem for a subset
```

See `THIRD_PARTY.md` for the upstream projects, commits and licenses.

Each baseline has conflicting dependencies, so give each its own environment and tell the runner where its interpreter is:

```bash
conda create -n pmm_mem0 python=3.10 -y
conda run -n pmm_mem0 pip install -r requirements/mem0.txt
export PMM_PYTHON_MEM0=$(conda run -n pmm_mem0 which python)
```

The variable name is `PMM_PYTHON_<SYSTEM>` (upper case, `-` becomes `_`). `requirements/<system>.txt` lists each environment. LightMem additionally needs the LLMLingua-2 model under `model/` (see `configs/pmm/lightmem.json`).

## 3. Run

```bash
# one user
python -m pmm_bench run --data PMM-Bench --systems hyperos-mem --users U01 --out runs/u01

# several users and systems, four pairs at a time
python -m pmm_bench run --data PMM-Bench --systems hyperos-mem,mem0,simplemem,lightmem,amem \
       --users U01,U02,U03 --jobs 4 --out runs/subset

# everything
python -m pmm_bench run --data PMM-Bench --systems all --users all --jobs 4 --out runs/full
```

| Option | Meaning |
| --- | --- |
| `--systems` | comma list from `python -m pmm_bench systems`, or `all` |
| `--users` | comma list from `python -m pmm_bench users --data …`, or `all` |
| `--jobs` | (system, user) pairs run in parallel |
| `--input` | `cleaned` (default, used in the paper) or `raw` |
| `--config` | JSON merged over `configs/pmm/default.json`: models, top-k, context size, embedding dimension |
| `--limit N` | build from the first N observations only; a pipeline check, never an official score. OS-Memory then builds without answering, into `runs/hyperos-mem/<user>/smoke/`, and has no row in the report |
| `--force` | rerun pairs that already finished |

Defaults follow the paper: GPT-4.1-mini builds memory and answers, GPT-4o judges, top-30 retrieval, 48,000-character answer context. Change model names in an override file if your gateway uses different identifiers:

```json
{"models": {"construction": "gpt-4.1-mini", "answer": "gpt-4.1-mini", "judge": "gpt-4o"}}
```

## 4. What each system does

**OS-Memory** runs through `pmm_bench.final` in the stages `prepare`, `dred`, `sage`, `dashboard`, `evaluate`. DRED attributes observations to events online without a model; SAGE then consolidates event boundaries, runs full ToolSAGE (SplitEvent, SummarizeEvent, ExtractFacts) and distils LongMemory into the grounded Event / Anchor / Fact / LongMemory graph, which HyperOS graph retrieval searches. Model tasks that still fail after their retries are skipped and listed in `sage_report.json` and the evaluation `contract.json`. A finished run contains `final_run_contract.json`; verify it with

```bash
python -m pmm_bench.final check --run runs/u01/runs/hyperos-mem/U01/build
```

(`pmm_bench.final` is the benchmark front-end; to embed the memory system in
your own agent use the `hypermem.HyperMemory` library instead — see the root
`README.md`.)

**Baselines** run through `pmm_bench.worker` in their own environment: each ingests the user's observations in time order through its adapter, retrieves the top 30 memories per question, and passes them to the shared answer and judge prompts.

## 5. Outputs

```text
runs/<name>/
├── results/<system>/<user>/<hash>.json   # one record per question: retrieval, context, prediction, judge, score, f1
├── results/<system>/<user>/summary.json
├── runs/hyperos-mem/<user>/build/        # OS-Memory memory database and audits (smoke/ with --limit)
├── runs/hyperos-mem/<user>/hypermem_dashboard.html   # OS-Memory memory graph
├── runs/hyperos-mem/<user>/protocol.resolved.json    # models and services this build used
├── stores/<baseline>/<user>/             # baseline stores
├── logs/<system>/<user>.log
├── report.json
└── report.md
```

Per-type accuracy pools all completed questions of the selected users. **Judge** is the unweighted mean of the five per-type accuracies (Single, Cross, Temp, Track, Habit); it is shown as `–` when the selected questions miss one of the five types. **F1** is the mean Chinese character/word unigram F1 against the reference answer. The report gives all three as percentages. Failed questions are counted separately and excluded from the means; rerun the same command to retry them.

`python -m pmm_bench report --out runs/full --users U01,U02` re-aggregates any subset without rerunning.

## 6. Resuming

Per-question records are written as soon as each step returns, so an interrupted run resumes where it stopped. OS-Memory additionally resumes its construction stages from its database journal. Baseline stores live in memory during a run; if a baseline process stops during construction, that user's store is rebuilt on the next run, while already answered questions are kept.

Model service settings may change before a rerun: a corrected `OPENAI_BASE_URL` or `PMM_EMBEDDING_URL`, a new key, or other construction, answer or judge models in the override file. OS-Memory lists what changed, continues with the new settings, reports the pair as `ok (settings changed since the last run, …)` and records the change in `settings_revisions.json` in its build directory. The embedding model and dimension cannot change within a run, because the stored vectors could not be reused: OS-Memory stops with `Embedding model or dimension changed …`, and the build must be redone under a new `--out`.
