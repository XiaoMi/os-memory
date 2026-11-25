# OS-Memory version statement

OS-Memory 1.0.0 = **DRED + SAGE**. This source tree contains exactly one implementation of it. The supported library interface is `hypermem.HyperMemory` (see the repository `README.md`); the PMM-Bench harness in the `eval/` tree drives the same pipeline through `pmm_bench.final`.

```text
Observation
  -> DRED (fast path, online, LLM-free)           hypermem/online/engine.py
  -> SAGE (slow path, idle)
       1. boundary consolidation                  hypermem/online/consolidation.py
       2. ToolSAGE: SplitEvent, SummarizeEvent, ExtractFacts
                                                  hypermem/online/sage_tools.py
       3. LongMemory consolidation                hypermem/online/sage.py
  -> grounded Event / Anchor / Fact / LongMemory graph
  -> HyperOS graph retrieval                      hypermem/online/hyperos_retrieval.py
```

Paths are relative to the repository root. There are no alternative formation modes, SAGE variants or configuration switches that turn a step off: every build runs DRED and then all three SAGE steps.

DRED attribution keys: the strong anchors (`order_id`, `transaction_id`, `tracking_id`, `reservation_id`) are event-level keys, an attachment that ignores similarity; phone numbers, account and contact IDs, role-bearing entities and a chat `session_id` (supplied through `HyperMemory.add`, scoped to its source app) are entity-level keys, candidates that still need `tau_s` similarity. Data without a `session_id`, such as PMM-Bench, is unaffected by the last.

Failure policy (`hypermem/online/failures.py`): every model task is retried within its budget; a task that still fails is skipped, the build continues, and `sage_report.json` lists it under `skipped_failures` with status `complete_with_skipped_failures`. The evaluation contract records the counts. Structural audit failures and pending work still stop the build. A failure of the model service itself (`hypermem.standardization.ModelServiceError`: unreachable, timing out, overloaded, or refusing the key, URL or model) is not a task failure: it spends no task's budget and stops the build, and running the same steps again once the service is restored resumes from the saved journals.

`eval/pmm_bench/pmm_benchmark.py` runs the PMM-Bench stages and `pmm_bench.final` wraps it:

| Stage | Does | Writes |
| --- | --- | --- |
| `prepare` | validates the package, imports one user's observations | `config.json`, `sage_config.json`, `protocol.json`, `run_manifest.json`, `memory.sqlite` |
| `dred` | replays observations through DRED | `dred_report.json`, `dred_audit.json` |
| `sage` | runs the three SAGE steps, then freezes the memory | `sage_boundary_report.json`, `sage_boundary_audit.json`, `sage_report.json`, `sage_tool_jobs.json`, `frozen.json` |
| `dashboard` | renders the memory graph | `../hypermem_dashboard.html` |
| `evaluate` | retrieves, answers and judges every question | `../results/<scope>/hyperos-mem/<user>/` |

`final_version.json` is the machine-readable form of this statement.

Embed the memory system through the library interface:

```python
from hypermem import HyperMemory   # see repository README.md
```

Describe or verify a benchmark build from the `eval/` tree:

```bash
cd eval
python -m pmm_bench.final describe
python -m pmm_bench.final check --run <output>/build
```
