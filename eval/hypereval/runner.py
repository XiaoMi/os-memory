# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from hypereval.judge import LLMEvaluator
from hypereval.metrics import (
    bleu1,
    evidence_recall_at_k_multi,
    exact_match,
    gallery_bleu,
    gallery_exact_match,
    gallery_f1,
    locomo_qa_score,
    longmemeval_retrieval_metrics,
    longmemeval_session_retrieval_metrics,
    precision_recall_hitrate_at_k,
    reciprocal_rank_multi,
    token_f1,
)
from hypereval.model_config import api_key, base_url, model_name
from hypereval.openai_compat import install_openai_gateway_compat
from hypereval.protocol import BenchmarkSample, to_jsonable
from hypereval.registry import create_adapter, create_dataset

# Retained for compatibility with old run directories. Official LoCoMo scoring
# now includes category 5 rather than silently dropping it.
_ADVERSARIAL_CATEGORY = "5"


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_json_atomic(path: Path, data: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_resumable_predictions(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    discarded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("error"):
            discarded += 1
            continue
        query_id = str(item.get("query_id") or "")
        if query_id:
            existing[query_id] = item
    if discarded:
        _write_jsonl_atomic(path, list(existing.values()))
        print(
            f"[runner] discarded {discarded} failed retrieval prediction(s); "
            "they will be retried"
        )
    return existing


def _build_evaluator(system_context: dict[str, Any] | None) -> LLMEvaluator:
    """Construct the LLM answerer/judge from the system's own model config.

    Reuses the same OpenAI-compatible model + credentials the system under test
    uses (read from its system_config json), so answering and judging run on the
    configured gateway model (e.g. azure_openai/gpt-4o-mini) with no extra setup.
    """
    cfg: dict[str, Any] = {}
    config_path = (system_context or {}).get("system_config")
    if config_path:
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    model = model_name(cfg, "gpt-4o-mini")
    key = api_key(cfg, "OPENAI_API_KEY")
    url = base_url(cfg)
    return LLMEvaluator(model=model, api_key=key, base_url=url)


def run_benchmark(
    system_name: str,
    dataset_name: str,
    sample_size: int | None = None,
    output_dir: str | None = None,
    top_k: int = 10,
    resume: bool = False,
    rerun: bool = False,
    system_context: dict[str, Any] | None = None,
    data_path: str | None = None,
    resume_from: str | None = None,
    retry_rounds: int = 1,
) -> Path:
    install_openai_gateway_compat()
    root = Path(output_dir or f"runs/{_timestamp()}_{dataset_name}_{system_name}")
    if rerun and root.exists():
        shutil.rmtree(root)
    for directory in (root, root / "checkpoints", root / "system_state"):
        directory.mkdir(parents=True, exist_ok=True)

    samples = create_dataset(dataset_name).load(path=data_path, sample_size=sample_size)
    config = {
        "system": system_name,
        "dataset": dataset_name,
        "sample_size": sample_size,
        "top_k": top_k,
        "resume": resume,
        "resume_from": resume_from,
        "retry_rounds": retry_rounds,
        "official_protocol": _official_protocol(dataset_name),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {"run_id": root.name, "created_at": time.time(), **config}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    existing = {}
    predictions_path = root / "predictions.jsonl"
    if resume and predictions_path.exists():
        existing = _load_resumable_predictions(predictions_path)

    # A completed MemBench sample can contain one or more questions. When all
    # its scoreable questions already exist in predictions.jsonl, skip the
    # write phase as well as retrieval; this makes interrupted runs resumable
    # without re-running expensive memory construction for completed samples.
    carried_existing: set[str] = set()
    if existing:
        for sample in samples:
            scoreable = [
                question.query_id
                for question in sample.questions
            ]
            if scoreable and all(query_id in existing for query_id in scoreable):
                carried_existing.add(sample.sample_id)

    # ---- carry-forward: 结转上次目录里已成功样本的预测到本次 ----
    # 让本次目录自包含"迄今所有成功样本"的并集, metrics 自动从并集重算。
    all_results: list[dict[str, Any]] = []
    sample_status: dict[str, dict[str, Any]] = {}
    carried_ok: set[str] = set()
    if resume_from:
        carried_ok = _carry_forward(
            Path(resume_from), predictions_path, all_results, sample_status
        )
        if carried_ok:
            print(f"[runner] carried forward {len(carried_ok)} succeeded sample(s) from {resume_from}")

    adapter = create_adapter(system_name)
    adapter.initialize({
        "run_dir": str(root),
        "system": system_name,
        "dataset": dataset_name,
        **(system_context or {}),
    })
    evaluator = _build_evaluator(system_context)
    recorded_query_ids = {
        str(item.get("query_id")) for item in all_results if item.get("query_id")
    }
    # 待跑集 = 全部样本 - 已结转成功的样本
    pending = [
        s for s in samples
        if s.sample_id not in carried_ok and s.sample_id not in carried_existing
    ]
    for sample_id in carried_existing:
        sample_status[sample_id] = {
            "status": "success",
            "questions": sum(
                1 for item in existing.values() if item.get("sample_id") == sample_id
            ),
            "attempt": 0,
            "carried_from": "same_run_predictions",
        }
        all_results.extend(
            item for item in existing.values() if item.get("sample_id") == sample_id
        )
    status_path = root / "sample_status.json"
    if sample_status:
        _write_json_atomic(status_path, sample_status)
    try:
        if (
            dataset_name == "personamem"
            and getattr(adapter, "supports_incremental_checkpoints", False)
        ):
            _run_personamem_incremental(
                adapter,
                pending,
                all_results,
                existing,
                recorded_query_ids,
                predictions_path,
                root,
                top_k,
                evaluator,
                sample_status,
                status_path,
                retry_rounds,
            )
        else:
            remaining = pending
            # Pass 1 (attempt 1) plus at most retry_rounds retries.
            for attempt in range(1, max(0, retry_rounds) + 2):
                if not remaining:
                    break
                if attempt > 1:
                    print(f"[runner] retry round {attempt - 1}: {len(remaining)} failed sample(s)")
                still_failed = []
                for sample in remaining:
                    ok, err = _run_one_sample_guarded(
                        adapter,
                        sample,
                        all_results,
                        existing,
                        predictions_path,
                        root,
                        top_k,
                        evaluator,
                        recorded_query_ids,
                    )
                    if ok:
                        sample_status[sample.sample_id] = {
                            "status": "success",
                            "questions": sum(1 for r in all_results if r.get("sample_id") == sample.sample_id),
                            "attempt": attempt,
                        }
                    else:
                        sample_status[sample.sample_id] = {
                            "status": "failed", "error": err, "attempt": attempt,
                        }
                        still_failed.append(sample)
                    _write_json_atomic(status_path, sample_status)
                remaining = still_failed
        adapter.dump_state(root / "system_state")
    finally:
        adapter.close()

    _write_json_atomic(status_path, sample_status)
    metrics = _summarize(
        all_results,
        sample_status,
        total_samples=len(samples),
        dataset_name=dataset_name,
    )
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_official_report(root, config, metrics)
    return root


def _carry_forward(
    prev_dir: Path,
    predictions_path: Path,
    results: list[dict[str, Any]],
    sample_status: dict[str, dict[str, Any]],
) -> set[str]:
    """Copy succeeded samples' predictions from a previous run dir into this one.

    Reads prev_dir/sample_status.json (source of truth for which samples succeeded)
    and prev_dir/predictions.jsonl, appends the succeeded samples' prediction items
    to this run's predictions.jsonl + results + sample_status. Returns the set of
    carried-over succeeded sample_ids (so the caller can skip re-running them).

    Missing/unreadable prev files are tolerated (returns empty set): a resume that
    points at a bad path simply runs everything fresh rather than crashing.
    """
    status_path = prev_dir / "sample_status.json"
    preds_path = prev_dir / "predictions.jsonl"
    if not status_path.exists() or not preds_path.exists():
        print(f"[runner] resume_from {prev_dir} has no sample_status/predictions; running all samples")
        return set()
    try:
        prev_status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[runner] cannot read {status_path}: {exc}; running all samples")
        return set()
    requested_ok = {
        sid for sid, status in prev_status.items() if status.get("status") == "success"
    }
    rows_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    invalid_samples: set[str] = set()
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sample_id = str(item.get("sample_id") or "")
        query_id = str(item.get("query_id") or "")
        if not sample_id or not query_id:
            continue
        if item.get("error"):
            invalid_samples.add(sample_id)
            continue
        rows_by_sample.setdefault(sample_id, {})[query_id] = item
    ok_samples = set()
    for sample_id in requested_ok - invalid_samples:
        rows = rows_by_sample.get(sample_id, {})
        expected = prev_status.get(sample_id, {}).get("questions")
        if rows and (expected is None or len(rows) == int(expected)):
            ok_samples.add(sample_id)
    if not ok_samples:
        return set()
    with predictions_path.open("a", encoding="utf-8") as out:
        for sample_id in ok_samples:
            for item in rows_by_sample[sample_id].values():
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                results.append(item)
    for sid in ok_samples:
        prev = prev_status.get(sid, {})
        sample_status[sid] = {
            "status": "success",
            "questions": prev.get("questions"),
            "attempt": prev.get("attempt"),
            "carried_from": str(prev_dir),
        }
    return ok_samples


def _run_one_sample_guarded(
    adapter,
    sample,
    results,
    existing,
    predictions_path,
    root,
    top_k,
    evaluator,
    recorded_query_ids=None,
) -> tuple[bool, str | None]:
    """begin_sample + _run_sample with top-level fault isolation.

    Any exception from begin_sample (which may re-initialize a system) or an
    unexpected error is caught and reported as a sample-level failure, so one bad
    sample never aborts the run.
    """
    try:
        adapter.begin_sample(sample.sample_id)
        return _run_sample(
            adapter,
            sample,
            results,
            existing,
            predictions_path,
            root,
            top_k,
            evaluator,
            recorded_query_ids=recorded_query_ids,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one sample, keep the run alive
        err = f"{type(exc).__name__}: {exc}"
        print(f"[runner] sample {sample.sample_id} failed: {err}")
        return False, err


def _personamem_context_groups(
    samples: list[BenchmarkSample],
) -> list[list[BenchmarkSample]]:
    groups: dict[str, list[BenchmarkSample]] = {}
    for sample in samples:
        context_id = str(sample.metadata.get("shared_context_id") or sample.sample_id)
        groups.setdefault(context_id, []).append(sample)
    return [
        sorted(
            group,
            key=lambda sample: int(
                sample.metadata.get("end_index") or len(sample.events)
            ),
        )
        for group in groups.values()
    ]


def _run_personamem_incremental(
    adapter,
    samples,
    results,
    existing,
    recorded_query_ids,
    predictions_path,
    root,
    top_k,
    evaluator,
    sample_status,
    status_path,
    retry_rounds,
) -> None:
    remaining = _personamem_context_groups(samples)
    for attempt in range(1, max(0, retry_rounds) + 2):
        if not remaining:
            break
        if attempt > 1:
            print(
                f"[runner] PersonaMem retry round {attempt - 1}: "
                f"{len(remaining)} failed context(s)"
            )
        still_failed = []
        for group in remaining:
            completed_count, err = _run_personamem_group_guarded(
                adapter,
                group,
                attempt,
                results,
                existing,
                recorded_query_ids,
                predictions_path,
                root,
                top_k,
                evaluator,
            )
            for sample in group[:completed_count]:
                sample_status[sample.sample_id] = {
                    "status": "success",
                    "questions": sum(
                        1
                        for result in results
                        if result.get("sample_id") == sample.sample_id
                    ),
                    "attempt": attempt,
                    "incremental_context": True,
                }
            if completed_count < len(group):
                still_failed.append(group)
                for sample in group[completed_count:]:
                    question_ids = [question.query_id for question in sample.questions]
                    if question_ids and all(query_id in existing for query_id in question_ids):
                        continue
                    sample_status[sample.sample_id] = {
                        "status": "failed",
                        "error": err,
                        "attempt": attempt,
                        "incremental_context": True,
                    }
            _write_json_atomic(status_path, sample_status)
        remaining = still_failed


def _run_personamem_group_guarded(
    adapter,
    samples,
    attempt,
    results,
    existing,
    recorded_query_ids,
    predictions_path,
    root,
    top_k,
    evaluator,
) -> tuple[int, str | None]:
    context_id = str(
        samples[0].metadata.get("shared_context_id") or samples[0].sample_id
    )
    try:
        adapter.begin_sample(f"personamem-{context_id}-attempt-{attempt}")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"

    previous_event_ids: tuple[str, ...] = ()
    for index, sample in enumerate(samples):
        current_event_ids = tuple(event.event_id for event in sample.events)
        if current_event_ids[: len(previous_event_ids)] != previous_event_ids:
            return index, (
                "RuntimeError: PersonaMem checkpoints are not monotonic prefixes "
                f"for context {context_id}"
            )
        delta_events = sample.events[len(previous_event_ids) :]
        try:
            ok, err = _run_sample(
                adapter,
                sample,
                results,
                existing,
                predictions_path,
                root,
                top_k,
                evaluator,
                events_to_ingest=delta_events,
                reset_before_write=index == 0,
                recorded_query_ids=recorded_query_ids,
            )
        except Exception as exc:  # noqa: BLE001
            return index, f"{type(exc).__name__}: {exc}"
        if not ok:
            return index, err
        previous_event_ids = current_event_ids
    return len(samples), None


def _run_sample(
    adapter,
    sample: BenchmarkSample,
    results,
    existing,
    predictions_path,
    root,
    top_k,
    evaluator,
    events_to_ingest=None,
    reset_before_write: bool = True,
    recorded_query_ids=None,
) -> tuple[bool, str | None]:
    """Run one sample end-to-end.

    Returns ``(ok, error)``: ok=True when the sample's memory was written and all
    its questions were scored (individual retrieval failures are tolerated and
    scored ~0, the sample still counts as success). ok=False when the write phase
    (reset/ingest/finalize_ingest — graph build / memory write) raised, in which
    case the sample produced no predictions and should be retried later.
    """
    # Write phase (reset + ingest + finalize) is where graph-building / memory
    # extraction happens and where a gateway error aborts a whole sample. Wrap it
    # all: on failure the sample wrote 0 predictions and is reported as failed.
    try:
        ingest_events = sample.events if events_to_ingest is None else events_to_ingest
        if reset_before_write:
            users = {event.user_id for event in sample.events}
            for user_id in users:
                adapter.reset(user_id)
        write_started = time.perf_counter()
        for event in ingest_events:
            adapter.ingest(event)
        adapter.finalize_ingest()
        write_ms = (time.perf_counter() - write_started) * 1000
        write_per_event_ms = write_ms / len(ingest_events) if ingest_events else 0.0
    except Exception as exc:  # noqa: BLE001 - isolate one sample, keep the run alive
        err = f"{type(exc).__name__}: {exc}"
        print(f"[runner] write phase failed for sample {sample.sample_id}: {err}")
        return False, err
    with predictions_path.open("a", encoding="utf-8") as output:
        for question in sample.questions:
            if question.query_id in existing:
                if recorded_query_ids is None or question.query_id not in recorded_query_ids:
                    results.append(existing[question.query_id])
                    if recorded_query_ids is not None:
                        recorded_query_ids.add(question.query_id)
                continue
            started = time.perf_counter()
            # A retrieval timeout is not a valid low-scoring answer. Fail the
            # sample attempt without writing this question so the retry loop (or
            # a later --resume run) executes the missing question again.
            try:
                retrieval_query = question.question
                if sample.dataset == "gallerymem":
                    caption = str(
                        (question.metadata or {}).get("question_image_caption") or ""
                    ).strip()
                    if caption:
                        retrieval_query += f"\nquestion's image:\nimage_caption: {caption}"
                retrieval_limit = (
                    min(len(sample.events), max(top_k, 100))
                    if sample.dataset == "longmemeval"
                    and getattr(adapter, "supports_evidence_tracing", True)
                    else top_k
                )
                if hasattr(adapter, "retrieve_question"):
                    records = adapter.retrieve_question(question, retrieval_limit)
                else:
                    records = adapter.retrieve(
                        retrieval_query,
                        question.user_id,
                        question.session_id,
                        retrieval_limit,
                    )
            except Exception as exc:  # noqa: BLE001 - isolate one question, keep the run alive
                retrieve_error = f"{type(exc).__name__}: {exc}"
                print(f"[runner] retrieve failed for {question.query_id}: {retrieve_error}")
                return False, retrieve_error
            retrieval_ms = (time.perf_counter() - started) * 1000
            memories = [record.content for record in records[:top_k]]
            # Multiple-choice datasets (e.g. PersonaMem) carry the option list in
            # ``question.metadata["all_options"]``. Detecting MCQ by that generic
            # signal keeps this branch dataset-agnostic (no dataset-name check):
            # the answerer picks an option letter and we score it exact-match
            # against the correct letter — that IS the official MCQ accuracy.
            options = (question.metadata or {}).get("all_options")
            is_mcq = bool(options)
            if is_mcq:
                correct_option = str((question.metadata or {}).get("correct_option") or "")
                prediction = evaluator.answer_mcq(
                    question.question,
                    memories,
                    options,
                    dataset=sample.dataset,
                    question_metadata=question.metadata,
                )
                mcq_score = evaluator.score_mcq(prediction, correct_option)
                judge_score = mcq_score
                em_score = None
                f1_score = None
                bleu1_score = None
            else:
                # LLM reads the retrieved memories and produces a short answer (falls
                # back to concatenated memories if the LLM is unavailable).
                question_metadata = question.metadata or {}
                prediction = evaluator.generate_answer(
                    question.question,
                    memories,
                    dataset=sample.dataset,
                    question_type=question.category,
                    question_date=str(question_metadata.get("question_date") or ""),
                    question_metadata=question_metadata,
                )
                # LLM judge grades the answer against the gold answers with the official
                # generous口径; None means the judge did not run (unavailable/ambiguous).
                if sample.dataset == "locomo":
                    judge_score = None
                    em_score = None
                    f1_score = None
                    bleu1_score = None
                elif sample.dataset == "longmemeval":
                    judge_score = evaluator.judge(
                        question.question,
                        prediction,
                        question.answers,
                        dataset=sample.dataset,
                        question_type=question.category,
                        question_metadata=question_metadata,
                    )
                    if judge_score is None:
                        raise RuntimeError(
                            f"LongMemEval official judge returned no score for {question.query_id}"
                        )
                    em_score = None
                    f1_score = None
                    bleu1_score = None
                elif sample.dataset == "gallerymem":
                    judge_score = (
                        evaluator.judge(
                            question.question,
                            prediction,
                            question.answers,
                            dataset=sample.dataset,
                            question_type=question.category,
                            question_metadata=question_metadata,
                        )
                        if os.getenv("HYPEREVAL_GALLERYMEM_LLM_JUDGE", "0") == "1"
                        else None
                    )
                    em_score = None
                    f1_score = None
                    bleu1_score = None
                else:
                    judge_score = evaluator.judge(
                        question.question,
                        prediction,
                        question.answers,
                        dataset=sample.dataset,
                        question_type=question.category,
                        question_metadata=question_metadata,
                    )
                    em_score = exact_match(prediction, question.answers)
                    f1_score = token_f1(prediction, question.answers)
                    bleu1_score = bleu1(prediction, question.answers)
            # Evidence metrics trace retrieved records back to their source event ids
            # (the dataset's native dia_ids), not the adapter's internal memory ids.
            # Extraction/rewriting systems can't trace reliably; they declare
            # supports_evidence_tracing = False and we record recall/mrr as N/A.
            source_id_lists = [list(record.source_event_ids) for record in records]
            if sample.dataset == "longmemeval":
                event_roles = {event.event_id: event.role for event in sample.events}
                source_id_lists = [
                    [
                        source_id
                        for source_id in source_ids
                        if event_roles.get(str(source_id)) == "user"
                    ]
                    for source_ids in source_id_lists
                ]
            if sample.dataset in {
                "locomo",
                "longmemeval",
                "personamem",
                "membench",
                "gallerymem",
            }:
                recall_5 = None
                mrr = None
            elif getattr(adapter, "supports_evidence_tracing", True):
                recall_5 = evidence_recall_at_k_multi(source_id_lists, question.evidence_event_ids, 5)
                mrr = reciprocal_rank_multi(source_id_lists, question.evidence_event_ids)
            else:
                recall_5 = None
                mrr = None
            official: dict[str, Any] = {}
            if sample.dataset == "locomo":
                official["f1"] = locomo_qa_score(
                    prediction, question.answers, question.category
                )
            elif sample.dataset == "longmemeval":
                official["qa_accuracy"] = judge_score
                if question.metadata.get("abstention"):
                    official["abstention_accuracy"] = judge_score
                elif (
                    getattr(adapter, "supports_evidence_tracing", True)
                    and question.metadata.get("has_user_evidence")
                ):
                    for cutoff in (5, 10, 50):
                        _, turn_all, turn_ndcg = longmemeval_retrieval_metrics(
                            source_id_lists, question.evidence_event_ids, cutoff
                        )
                        official[f"turn_recall_all_at_{cutoff}"] = turn_all
                        official[f"turn_ndcg_any_at_{cutoff}"] = turn_ndcg
                        if cutoff <= 10:
                            event_sessions = {
                                event.event_id: event.session_id for event in sample.events
                            }
                            corpus_sessions = [
                                event.session_id
                                for event in sample.events
                                if event.role == "user"
                            ]
                            _, session_all, session_ndcg = longmemeval_session_retrieval_metrics(
                                source_id_lists,
                                question.metadata.get("answer_session_ids") or [],
                                cutoff,
                                event_sessions,
                                corpus_sessions,
                            )
                            official[f"session_recall_all_at_{cutoff}"] = session_all
                            official[f"session_ndcg_any_at_{cutoff}"] = session_ndcg
            elif sample.dataset in {"personamem", "membench"}:
                official["accuracy"] = judge_score
                if sample.dataset == "membench":
                    official["write_time_seconds"] = write_per_event_ms / 1000
                    official["read_time_seconds"] = retrieval_ms / 1000
                    if (
                        getattr(adapter, "supports_evidence_tracing", True)
                        and question.evidence_event_ids
                    ):
                        retrieved_sources = {
                            str(source_id)
                            for source_ids in source_id_lists[:top_k]
                            for source_id in source_ids
                        }
                        evidence = {str(value) for value in question.evidence_event_ids}
                        official["recall_at_10"] = len(
                            retrieved_sources & evidence
                        ) / len(evidence)
                    else:
                        official["recall_at_10"] = None
            elif sample.dataset == "gallerymem":
                official["f1"] = gallery_f1(prediction, question.answers)
                official["bleu_1"] = gallery_bleu(
                    prediction, question.answers, weights=(1.0, 0.0, 0.0, 0.0)
                )
                official["em"] = gallery_exact_match(prediction, question.answers)
                if judge_score is not None:
                    official["llm_judge_score"] = judge_score
                if question.category == "AR":
                    official["precision_at_10"] = None
                    official["recall_at_10"] = None
                    official["hitrate_at_10"] = None
                elif getattr(adapter, "supports_evidence_tracing", True):
                    precision, recall, hitrate = precision_recall_hitrate_at_k(
                        source_id_lists, question.evidence_event_ids, 10
                    )
                    official["precision_at_10"] = precision
                    official["recall_at_10"] = recall
                    official["hitrate_at_10"] = hitrate
                else:
                    official["precision_at_10"] = None
                    official["recall_at_10"] = None
                    official["hitrate_at_10"] = None
            item = {
                "sample_id": sample.sample_id,
                "query_id": question.query_id,
                "question": question.question,
                "category": question.category,
                "prediction": prediction,
                "references": list(question.answers),
                "retrieved_memory_ids": [record.memory_id for record in records],
                "retrieved_source_event_ids": source_id_lists,
                "retrieved_records": to_jsonable(records),
                "exact_match": em_score,
                "f1": f1_score,
                "bleu1": bleu1_score,
                "llm_judge": judge_score,
                "evidence_recall_at_5": recall_5,
                "mrr": mrr,
                "retrieval_latency_ms": retrieval_ms,
                "write_latency_ms": write_per_event_ms,
                "sample_write_latency_ms": write_ms,
                "error": None,
                "official": official,
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            output.flush()
            existing[question.query_id] = item
            if recorded_query_ids is None or question.query_id not in recorded_query_ids:
                results.append(item)
                if recorded_query_ids is not None:
                    recorded_query_ids.add(question.query_id)
    return True, None


def _mean_present(results: list[dict[str, Any]], key: str) -> float | None:
    """Average a metric over items where it is present (not None).

    Returns None when no item carries the metric (e.g. evidence recall for an
    extraction system that declared it untraceable), so it renders as N/A rather
    than a misleading 0.
    """
    values = [float(item[key]) for item in results if item.get(key) is not None]
    return sum(values) / len(values) if values else None


def _official_protocol(dataset_name: str) -> dict[str, Any]:
    protocols = {
        "locomo": {"metrics": ["F1"]},
        "longmemeval": {
            "metrics": [
                "overall_accuracy",
                "task_averaged_accuracy",
                "accuracy_by_question_type",
                "abstention_accuracy",
                "turn_recall_all_at_5_10_50",
                "turn_ndcg_any_at_5_10_50",
                "session_recall_all_at_5_10",
                "session_ndcg_any_at_5_10",
            ],
            "judge_model": os.getenv(
                "HYPEREVAL_LONGMEMEVAL_JUDGE_MODEL", "azure_openai/gpt-4o"
            ),
            "official_reference_model": "gpt-4o-2024-08-06",
        },
        "personamem": {"metrics": ["overall_accuracy", "accuracy_by_query_type"]},
        "gallerymem": {
            "metrics": [
                "F1",
                "BLEU-1",
                "EM",
                "LLM_JUDGE",
                "Precision@10",
                "Recall@10",
                "Hit@10",
            ],
            "official_reference_judge_model": "Qwen-2.5-72B-Instruct",
            "judge_model": os.getenv(
                "HYPEREVAL_GALLERYMEM_JUDGE_MODEL",
                "Qwen2.5-72B-Instruct",
            ),
        },
        "membench": {
            "metrics": ["Accuracy", "Recall@10", "RT", "WT", "Capacity"]
        },
    }
    return protocols.get(dataset_name, {})


def _sample_summary(
    sample_status: dict[str, dict[str, Any]] | None, total_samples: int | None
) -> dict[str, Any]:
    """Sample-level coverage fields shared by success/empty summaries."""
    status = sample_status or {}
    completed = sorted(sid for sid, st in status.items() if st.get("status") == "success")
    failed = sorted(sid for sid, st in status.items() if st.get("status") == "failed")
    total = total_samples if total_samples is not None else len(status)
    out: dict[str, Any] = {
        "total_samples": total,
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "failed_sample_ids": failed,
        "sample_coverage": (len(completed) / total) if total else None,
    }
    return out


def _summarize(
    results: list[dict[str, Any]],
    sample_status: dict[str, dict[str, Any]] | None = None,
    total_samples: int | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    sample_fields = _sample_summary(sample_status, total_samples)
    if not results:
        # No succeeded-sample questions to score, but still report sample coverage
        # (e.g. every sample failed) so the run is transparent rather than empty.
        return {"questions": 0, **sample_fields}
    summary: dict[str, Any] = {"questions": len(results)}
    official_dataset = dataset_name in {
        "locomo",
        "longmemeval",
        "personamem",
        "gallerymem",
        "membench",
    }
    diagnostics: dict[str, Any] = {}
    # Always-present per-item floats.
    for key in (
        "exact_match",
        "f1",
        "bleu1",
        "retrieval_latency_ms",
        "write_latency_ms",
        "sample_write_latency_ms",
    ):
        diagnostics[key] = _mean_present(results, key)
    # llm_judge is the primary, official-comparable accuracy; evidence recall/mrr
    # may be None for extraction systems. Average only over items where present.
    for key in ("llm_judge", "evidence_recall_at_5", "mrr"):
        diagnostics[key] = _mean_present(results, key)
    # How much of the set the LLM judge actually covered.
    judged = [item for item in results if item.get("llm_judge") is not None]
    diagnostics["llm_judge_coverage"] = len(judged) / len(results)
    # How many questions had an isolated retrieval failure (fault-tolerated, scored
    # ~0 instead of aborting the run). 0 means a clean run.
    diagnostics["retrieval_errors"] = sum(1 for item in results if item.get("error"))
    summary["retrieval_errors"] = diagnostics["retrieval_errors"]
    official = _summarize_official(results, dataset_name)
    if official:
        summary["official"] = official
    # Per-category accuracy breakdown (llm_judge averaged within each category).
    # PersonaMem reports accuracy per question_type; LoCoMo can use it too. Only
    # emitted when questions carry a category and a judged score.
    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        category = item.get("category")
        score = item.get("llm_judge")
        if not category or score is None:
            continue
        bucket = by_category.setdefault(category, {"n": 0, "sum": 0.0})
        bucket["n"] += 1
        bucket["sum"] += float(score)
    if by_category:
        diagnostics["by_category"] = {
            cat: {"accuracy": b["sum"] / b["n"], "questions": b["n"]}
            for cat, b in sorted(by_category.items())
        }
    # Sample-level coverage: metrics above are computed only over succeeded samples'
    # questions; these fields say how many samples that is and which failed.
    if not official_dataset:
        summary["diagnostics"] = diagnostics
    summary.update(sample_fields)
    return summary


def _summarize_official(
    results: list[dict[str, Any]], dataset_name: str | None = None
) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    category_values: dict[str, dict[str, list[float]]] = {}
    for item in results:
        category = str(item.get("category") or "")
        for name, value in (item.get("official") or {}).items():
            values.setdefault(name, [])
            if category:
                category_values.setdefault(category, {}).setdefault(name, [])
            if value is None:
                continue
            values[name].append(float(value))
            if category:
                category_values[category][name].append(float(value))
    if not values:
        return {}
    raw_overall = {
        name: (sum(items) / len(items) if items else None)
        for name, items in sorted(values.items())
    }
    raw_by_category = {
        category: {
            name: (sum(items) / len(items) if items else None)
            for name, items in sorted(metrics.items())
        }
        for category, metrics in sorted(category_values.items())
    }
    if dataset_name == "locomo":
        return {
            "overall": {"f1": raw_overall["f1"]},
            "by_category": {
                category: {"f1": metrics["f1"]}
                for category, metrics in raw_by_category.items()
                if "f1" in metrics
            },
        }
    if dataset_name == "personamem":
        return {
            "overall": {"overall_accuracy": raw_overall["accuracy"]},
            "by_category": {
                category: {"accuracy": metrics["accuracy"]}
                for category, metrics in raw_by_category.items()
                if "accuracy" in metrics
            },
        }
    if dataset_name == "longmemeval":
        overall = {
            name: value
            for name, value in raw_overall.items()
            if name != "qa_accuracy"
        }
        if "qa_accuracy" in raw_overall:
            overall["overall_accuracy"] = raw_overall["qa_accuracy"]
        task_means = [
            sum(items["qa_accuracy"]) / len(items["qa_accuracy"])
            for items in category_values.values()
            if items.get("qa_accuracy")
        ]
        if task_means:
            overall["task_averaged_accuracy"] = sum(task_means) / len(task_means)
        return {
            "overall": dict(sorted(overall.items())),
            "by_category": {
                category: {"accuracy": metrics["qa_accuracy"]}
                for category, metrics in raw_by_category.items()
                if "qa_accuracy" in metrics
            },
        }
    if dataset_name == "membench":
        return {
            "overall": raw_overall,
            "by_category": {
                category: {
                    name: value
                    for name, value in metrics.items()
                    if name in {"accuracy", "recall_at_10", "read_time_seconds"}
                }
                for category, metrics in raw_by_category.items()
            },
        }
    return {"overall": raw_overall, "by_category": raw_by_category}


def _write_report(root: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# HyperMemory Eval Run Report",
        "",
        f"- System: `{config['system']}`",
        f"- Dataset: `{config['dataset']}`",
        f"- Questions: `{metrics.get('questions', 0)}`",
        "",
        "## Official Benchmark Metrics",
        "",
    ]
    diagnostics = metrics.get("diagnostics") or {}
    by_category = None
    official = metrics.get("official")
    if official:
        for key, value in official.get("overall", {}).items():
            lines.append(f"- `{key}`: `{_fmt(value)}`")
        if official.get("by_category"):
            lines += ["", "### By Category", ""]
            for category, values in official["by_category"].items():
                rendered = ", ".join(
                    f"{key}={_fmt(value)}" for key, value in values.items()
                )
                lines.append(f"- `{category}`: {rendered}")
    else:
        lines.append("- No official metric was produced.")
    lines += ["", "## Run Coverage", ""]
    official = None
    for key, value in metrics.items():
        if key in {"by_category", "diagnostics", "official"}:
            continue  # rendered as its own table below
        if value is None:
            lines.append(f"- `{key}`: `N/A`")
        elif isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.4f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    if by_category:
        lines += ["", "## 按题型 accuracy", "", "| 题型 | accuracy | 题数 |", "|---|---:|---:|"]
        for cat, stats in by_category.items():
            lines.append(f"| `{cat}` | {stats['accuracy']:.4f} | {stats['questions']} |")
    if official:
        lines += ["", "## Official Benchmark Metrics", ""]
        for key, value in official.get("overall", {}).items():
            lines.append(f"- `{key}`: `{value:.4f}`")
        if official.get("by_category"):
            lines += ["", "### By Category", ""]
            for category, values in official["by_category"].items():
                rendered = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
                lines.append(f"- `{category}`: {rendered}")
    lines += ["", "## Diagnostics", ""]
    for key, value in diagnostics.items():
        if key == "by_category":
            continue
        if value is None:
            lines.append(f"- `{key}`: `N/A`")
        elif isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.4f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    # 失败样本单独提示, 方便下次 --resume 补跑。
    failed = metrics.get("failed_sample_ids") or []
    if failed:
        lines += [
            "",
            "## 失败样本（下次可用 --resume 指向本目录补跑）",
            "",
            f"共 {len(failed)} 个：{', '.join(failed)}",
        ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_official_report(
    root: Path, config: dict[str, Any], metrics: dict[str, Any]
) -> None:
    lines = [
        "# HyperMemory Eval Run Report",
        "",
        f"- System: `{config['system']}`",
        f"- Dataset: `{config['dataset']}`",
        f"- Questions: `{metrics.get('questions', 0)}`",
        "",
        "## Official Benchmark Metrics",
        "",
    ]
    official = metrics.get("official") or {}
    if official:
        for key, value in official.get("overall", {}).items():
            lines.append(f"- `{key}`: `{_fmt(value)}`")
        if official.get("by_category"):
            lines += ["", "### By Category", ""]
            for category, values in official["by_category"].items():
                rendered = ", ".join(
                    f"{key}={_fmt(value)}" for key, value in values.items()
                )
                lines.append(f"- `{category}`: {rendered}")
    else:
        lines.append("- No official metric was produced.")

    lines += ["", "## Run Coverage", ""]
    for key in (
        "questions",
        "total_samples",
        "completed_samples",
        "failed_samples",
        "sample_coverage",
    ):
        value = metrics.get(key)
        if value is None:
            lines.append(f"- `{key}`: `N/A`")
        elif isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.4f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")

    diagnostics = metrics.get("diagnostics") or {}
    if diagnostics:
        lines += ["", "## Diagnostics", ""]
        for key, value in diagnostics.items():
            if key == "by_category":
                continue
            if value is None:
                lines.append(f"- `{key}`: `N/A`")
            elif isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.4f}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
    if diagnostics.get("by_category"):
        lines += [
            "",
            "### Diagnostic Accuracy By Category",
            "",
            "| Category | Accuracy | Questions |",
            "|---|---:|---:|",
        ]
        for category, values in diagnostics["by_category"].items():
            lines.append(
                f"| `{category}` | {values['accuracy']:.4f} | {values['questions']} |"
            )

    failed = metrics.get("failed_sample_ids") or []
    if failed:
        lines += ["", "## Failed Samples", "", ", ".join(failed)]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    """Format a metric cell: 4-decimals for numbers, N/A for missing/None."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def compare_runs(run_dirs: list[str], output: str | None = None) -> Path:
    target = Path(output or "comparison.md")
    lines = [
        "# HyperMemory Eval Comparison",
        "",
        "| Run | System | Dataset | Questions | Judge | EM | F1 | BLEU-1 | Recall@5 | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_dir in run_dirs:
        root = Path(run_dir)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        lines.append(
            f"| `{root.name}` | `{config['system']}` | `{config['dataset']}` "
            f"| {metrics.get('questions', 0)} | {_fmt(metrics.get('llm_judge'))} "
            f"| {_fmt(metrics.get('exact_match'))} | {_fmt(metrics.get('f1'))} "
            f"| {_fmt(metrics.get('bleu1'))} "
            f"| {_fmt(metrics.get('evidence_recall_at_5'))} | {_fmt(metrics.get('mrr'))} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
