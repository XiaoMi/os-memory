# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Answer, judge and score one user's questions from retrieved memory."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from pmm_bench import llm
from pmm_bench.data import ABSTAIN, CATEGORIES, chinese_f1
from pmm_bench.prompts import VERSION, answer_messages, judge_messages, parse_judge


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_path(destination: Path, query_id: str) -> Path:
    return destination / (hashlib.sha256(query_id.encode()).hexdigest()[:20] + ".json")


def build_context(records, char_limit: int) -> str:
    """Concatenate retrieved memories in rank order within a character budget."""
    parts, used = [], 0
    for i, record in enumerate(records, 1):
        text = f"[Memory {i}] {record.content}\n"
        if used + len(text) > char_limit:
            text = text[:max(0, char_limit - used)]
        if not text:
            break
        parts.append(text)
        used += len(text)
    return "".join(parts)


def summarize(records: list[dict], questions) -> dict:
    expected = Counter(q.category for q in questions)
    def group(items, count):
        done = [r for r in items if r.get("status") == "complete"]
        return {"expected": count, "completed": len(done),
                "correct": sum(r["score"] for r in done),
                "accuracy": sum(r["score"] for r in done) / len(done) if done else None,
                "f1": sum(r["f1"] for r in done) / len(done) if done else None,
                "abstentions": sum(r.get("prediction", "").strip().rstrip("。.!！?？") == ABSTAIN for r in done),
                "errors": sum(r.get("status") == "failed" for r in items)}
    return {"overall": group(records, len(questions)),
            "by_category": {c: group([r for r in records if r["category"] == c], expected.get(c, 0))
                            for c in CATEGORIES}}


def answer_all(questions, retrieve, destination: Path, cfg: dict, meta: dict) -> dict:
    """retrieve(question) -> list[MemoryRecord]. Each question is resumable on disk."""
    destination.mkdir(parents=True, exist_ok=True)
    top_k, char_limit = cfg["retrieval"]["top_k"], cfg["retrieval"]["context_chars"]
    results = []
    for q in questions:
        path = record_path(destination, q.query_id)
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "query_id": q.query_id, "user_id": q.user_id, "category": q.category,
            "question": q.question, "references": list(q.answers), "prompt_version": VERSION, **meta}
        if record.get("status") == "complete":
            results.append(record)
            continue
        record["attempts"] = record.get("attempts", 0) + 1
        started = time.monotonic()
        try:
            if "context" not in record:
                t0 = time.perf_counter()
                memories = list(retrieve(q))[:top_k]
                record["retrieval"] = {"latency_ms": (time.perf_counter() - t0) * 1000,
                    "records": [{"memory_id": m.memory_id, "score": m.score, "content": m.content,
                                 "source_event_ids": list(m.source_event_ids)} for m in memories]}
                record["context"] = build_context(memories, char_limit)
                write_json(path, record)
            if "prediction" not in record:
                reply = llm.chat(cfg["llm"], {"model": cfg["models"]["answer"], "max_tokens": cfg["answer_max_tokens"],
                                              "messages": answer_messages(q, record["context"])})
                record.update(prediction=reply["content"], answer_response=reply["response"])
                write_json(path, record)
            reply = llm.chat(cfg["llm"], {"model": cfg["models"]["judge"], "max_tokens": cfg["judge_max_tokens"],
                                          "messages": judge_messages(q, record["prediction"])})
            judged = parse_judge(reply["content"])
            record.update(judge=judged, judge_response=reply["response"], status="complete",
                          score=int(judged["label"] == "CORRECT"), f1=chinese_f1(record["prediction"], q.answers))
            record.pop("error", None)
        except Exception as exc:  # noqa: BLE001 - one question fails, the run continues
            record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)[:300]})
        record["elapsed_seconds"] = record.get("elapsed_seconds", 0) + time.monotonic() - started
        write_json(path, record)
        results.append(record)
        print(json.dumps({"user": q.user_id, "query_id": q.query_id, "status": record["status"],
                          "score": record.get("score")}, ensure_ascii=False), flush=True)
    summary = {**meta, **summarize(results, questions)}
    write_json(destination / "summary.json", summary)
    return summary
