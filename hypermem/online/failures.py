# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Failure policy of the SAGE slow path: retry, then skip.

Every model task gets its retry budget. A task that still fails is skipped:
its records stay in SQLite unchanged, the build continues, and the reports
list it. Skipped tasks never count as successes.

- boundary consolidation: a task that exhausted ``max_attempts``;
- ToolSAGE: an event job that exhausted its tool budget (the event keeps its
  DRED membership and boundary event card, and stays out of LongMemory);
- LongMemory: a group task that exhausted ``long_memory.max_attempts``.

Pending or still retryable work is never skipped.

A failure of the model service itself (``ModelServiceError``) is not a task
failure: it spends no budget, stops the stage, and the stage resumes from its
journal once the service is restored.
"""
import json

from hypermem.online.consolidation import iter_records


def boundary_failures(memory):
    """Skipped boundary consolidation tasks; raises if work is still pending."""
    last = memory.get_metadata("online_consolidation_last", {})
    if last.get("status") == "complete":
        return []
    if last.get("status") != "needs_review":
        raise ValueError("SAGE boundary consolidation has not finished")
    budget = memory.get_metadata("config")["online"]["consolidation"]["max_attempts"]
    failures = []
    for tid, t in iter_records(memory, "on_consolidation_tasks",
                               "json_extract(record_json,'$.status') NOT IN ('complete','obsolete')"):
        if t["status"] not in {"failed", "interrupted"} or t["attempts"] < t.get("attempt_limit", budget):
            raise ValueError("SAGE boundary consolidation has pending or retryable tasks")
        failures.append({"stage": "boundary", "id": tid, "kind": t["input"]["stage"], "attempts": t["attempts"],
                         "status": t["status"], "error_type": t.get("error_type"), "error": t.get("error")})
    if not failures or len(failures) != last.get("unfinished_tasks"):
        raise ValueError("SAGE boundary failure list disagrees with its report")
    return failures


def sage_failures(sage, *, include_long=False):
    """Skipped ToolSAGE event jobs and, after LongMemory, skipped group tasks."""
    current = set(sage.current_ids())
    db = sage.m._connection
    failures = []
    for jid, raw in db.execute("SELECT job_id, record_json FROM sage_tool_jobs "
                             "WHERE json_extract(record_json,'$.status')='budget_exhausted' ORDER BY job_id"):
        job = json.loads(raw)
        if job["event_id"] in current and job["version"] == sage.version(job["event_id"]):
            failures.append({"stage": "event_tools", "id": jid, "event_id": job["event_id"],
                             "version": job["version"], "rounds": job.get("rounds"), "calls": job.get("calls"),
                             "summary_done": job.get("summary_done"), "facts_done": job.get("facts_done"),
                             "last_tool_error": next(({"tool": r["name"], **r["result"]}
                                                      for r in reversed(job.get("results", []))
                                                      if r["result"].get("ok") is False), None),
                             "error": job.get("error")})
    if include_long:
        for tid, raw in db.execute("SELECT task_id, record_json FROM sage_tasks "
                                   "WHERE json_extract(record_json,'$.status')='failed' ORDER BY task_id"):
            task = json.loads(raw)
            if (task["input"].get("stage") == "consolidate_long_memory" and
                    task["attempts"] >= sage.cfg["long_memory"]["max_attempts"]):
                failures.append({"stage": "long_memory", "id": tid, "attempts": task["attempts"],
                                 "error_type": task.get("error_type"), "error": task.get("error")})
    return failures


def completion_gate(sage, audit, failures, *, event_only=False):
    """Match every SAGE audit issue to a skipped task; anything else is unresolved."""
    current = set(sage.current_ids())
    events = {r[0] for r in sage.m._connection.execute("SELECT event_id FROM sage_events")}
    skipped_events = {f["event_id"] for f in failures if f["stage"] == "event_tools"}
    skipped_tasks = {f["id"] for f in failures if f["stage"] == "long_memory"}
    missing = current - events
    extra = events - current
    unfinished = {r[0] for r in sage.m._connection.execute(
        "SELECT task_id FROM sage_tasks WHERE json_extract(record_json,'$.status') NOT IN ('complete','obsolete')")}
    accepted, unresolved, deferred = [], [], []
    for issue in audit["issues"]:
        code = issue["code"]
        if code == "unfinished_tasks" and event_only:
            deferred.append(issue)
        elif (code == "unfinished_tool_event" and issue["id"] in skipped_events or
              code == "event_coverage" and bool(missing) and not extra and missing <= skipped_events or
              code == "unfinished_tasks" and bool(unfinished) and unfinished <= skipped_tasks
              and issue["count"] == len(unfinished)):
            accepted.append(issue)
        else:
            unresolved.append(issue)
    incomplete = {i["id"] for i in audit["issues"] if i["code"] == "unfinished_tool_event"}
    return {"status": "failed" if unresolved else "skipped_failures" if accepted else "passed",
            "skipped_issues": accepted, "unresolved_issues": unresolved, "deferred_issues": deferred,
            "coverage": {"current_events": len(current),
                         "fully_enriched_events": len(current - incomplete - missing),
                         "skipped_event_jobs": len(skipped_events),
                         "missing_event_summaries": len(missing),
                         "skipped_long_memory_tasks": len(skipped_tasks)}}
