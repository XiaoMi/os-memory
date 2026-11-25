# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""PMM-Bench stages for HyperOS-Mem V11: prepare -> dred -> sage -> dashboard -> evaluate.

Run through pmm_bench.final, the benchmark entry point. No QA is read during
construction; evaluation opens the frozen SQLite database read-only.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import gc
from types import SimpleNamespace

from hypermem.config import config
from hypermem.import_observations import import_jsonl
from hypermem.memory import EventMemory
from hypermem.memory.writer_lock import writer_lock
from hypermem.online.io import write_json, audit
from hypermem.online.engine import OnlineHyperMem
from hypermem.source_verify import verify_source
from hypermem.standardization import make_encoder, request_json

DEFAULT_USER = None  # pass --user explicitly
STAGES = ("prepare","dred","sage","dashboard","evaluate")
# Finished SAGE states; skipped failures are listed in the reports (hypermem.online.failures).
FINISHED = ("complete","complete_with_skipped_failures")
SMOKE_LIMIT = 1000
# These decide the stored vectors, so a run cannot switch them midway; other settings may change on resume.
EMBEDDING_IDENTITY = ("embedding.model", "embedding.dimension")
SETTINGS_CHANGED = "settings_changed"


def setting_changes(old, new, prefix=""):
    """Leaf settings that differ between two JSON objects, as (dotted path, old, new)."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [change for key in sorted(set(old) | set(new))
                for change in setting_changes(old.get(key), new.get(key), f"{prefix}.{key}" if prefix else key)]
    return [] if old == new else [(prefix, old, new)]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def settings_for(protocol):
    emb = protocol["embedding"]
    base, path = emb["url"].rsplit("/v1/", 1)
    embedding = {"backend":"http", "base_url":base, "path":"/v1/"+path,
                 "base_url_env":"", "model_env":"", "api_key_env":emb.get("api_key_env",""),
                 "model":emb["model"], "dimension":emb["dimension"], "batch_size":emb["batch_size"],
                 "protocol":"openai", "timeout":60, "retries":2}
    llm = {"base_url":protocol["api_base_url"], "path":"/chat/completions",
           "base_url_env":"", "model_env":"", "api_key_env":protocol["api_key_env"],
           "model":protocol["construction_model"], "temperature":0, "max_tokens":8192,
           "timeout":180, "retries":2}
    formation = config({"llm":llm, "embedding":embedding,
                        "online":{"consolidation":{"batch_size":2, "concurrency":8}}})
    from hypermem.online.sage import config as sage_config
    sage = sage_config({"llm":{k:v for k,v in llm.items() if k not in {"base_url_env","model_env"}},
                        "embedding":{k:v for k,v in embedding.items() if k not in {"base_url_env","model_env"}},
                        "event":{"concurrency":8}})
    return formation, sage


def validate_user(package, user):
    """Validate the explicitly selected cleaned input, without consulting QA."""
    manifest = load(package/"dataset_manifest.json")
    info = next(u for u in manifest["users"] if u["user_id"] == user)
    checksums = load(package/"checksums.json")
    relative = info["cleaned_observations"]
    path = package/relative
    if file_hash(path) != checksums[relative]["sha256"] or path.stat().st_size != checksums[relative]["bytes"]:
        raise ValueError("Input checksum mismatch: " + relative)
    previous = None
    counts, types = Counter(), Counter()
    mapped = set()
    owners = set()
    months = set()
    from zoneinfo import ZoneInfo
    for count, line in enumerate(path.open(encoding="utf-8-sig"), 1):
        raw = json.loads(line)
        position = (raw["ts"], count)
        if previous is not None and position <= previous:
            raise ValueError("Cleaned input chronology mismatch")
        cleaning = raw.get("_cleaning", {})
        source_lines = cleaning.get("source_lines")
        if not isinstance(source_lines,list) or not source_lines or any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in source_lines):
            raise ValueError("Cleaned input lost original source-line mapping")
        mapped.update(source_lines)
        owners.add(raw["payload"]["user_id"])
        months.add(datetime.fromtimestamp(raw["ts"]/1000,ZoneInfo("Asia/Shanghai")).strftime("%Y-%m"))
        previous = position
        counts[raw["payload"]["event"].get("perception_type","unknown")] += 1
        types[raw["type"]] += 1
    if sum(types.values()) != info["cleaned_event_count"] or file_hash(path) != info["cleaned_sha256"]:
        raise ValueError("Incomplete cleaned history")
    if len(owners)!=1 or not months<=set(info["selected_months"]):
        raise ValueError("Cleaned input contains mixed users or out-of-scope months")
    return {"status":"passed", "user":info, "perception_types":dict(counts), "raw_types":dict(types),
            "verified_records":sum(types.values()), "mapped_raw_source_lines":len(mapped),
            "input_kind":"cleaned", "protocol_deviation":"Package Config A declares raw input; operator explicitly selected cleaned input",
            "qa_used_for_construction":False}


def protocol_path(package):
    """Protocol file: $PMM_PROTOCOL overrides the package copy (endpoints differ per site)."""
    override = os.environ.get("PMM_PROTOCOL")
    return Path(override) if override else package/"protocol/config_A.json"


def prepare(package, output, user, smoke=False, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    protocol = load(protocol_path(package))
    if protocol["configuration"] != "A" or protocol["input_kind"] != "raw":
        raise ValueError("This runner requires Config A raw input")
    protocol = {**protocol,"configuration":"A-cleaned","input_kind":"cleaned",
                "input_policy":"Operator-selected complete upstream cleaned input; preserve source_lines; models from Config A",
                "base_protocol":"protocol/config_A.json"}
    validated = validate_user(package, user)
    formation, sage = settings_for(protocol)
    settings = (("config.json",formation),("sage_config.json",sage),("protocol.json",protocol))
    changes = []
    for name, data in settings:
        if (output/name).exists():
            changes += [c for c in setting_changes(load(output/name), data) if c not in changes]
    rebuild = [c for c in changes if c[0] in EMBEDDING_IDENTITY]
    if rebuild:
        raise ValueError("Embedding model or dimension changed ("
                         + "; ".join(f"{path}: {old!r} -> {new!r}" for path, old, new in rebuild)
                         + "); the vectors already stored cannot be reused. Rebuild in a new output directory")
    if changes:
        revisions = output/"settings_revisions.json"
        history = load(revisions) if revisions.exists() else []
        record = {"recorded_at":datetime.now(timezone.utc).isoformat(),
                  "changes":[{"setting":path, "from":old, "to":new} for path, old, new in changes]}
        write_json(revisions, [*history, record])
        emit({"stage":SETTINGS_CHANGED, "note":"Settings differ from the previous run; resuming with the new ones",
              **record})
    for name, data in settings:
        write_json(output/name, data)
    write_json(output/"input_validation.json", validated)
    source = package/validated["user"]["cleaned_observations"]
    limit = (limit or SMOKE_LIMIT) if smoke else None
    if not (output/"import_report.json").exists():
        with writer_lock(output/"memory.sqlite"), EventMemory(output/"memory.sqlite") as memory:
            report = import_jsonl(source, source_id="pmm-A-cleaned/"+user, user_id=user, memory=memory,
                                  input_format="pmm_cleaned", limit=limit)
            memory.set_metadata("import_report", report)
            memory.set_metadata("input_complete", not smoke)
            write_json(output/"import_report.json", report)
    else:
        report = load(output/"import_report.json")
        if report["config"]["limit"] != limit or Path(report["config"]["input"]) != source:
            raise ValueError(f"Run input or prefix changed (built with limit {report['config']['limit']}, "
                             f"now {limit}); use a new output directory")
    checked = verify_source(output, allow_prefix=smoke)
    write_json(output/"source_verification.json", checked)
    if checked["status"] != "passed":
        raise ValueError("Imported source verification failed")
    code = {str(p.relative_to(Path(__file__).parent)):file_hash(p) for p in Path(__file__).parent.rglob("*.py")}
    provenance = {"code_sha256":code, "protocol_sha256":file_hash(protocol_path(package)),
                  "input_sha256":validated["user"]["cleaned_sha256"], "input_kind":"cleaned", "smoke":smoke,
                  "centering":"all_selected_history_offline_not_strict_online",
                  "raw_qa_mapping":"raw-line:N is recovered from each cleaned record's _cleaning.source_lines"}
    if (output/"run_manifest.json").exists():
        # The protocol may change between runs (see settings_revisions.json); code and input may not.
        recorded = load(output/"run_manifest.json")
        if {k:v for k,v in recorded.items() if k != "protocol_sha256"} != {k:v for k,v in provenance.items() if k != "protocol_sha256"}:
            raise ValueError("Code/input changed since preparation; use a new run")
        provenance = recorded
    write_json(output/"run_manifest.json", provenance)
    emit({"stage":"import_complete", "run":str(output), "observations":report["accepted_observations"]})


def verify_run_code(run):
    manifest = load(run/"run_manifest.json")
    expected_code = manifest["code_sha256"]
    revision = run/"code_revision.json"
    if revision.exists():
        recorded = load(revision)
        if recorded["original_manifest_sha256"] != file_hash(run/"run_manifest.json"):
            raise ValueError("Code revision does not match original manifest")
        expected_code = recorded["code_sha256"]
    for relative, expected in expected_code.items():
        if file_hash(Path(__file__).parent/relative) != expected:
            raise ValueError("Code changed since preparation: " + relative)


def dred(run):
    """DRED fast path over the full observation stream, then its structural audit."""
    verify_run_code(run)
    started = time.monotonic()
    with writer_lock(run/"memory.sqlite"), EventMemory(run/"memory.sqlite") as memory:
        settings = load(run/"config.json")
        from hypermem.pmm_embeddings import prefetch, TextCacheEncoder
        def embedding_progress(value):
            write_json(run/"progress.json",value)
            if value["completed_texts"]%256==0 or value["completed_texts"]==value["unique_missing_texts"]:emit(value)
        prefetch(memory, settings, embedding_progress)
        engine = OnlineHyperMem(memory, settings, encoder=TextCacheEncoder(memory,settings["embedding"]))
        def progress(count, report):
            value = {"stage":"dred", "processed":count, "elapsed_seconds":time.monotonic()-started, **report}
            write_json(run/"progress.json",value)
            emit({"stage":"dred", "processed":count, "events":report["events"]})
        report = engine.replay(progress=progress, detailed_progress=False)
        del engine
        gc.collect()
        checked = audit(memory, require_complete=True)
        write_json(run/"dred_audit.json", checked)
        write_json(run/"dred_report.json", {**report,"elapsed_seconds":time.monotonic()-started})
        if checked["status"] != "passed":
            raise ValueError("DRED structural audit failed")
        memory._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    emit({"stage":"dred_complete", **report})


def require_llm(protocol):
    if not os.environ.get(protocol["api_key_env"]):
        raise RuntimeError("Missing " + protocol["api_key_env"] + "; no substitute model will be used")


def sage_boundary(run, *, retry_unresolved=False):
    """SAGE step 1: idle boundary consolidation of DRED events (adjudicate, split, stitch, event cards)."""
    def progress(**state):
        state.update(updated_at=datetime.now(timezone.utc).isoformat())
        write_json(run/"sage_progress.json",state)
        emit({"stage":"sage_boundary",**state})
    with writer_lock(run/"memory.sqlite"), EventMemory(run/"memory.sqlite") as memory:
        engine = OnlineHyperMem(memory,load(run/"config.json"))
        # Consolidation reads the persisted centered vectors (z); release the raw copies.
        engine.raw.clear()
        gc.collect()
        engine.run_slow(is_idle=True, retry_unresolved=retry_unresolved, progress=progress)
        result = memory.get_metadata("online_consolidation_last")
        del engine
        gc.collect()
        from hypermem.online.failures import boundary_failures
        try:
            skipped = boundary_failures(memory)
        except ValueError as exc:
            write_json(run/"sage_boundary_report.json", result)
            raise RuntimeError("SAGE boundary consolidation incomplete; inspect sage_boundary_report.json") from exc
        if skipped:
            result = {**result, "status":"complete_with_skipped_failures", "skipped_failures":skipped}
        write_json(run/"sage_boundary_report.json", result)
        checked = audit(memory, require_complete=True)
        write_json(run/"sage_boundary_audit.json", checked)
        if checked["status"] != "passed":
            raise RuntimeError("SAGE boundary structural audit failed; inspect sage_boundary_audit.json")
        memory._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    emit({"stage":"sage_boundary_complete", **result})


def sage(run, *, retry_unresolved=False):
    """SAGE slow path: boundary consolidation, then ToolSAGE, LongMemory and the frozen graph."""
    verify_run_code(run)
    require_llm(load(run/"protocol.json"))
    if not (run/"dred_audit.json").exists() or load(run/"dred_audit.json")["status"] != "passed":
        raise ValueError("Complete DRED before SAGE")
    boundary = run/"sage_boundary_audit.json"
    if retry_unresolved or not boundary.exists() or load(boundary)["status"] != "passed" or \
            load(run/"sage_boundary_report.json").get("status") not in FINISHED:
        sage_boundary(run, retry_unresolved=retry_unresolved)
    from hypermem.online.sage_tools import ToolSAGE
    with writer_lock(run/"memory.sqlite"), EventMemory(run/"memory.sqlite") as memory:
        # Connection-local page cache only; stored configuration and data are unchanged.
        memory._connection.execute("PRAGMA cache_size=-65536")
        def progress(value):
            write_json(run/"sage_progress.json",value)
            emit(value)
        tools = ToolSAGE(memory,load(run/"sage_config.json"),progress=progress)
        result = tools.run()
        write_json(run/"sage_report.json",result)
        write_json(run/"sage_tool_jobs.json",tools._jobs())
        if result["status"] not in FINISHED:
            raise RuntimeError("SAGE completion checks failed; QA is blocked. Inspect sage_report.json")
        memory._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    write_json(run/"frozen.json", {"memory_sha256":file_hash(run/"memory.sqlite"),
                                  "frozen_at":datetime.now(timezone.utc).isoformat(),
                                  "smoke":load(run/"run_manifest.json")["smoke"]})
    emit({"stage":"sage_complete", **result})


def without_vectors(value):
    if isinstance(value, dict):
        return {k:without_vectors(v) for k,v in value.items() if k not in {"embedding","prototype"}}
    if isinstance(value, list):
        return [without_vectors(v) for v in value]
    return value


def answer_context(retrieved, memory, *, top_k=30, char_limit=48000):
    """Rank native Event/LongMemory units; bounded, gold-blind source expansion.

    Each unit gets its fair share of the remaining character budget. At most
    three whole raw records per unit; omitted/oversized records are logged.
    Source coverage counts only raw text actually sent, never merely an ID.
    """
    records = []
    for event in retrieved["events"]:
        eid = event["event_id"]
        facts = without_vectors(event.get("facts", []))
        support = sorted({oid for f in facts for oid in f["support"]})
        if not support:
            support = memory.get_event(eid).observation_ids
        records.append({"id":eid,"kind":"event","score":retrieved["node_scores"][eid],
                        "title":event.get("title"),"summary":event.get("summary"),
                        "span":event.get("span"),"facts":facts,"support":support})
    for item in retrieved["long_memories"]:
        mid = item["memory_id"]
        records.append({"id":mid,"kind":"long_memory","score":retrieved["node_scores"][mid],
                        "text":item["text"],"support":item["evidence_ids"]})
    records.sort(key=lambda r:(-r["score"],r["id"]))
    records = records[:top_k]
    sections, sent, omitted = [], set(), []
    used = 0
    for i, record in enumerate(records):
        allocation = (char_limit-used)//max(1,len(records)-i)
        semantic = json.dumps({k:v for k,v in record.items() if k != "support"},ensure_ascii=False)
        header = f"\n[Memory {record['id']}]\n"
        # Deliberately account for truncation; never infer evidence from a cut ID.
        cap = max(0,min(len(semantic),allocation//2))
        text = header+semantic[:cap]+("\n[语义条目已截断]" if cap<len(semantic) else "")
        for oid in record["support"][:3]:
            observation = memory.get_observation(oid)
            source_ids = [f"raw-line:{x}" for x in observation.provenance.metadata.get("source_lines", [])]
            if not source_ids and observation.provenance.metadata.get("upstream_event_id"):
                source_ids = [observation.provenance.metadata["upstream_event_id"]]
            raw = f"\n[Source {','.join(source_ids)}; {observation.start_time.isoformat()}]\n{observation.content}\n"
            if len(text)+len(raw) <= allocation:
                text += raw
                sent.update(source_ids)
            else:
                omitted.append({"observation_id":oid,"reason":"whole_raw_record_exceeds_unit_budget"})
        if len(text) > allocation:
            text = text[:allocation]
        sections.append(text)
        used += len(text)
    return {"text":"".join(sections),"records":records,"top_k":top_k,
            "raw_expansion_max_per_record":3,"char_limit":char_limit,"chars":used,
            "sent_raw_source_ids":sorted(sent),"omitted_raw_records":omitted,
            "policy":"rank native Event/LongMemory by graph score; fair-share character budget; whole raw records only"}


def parse_judge(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated judge response")
    result = json.loads(choice["message"]["content"])
    if result.get("label") not in {"CORRECT","WRONG"} or not isinstance(result.get("reason"),str):
        raise ValueError("Invalid judge JSON")
    return result


def summarize(records, expected):
    def group(items, count):
        done = [r for r in items if r.get("status") == "complete"]
        return {"expected":count,"completed":len(done),"coverage":len(done)/count if count else None,
                "accuracy":sum(r["score"] for r in done)/len(done) if done else None,
                "f1":sum(r["f1"] for r in done)/len(done) if done else None,
                "abstentions":sum(r.get("prediction", "").strip().rstrip("。.!！?？").strip() == "证据不足，无法回答" for r in items),
                "errors":sum(r.get("status") == "failed" for r in items)}
    return {"overall":group(records,sum(expected.values())),
            "by_category":{c:group([r for r in records if r["category"] == c],n) for c,n in expected.items()}}


def evaluate(package, run, user, *, smoke_prefix=False):
    verify_run_code(run)
    protocol = load(run/"protocol.json")
    require_llm(protocol)
    frozen = load(run/"frozen.json")
    if frozen["smoke"] and not smoke_prefix:
        raise ValueError("Official QA may not run against a smoke prefix")
    if file_hash(run/"memory.sqlite") != frozen["memory_sha256"]:
        raise ValueError("Frozen memory changed")
    # Shared with every baseline through the pmm_bench harness.
    from pmm_bench.prompts import answer_messages, judge_messages
    from pmm_bench.llm import normalize_request
    from pmm_bench.data import chinese_f1, load_questions
    from hypermem.memory.readonly import memory_view
    from hypermem.online.hyperos_retrieval import HyperOSRetriever
    questions = load_questions(package/"users"/user)
    scope = "A-cleaned-prefix1000" if frozen["smoke"] else "A-cleaned"
    destination = run.parent/"results"/scope/"hyperos-mem"/user
    destination.mkdir(parents=True,exist_ok=True)
    contract = {"memory_sha256":frozen["memory_sha256"],"qa_sha256":file_hash(package/"users"/user/"qa.json"),
                "scope":scope,"official_full_history_score":not frozen["smoke"],
                "protocol":protocol,"retrieval_adapter":"native-events-longmem-fairshare-raw3-v1",
                "evidence_coverage_scope":"source-line provenance represented by cleaned observations, not raw-content equivalence"}
    skipped = load(run/"sage_report.json").get("skipped_failures",{})
    if skipped.get("boundary") or skipped.get("sage"):
        sage_skipped = skipped.get("sage",[])
        contract["skipped_construction_failures"] = {"boundary_tasks":len(skipped.get("boundary",[])),
            "sage_event_jobs":sum(f["stage"]=="event_tools" for f in sage_skipped),
            "long_memory_tasks":sum(f["stage"]=="long_memory" for f in sage_skipped)}
    if (destination/"contract.json").exists() and load(destination/"contract.json") != contract:
        raise ValueError("Evaluation contract changed; use new results")
    write_json(destination/"contract.json",contract)
    conn = sqlite3.connect((run/"memory.sqlite").resolve().as_uri()+"?mode=ro",uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("BEGIN")
    memory = memory_view(conn)
    cfg = load(run/"sage_config.json")
    retriever = HyperOSRetriever(SimpleNamespace(m=memory,cfg=cfg,encoder=make_encoder(cfg["embedding"]),
                    center=memory.get_metadata("online_center"),events={e.event_id:e for e in memory.iter_events()}))
    service = {"base_url":protocol["api_base_url"],"path":"/chat/completions",
               "api_key_env":protocol["api_key_env"],"timeout":180,"retries":2}
    results = []
    try:
        for q in questions:
            name = hashlib.sha256(q.query_id.encode()).hexdigest()[:20]+".json"
            path = destination/name
            record = load(path) if path.exists() else {"query_id":q.query_id,"category":q.category,"question":q.question}
            if record.get("status") == "complete":
                results.append(record)
                continue
            record["attempts"] = record.get("attempts",0)+1
            started = time.monotonic()
            try:
                if "retrieval" not in record:
                    retrieved = retriever.retrieve(q.question)
                    record["retrieval"] = without_vectors(retrieved)
                    record["context"] = answer_context(retrieved,memory,top_k=protocol["top_k"],char_limit=protocol["context_chars"])
                    write_json(path,record)
                if "prediction" not in record:
                    request = normalize_request({"model":protocol["answer_model"],
                        "messages":answer_messages(SimpleNamespace(question=q.question,category=q.category),record["context"]["text"]),
                        "max_tokens":protocol["answer_max_output_tokens"],"temperature":0})
                    record["answer_request"] = request
                    write_json(path,record)
                    response = request_json(service,request)
                    record["answer_response"] = response
                    write_json(path,record)
                    choice = response["choices"][0]
                    if choice.get("finish_reason") == "length" or not choice["message"].get("content", "").strip():
                        raise ValueError("Empty or truncated answer")
                    record["prediction"] = choice["message"]["content"].strip()
                    write_json(path,record)
                judge_request = normalize_request({"model":protocol["judge_model"],
                    "messages":judge_messages(q,record["prediction"]),"max_tokens":protocol["judge_max_output_tokens"],"temperature":0})
                record["judge_request"] = judge_request
                write_json(path,record)
                response = request_json(service,judge_request)
                record["judge_response"] = response
                write_json(path,record)
                judged = parse_judge(response)
                gold = {x.replace("line:","raw-line:",1) if x.startswith("line:") else x for x in q.evidence_event_ids}
                sent = set(record["context"]["sent_raw_source_ids"])
                record.update(status="complete",judge=judged,score=int(judged["label"]=="CORRECT"),
                              f1=chinese_f1(record["prediction"],q.answers),
                              evidence={"gold_source_ids":sorted(gold),"sent_raw_hits":sorted(gold & sent),
                                        "sent_mapped_source_recall":len(gold & sent)/len(gold) if gold else None,
                                        "scope":"cleaned observation source-line provenance; not proof cleaned text retained every raw fact"})
            except Exception as exc:
                record.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)[:250]})
            record["elapsed_seconds"] = record.get("elapsed_seconds",0)+time.monotonic()-started
            write_json(path,record)
            results.append(record)
            write_json(destination/"summary.json",summarize(results,Counter(q.category for q in questions)))
            emit({"query_id":q.query_id,"status":record["status"],"score":record.get("score")})
    finally:
        conn.close()
    summary = summarize(results,Counter(q.category for q in questions))
    summary["memory_unchanged"] = file_hash(run/"memory.sqlite") == frozen["memory_sha256"]
    summary.update(scope=scope,official_full_history_score=not frozen["smoke"])
    write_json(destination/"summary.json",summary)
    emit(summary)


def dashboard(run):
    from hypermem.online.sage_dashboard import snapshot_graph, export_dashboard
    graph = snapshot_graph(run/"memory.sqlite")
    user = load(run/"import_report.json")["config"]["user_id"]
    graph.update(user_id=user,title=user.split("_",1)[0]+" · 层级记忆图")
    destination = run if load(run/"run_manifest.json")["smoke"] else run.parent
    write_json(destination/"hypermem_dashboard.json",graph)
    export_dashboard(graph,destination/"hypermem_dashboard.html")
    emit({"stage":"dashboard", **graph["summary"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--user",default=DEFAULT_USER)
    p.add_argument("--stage",choices=STAGES+("all",),required=True)
    p.add_argument("--smoke",action="store_true",help="prefix build (first 1,000 observations, or --limit); builds memory but skips QA")
    p.add_argument("--limit",type=int,help="with --smoke: number of observations to build from")
    p.add_argument("--record-code-revision",action="store_true",help="Journal a code revision without rewriting the original run manifest")
    p.add_argument("--revision-reason",help="Explicit reason required when journalling a DRED/SAGE resume revision")
    p.add_argument("--retry-unresolved",action="store_true",help="sage only: grant one more attempt to each exhausted boundary-consolidation task")
    p.add_argument("--env-file",type=Path,help="KEY=VALUE file loaded before model calls; values are never logged")
    a = p.parse_args()
    if a.user is None and a.stage in {"prepare","evaluate","all"}:
        p.error("--user is required for prepare, evaluate and all")
    if a.retry_unresolved and a.stage not in {"sage","all"}:
        p.error("--retry-unresolved requires --stage sage or all")
    if a.limit is not None and (not a.smoke or a.limit < 1):
        p.error("--limit requires --smoke and a positive number")
    if a.env_file:
        for number,line in enumerate(a.env_file.read_text(encoding="utf-8-sig").splitlines(),1):
            line=line.strip()
            if not line or line.startswith("#"):continue
            if "=" not in line:raise ValueError(f"Invalid env file line {number}")
            key,value=line.split("=",1)
            if key.strip() != "OPENAI_API_KEY":raise ValueError("Env file may contain only OPENAI_API_KEY")
            value=value.strip()
            if not value:raise ValueError("OPENAI_API_KEY is empty")
            os.environ[key.strip()]=value
    run = a.output/("smoke" if a.smoke else "build")
    run.mkdir(parents=True,exist_ok=True)
    if a.record_code_revision:
        if a.stage in {"dred","sage"} and not (a.revision_reason or "").strip():
            raise ValueError("A DRED/SAGE resume revision requires --revision-reason")
        if a.stage not in {"dred","sage","evaluate","dashboard"}:
            raise ValueError("Code revisions can only be journalled for dred, sage, evaluate or dashboard")
        current = {str(p.relative_to(Path(__file__).parent)):file_hash(p) for p in Path(__file__).parent.rglob("*.py")}
        prior = load(run/"code_revision.json") if (run/"code_revision.json").exists() else None
        write_json(run/"code_revision.json",{"original_manifest_sha256":file_hash(run/"run_manifest.json"),
                   "code_sha256":current,"previous_revision":prior,"recorded_at":datetime.now(timezone.utc).isoformat(),
                   "reason":a.revision_reason or "Journalled evaluation/export revision; DRED/SAGE results retained"})
    os.environ["HYPERMEM_REQUEST_LOG"] = str(run/"model_requests.jsonl")
    stages = STAGES[:-1] if a.smoke else STAGES
    try:
        for stage in stages if a.stage == "all" else (a.stage,):
            write_json(run/"stage.json",{"stage":stage,"status":"running"})
            if stage == "prepare":prepare(a.package,run,a.user,a.smoke,a.limit)
            elif stage == "dred":dred(run)
            elif stage == "sage":sage(run,retry_unresolved=a.retry_unresolved)
            elif stage == "dashboard":dashboard(run)
            elif stage == "evaluate":evaluate(a.package,run,a.user,smoke_prefix=a.smoke)
            write_json(run/"stage.json",{"stage":stage,"status":"complete"})
    except Exception as exc:
        write_json(run/"stage.json",{"stage":stage,"status":"failed","error_type":type(exc).__name__,"error":str(exc)[:400]})
        raise


if __name__ == "__main__":
    main()
