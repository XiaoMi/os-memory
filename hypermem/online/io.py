# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Structural audit of a DRED/SAGE memory and atomic JSON writes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from hypermem.grounding import digest, ground
from hypermem.schema import AtomicObservation, CanonicalEvent, ObservationFeatures
from hypermem.schema.serialization import encode_json
from hypermem.standardization import observation_text
from hypermem.online.clustering import unit, similarity, cohesion


def write_json(path,value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def audit(memory,require_complete=False):
    """Checks source-derived vectors, assignment conservation and update traces.

This is a structural audit, not a semantic accuracy evaluation. It never constructs an engine or calls f.
"""
    db = memory._connection
    cfg = memory.get_metadata("config")
    center = memory.get_metadata("online_center")
    problems = []
    def check(condition,code,identifier=None):
        if not condition:
            problems.append({"code":code,"id":identifier})
    states = {r[0]:json.loads(r[1]) for r in db.execute("SELECT * FROM on_states")}
    runtime = {}
    for event_id, encoded in db.execute("SELECT * FROM on_events"):
        value = json.loads(encoded)
        # Prototypes are large numeric vectors. Keeping their JSON-decoded
        # Python-float lists alive multiplies memory use during a full audit.
        # The audit treats them numerically everywhere, so compact them at
        # load time instead of retaining millions of boxed floats.
        value["prototype"] = np.asarray(value["prototype"])
        runtime[event_id] = value
    obs = {o.observation_id:o for o in memory.iter_observations()}
    raw_ids = {r[0] for r in db.execute("SELECT observation_id FROM on_vectors")}
    check(not list(db.execute("PRAGMA foreign_key_check")),"foreign_key_violation")
    check(center is not None,"missing_center")
    if not center:
        return {"status":"failed","complete":False,"issue_count":len(problems),"issues":problems}
    mean = np.asarray(center["mean"])
    if cfg["online"]["mean_mode"]=="corpus":
        check(raw_ids==set(obs),"corpus_input_coverage")
        check(center["calibration_ids_sha256"]==digest(sorted(raw_ids)),"calibration_scope_changed")
        check(center["calibration_count"]==len(raw_ids),"calibration_count")
        total = np.zeros_like(mean)
        for encoded, in db.execute("SELECT vector_json FROM on_vectors"):
            total += np.asarray(json.loads(encoded))
        check(np.allclose(mean,total/max(1,len(raw_ids)),atol=1e-10),"mean_not_input_mean")
    else:
        check(center["mean"]==cfg["online"]["fixed_mean"],"fixed_mean_changed")
    z = {}
    for oid,o in obs.items():
        check(o.provenance.raw_id==db.execute("SELECT raw_id FROM observations WHERE observation_id=?",(oid,)).fetchone()[0],"raw_reference",oid)
        f = memory.get_features(oid)
        if oid not in raw_ids or f is None:
            check(not require_complete,"missing_vector",oid)
            continue
        regrounded = ground(o,aliases=cfg["standardization"]["entity_aliases"],max_chars=cfg["standardization"]["max_text_chars"])
        if not regrounded["text"]:
            regrounded["text"] = observation_text(o,cfg["standardization"]["max_text_chars"])
            regrounded["online_text_fallback"] = "supplied_activity_label" if regrounded["text"] else "empty"
        check(f.metadata["grounding"]==regrounded,"grounding_not_reproducible",oid)
        encoded = db.execute(
            "SELECT vector_json FROM on_vectors WHERE observation_id=?", (oid,)).fetchone()[0]
        expected = unit(np.asarray(json.loads(encoded))-mean) if regrounded["text"] else np.zeros_like(mean)
        check(np.allclose(expected,f.embedding,atol=1e-10),"centered_embedding_mismatch",oid)
        check(f.metadata["centering_digest"]==digest(center),"centering_version_mismatch",oid)
        check(f.metadata["encoder"]==cfg["embedding"],"encoder_config_mismatch",oid)
        z[oid] = np.asarray(f.embedding)
    ownership = dict(db.execute("SELECT observation_id,event_id FROM event_observations"))
    event_members = defaultdict(list)
    for oid,eid in ownership.items():event_members[eid].append(oid)
    check(set(ownership)<=set(obs),"ownership_unknown_observation")
    check(set(states)<=set(obs),"state_unknown_observation")
    check(set(runtime)=={e.event_id for e in memory.iter_events()},"runtime_event_coverage")
    for oid,s in states.items():
        check(s["status"] in {"assigned","pending"},"unknown_disposition",oid)
        check(ownership.get(oid)==s["event_id"],"assignment_state_mismatch",oid)
        check((s["status"]=="assigned")==bool(s["event_id"]),"invalid_disposition",oid)
        check(s.get("mainline_id") is None or s["mainline_id"] in runtime,"unknown_mainline",oid)
        if s["event_id"]:
            check(runtime[s["event_id"]]["lifecycle"]!="superseded","assigned_to_retired_event",oid)
        latest = memory.last_decision(oid)
        check(latest is not None and latest.get("selected_event")==s["event_id"],"latest_decision_assignment",oid)
    for eid,r in runtime.items():
        e = memory.get_event(eid)
        members = event_members[eid]
        if r["lifecycle"]=="superseded":
            check(not members,"retired_event_owns_members",eid)
            check(db.execute("SELECT 1 FROM on_splits WHERE parent_id=?",(eid,)).fetchone() is not None,"retired_without_split",eid)
        else:
            check(set(e.observation_ids)==set(members),"event_member_projection",eid)
            check(set(r["core_ids"])<=set(members),"prototype_source_outside_event",eid)
            check(all(states[oid]["role"]!="incidental" for oid in r["core_ids"]),"incidental_defines_prototype",eid)
        check(np.allclose(e.embedding,r["prototype"],atol=1e-10),"prototype_projection",eid)
        check(abs(np.linalg.norm(r["prototype"])-1)<1e-9 or np.linalg.norm(r["prototype"])<1e-12,"prototype_not_normalized",eid)
        check(all(e.start_time<=obs[oid].start_time and obs[oid].end_time<=e.end_time for oid in e.observation_ids),"event_time_coverage",eid)
    prototype_chain = {}
    for row in db.execute("SELECT observation_id,audit_json FROM decision_log ORDER BY sequence"):
        oid,t = row[0],json.loads(row[1])
        selected = t.get("selected_event")
        if t["formation_decision"] in {"MATCH","ATTACH"}:
            before,after,alpha = np.asarray(t["prototype_before"]),np.asarray(t["prototype_after"]),t["alpha"]
            check(selected in prototype_chain and np.allclose(before,prototype_chain[selected],atol=1e-10),"prototype_update_chain",oid)
            expected = before if t["formation_decision"]=="ATTACH" else unit((1-alpha)*before+alpha*z[oid])
            check(np.allclose(after,expected,atol=1e-10),"prototype_update_trace",oid)
            if t["formation_decision"]=="ATTACH":
                check(alpha==0,"incidental_alpha_not_zero",oid)
            elif t["reason"]=="semantic":
                check(similarity(z[oid],before)>=cfg["online"]["tau_a"]-1e-10,"below_assignment_threshold",oid)
            elif t["reason"]=="event_identifier":
                check(bool(t["evidence"]["matched_keys"]),"symbolic_match_without_key",oid)
            elif t["reason"]=="entity_identifier":
                check(bool(t["evidence"]["matched_keys"]),"entity_match_without_key",oid)
                check(similarity(z[oid],before)>=cfg["online"]["tau_s"]-1e-10,"below_entity_threshold",oid)
            prototype_chain[selected] = after
        elif t["formation_decision"]=="NEW":
            ids = t["member_ids"]
            expected = unit(np.sum([z[i] for i in ids],axis=0))
            check(np.allclose(t["prototype_after"],expected,atol=1e-10),"initial_prototype_not_member_mean",oid)
            prototype_chain[selected] = expected
            if len(ids)>1 and t["reason"]=="pending_confirmation":
                check(similarity(z[ids[0]],z[ids[1]])>=cfg["online"]["tau_p"]-1e-10,"below_confirmation_threshold",oid)
                check(abs((obs[ids[0]].start_time-obs[ids[1]].start_time).total_seconds())<=60*cfg["online"]["pending_minutes"],"confirmation_after_deadline",oid)
        elif t["formation_decision"]=="SPLIT":
            split = json.loads(db.execute("SELECT trace_json FROM on_splits WHERE parent_id=?",(t["parent_id"],)).fetchone()[0])
            # Immutable split lineage records include all members at split time.
            relation = db.execute("SELECT evidence_json FROM on_relations WHERE source=? AND target=? AND kind='split_from'",(selected,t["parent_id"])).fetchone()
            ids = json.loads(relation[0])["member_ids"]
            clustered = set(split["clustered_observation_ids"])
            expected = unit(np.mean([z[i] for i in ids if i in clustered],axis=0))
            check(np.allclose(t["prototype_after"],expected,atol=1e-10),"split_prototype_not_cluster_mean",oid)
            prototype_chain[selected] = expected
    for eid,r in runtime.items():
        check(eid in prototype_chain and np.allclose(r["prototype"],prototype_chain[eid],atol=1e-10),"runtime_prototype_not_latest_trace",eid)
    for source,target,kind,_ in db.execute("SELECT * FROM on_relations"):
        check(source in runtime and target in runtime and source!=target,"relation_endpoints",source)
    for parent,encoded in db.execute("SELECT * FROM on_splits"):
        t = json.loads(encoded)
        parent_members = set(t["historical_event"]["observation_ids"])
        # Child events can gain later members; use immutable split-time edges.
        flattened = [oid for eid in t["child_ids"] for oid in json.loads(db.execute(
            "SELECT evidence_json FROM on_relations WHERE source=? AND target=? AND kind='split_from'",(eid,parent)).fetchone()[0])["member_ids"]]
        check(len(flattened)==len(set(flattened)) and set(flattened)==parent_members,"split_member_conservation",parent)
        if t.get("method")=="sage_tool_partition":
            job=db.execute("SELECT record_json FROM sage_tool_jobs WHERE job_id=?",(t.get("job_id"),)).fetchone()
            job=json.loads(job[0]) if job else {}
            check(job.get("status")=="complete" and job.get("split_children")==t["child_ids"],"uncommitted_sage_split",parent)
            groups=t.get("groups",[])
            check(len(groups)>=2 and sorted(oid for g in groups for oid in g)==sorted(flattened),"sage_split_partition",parent)
        else:
            check(t["cohesion_after"]-t["cohesion_before"]>=cfg["online"]["split_min_gain"]-1e-10,"split_insufficient_gain",parent)
    from hypermem.online.consolidation import audit_consolidation
    audit_consolidation(memory,states,runtime,check)
    complete = len(states)==len(obs) and all(s["status"]=="assigned" for s in states.values()) and memory.get_metadata("online_complete",False)
    check(not require_complete or complete,"incomplete_run")
    return {"schema":"hypermem-online-audit-v1","status":"passed" if not problems else "failed",
            "complete":complete,"observations_checked":len(obs),"events_checked":len(runtime),
            "issue_count":len(problems),"issues":problems[:100],
            "scope":"centering, source-grounding, assignments, prototypes, decision gates, split conservation; not semantic accuracy"}
