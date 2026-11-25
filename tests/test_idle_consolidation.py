# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_online_semantic import Encoder, observation, settings, START
from hypermem.config import config
from hypermem.grounding import digest
from hypermem.memory import EventMemory
from hypermem.online.engine import OnlineHyperMem as HyperMem
from hypermem.online.consolidation import Consolidation, record, records, save, consolidated_view, validate, semantic_version, semantic_versions, dirty_semantics
from hypermem.online.io import audit


def model(payload):
    cards = payload["events"]
    ids = [o["obs_id"] for c in cards for o in c["member_summaries"]]
    if payload["stage"]=="adjudicate":
        return {"action":"isolate","target_event":None,"confidence":.7,
                "reason":"未见同一活动的充分证据。","evidence_ids":[payload["observation"]["obs_id"]]}
    if payload["stage"]=="stitch":
        return {"same_activity":True,"confidence":.9,"reason":"成员描述同一活动续接。","evidence_ids":ids}
    return {"title":"测试活动","summary":"根据所给成员原文生成的活动摘要。",
            "stable_keys":sorted({k for c in cards for o in c["member_summaries"] for k in o["keys"]}),
            "start_time":cards[0]["start_time"],"end_time":cards[0]["end_time"],"evidence_ids":ids}


class IdleConsolidationTests(unittest.TestCase):
    def engine(self,rows,*,options=None,**cfg):
        m = EventMemory()
        self.addCleanup(m.close)
        vectors = {}
        for row in rows:
            oid,v,second,*order = row
            vectors[oid] = v
            o = observation(oid,oid,second,order=order[0] if order else None)
            m.add_observation(o,{"input":o.to_dict()})
        return HyperMem(m,settings(consolidation={"enabled":True,**(options or {})},**cfg),
                        encoder=Encoder(vectors),slow_resolver=model)

    def checked(self,e):
        result = audit(e.memory,True)
        self.assertEqual(result["status"],"passed",result)

    def baseline(self,e):
        return {"states":copy.deepcopy(e.states),"runtime":copy.deepcopy(e.runtime),
                "events":[x.to_dict() for x in e.memory.iter_events()],
                "raw":[e.memory.resolve_provenance(o.observation_id) for o in e.memory.iter_observations()]}

    def test_fast_restores_full_fallback_without_any_llm(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),("x",[0,1,0],2),("y",[0,0,1],1200)])
        e.slow_resolver = lambda _:self.fail("Fast cannot call LLM")
        e.replay()
        self.assertEqual(e.states["x"]["role"],"incidental")
        self.assertEqual(e.states["y"]["role"],"isolated")
        self.assertTrue(all(s["status"]=="assigned" for s in e.states.values()))
        self.assertEqual(e.runtime[e.states["a"]["event_id"]]["prototype"],[1.,0.,0.])
        self.assertEqual(e.report()["consolidation"]["model_attempts"],0)
        self.checked(e)

    def test_keyed_record_lookup_does_not_decode_other_tasks(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        db = e.memory._connection
        with e.memory.transaction():
            save(e.memory,"on_consolidation_tasks","wanted",{"status":"responded","response":{"value":1}})
            db.execute("INSERT INTO on_consolidation_tasks VALUES (?,?)",("unrelated","not-json"))
        queries=[]
        db.set_trace_callback(queries.append)
        try:
            self.assertEqual(record(e.memory,"on_consolidation_tasks","wanted"),
                             {"status":"responded","response":{"value":1}})
            default=object()
            self.assertIs(record(e.memory,"on_consolidation_tasks","missing",default),default)
            with self.assertRaises(ValueError):record(e.memory,"unknown_table","wanted")
        finally:
            db.set_trace_callback(None)
        self.assertEqual(len(queries),2)
        self.assertTrue(all("WHERE task_id=" in q for q in queries))
        with self.assertRaises(json.JSONDecodeError):records(e.memory,"on_consolidation_tasks")

    def test_already_invalidated_cards_do_not_reload_all_event_vectors(self):
        e=self.engine([("a",[1,0,0],0),("b",[0,1,0],1200)]);e.replay()
        before=records(e.memory,"on_event_wiki")
        with patch("hypermem.online.consolidation.semantic_versions",side_effect=AssertionError("Unnecessary vector scan")):
            dirty_semantics(e.memory)
        self.assertEqual(records(e.memory,"on_event_wiki"),before)
        self.checked(e)

    def test_wiki_invalidation_preserves_original_status_rules(self):
        e=self.engine([("a",[1,0,0],0)]);e.replay()
        eid,base=next(iter(records(e.memory,"on_event_wiki").items()))
        for title in (None,"Existing grounded summary"):
            for status in ("ready","stale","pending"):
                for fingerprint in ("current","old"):
                    value={**base,"title":title,"semantic_status":status,"semantic_fingerprint":fingerprint}
                    save(e.memory,"on_event_wiki",eid,value)
                    expected=value if fingerprint=="current" else {**value,"semantic_status":"stale" if title else "pending"}
                    with patch("hypermem.online.consolidation.semantic_versions",return_value={eid:"current"}):
                        dirty_semantics(e.memory)
                    self.assertEqual(record(e.memory,"on_event_wiki",eid),expected)

    def test_batched_keyed_reads_match_bulk_reads_and_resume_without_calls(self):
        def batch_model(payload):
            if payload["stage"]=="adjudicate_batch":
                return {"decisions":[{"item_id":item["item_id"],**model({**item,"stage":"adjudicate"})}
                                     for item in payload["items"]]}
            if payload["stage"]=="wiki_batch":
                return {"cards":[{"event_id":item["event_id"],**model({"stage":"wiki","events":[item["event"]]})}
                                 for item in payload["items"]]}
            return model(payload)

        rows=[("a",[1,0,0],0),("b",[1,0,0],1),("q",[0,1,0],1200),
              ("r",[0,0,1],2400),("s",[-1,0,0],3600)]
        def run(use_bulk):
            e=self.engine(rows,options={"batch_size":2,"concurrency":2})
            e.replay()
            original=self.baseline(e)
            e.slow_resolver=batch_model
            if use_bulk:
                with patch("hypermem.online.consolidation.record",side_effect=lambda m,t,k,default=None:records(m,t).get(k,default)), \
                     patch("hypermem.online.consolidation.semantic_version",side_effect=lambda m,k:semantic_versions(m)[k]):
                    result=e.run_slow(is_idle=True)
            else:
                queries=[]
                e.memory._connection.set_trace_callback(queries.append)
                try:result=e.run_slow(is_idle=True)
                finally:e.memory._connection.set_trace_callback(None)
                self.assertTrue(any(q.startswith("SELECT record_json FROM on_consolidation_tasks WHERE task_id=") for q in queries))
                self.assertTrue(any(q.startswith("SELECT record_json FROM on_event_wiki WHERE event_id=") for q in queries))
            self.assertEqual(result["status"],"complete",result)
            self.assertEqual(original,self.baseline(e))
            self.checked(e)
            semantic_map=semantic_versions(e.memory)
            self.assertEqual(semantic_map,{eid:semantic_version(e.memory,eid) for eid in e.runtime})
            tasks=records(e.memory,"on_consolidation_tasks")
            self.assertEqual({t["input"]["stage"] for t in tasks.values()}, {"adjudicate_batch","wiki_batch"})
            self.assertTrue(all(t["status"]=="complete" for t in tasks.values()))
            e.slow_resolver=lambda _:self.fail("Completed batches must not call the model on resume")
            self.assertEqual(e.run_slow(is_idle=True)["status"],"complete")
            self.checked(e)
            return {"baseline":self.baseline(e),"wiki":records(e.memory,"on_event_wiki"),
                    "relations":records(e.memory,"on_soft_relations"),
                    "tasks":{tid:{k:v for k,v in t.items() if k!="model_seconds"} for tid,t in tasks.items()}}
        self.assertEqual(run(False),run(True))

    def test_idle_signal_is_required_and_interruptible(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        self.assertEqual(e.run_slow()["status"],"paused_not_idle")
        self.assertFalse(records(e.memory,"on_consolidation_tasks"))
        idle = [True]
        def resolver(payload):
            idle[0] = False
            return model(payload)
        e.slow_resolver = resolver
        self.assertEqual(e.run_slow(is_idle=lambda:idle[0])["status"],"paused_not_idle")
        task = next(iter(records(e.memory,"on_consolidation_tasks").values()))
        self.assertEqual(task["status"],"responded")
        self.assertFalse(records(e.memory,"on_soft_relations"))
        e.slow_resolver = model
        e.run_slow(is_idle=True)
        self.checked(e)

    def test_consolidation_resume_summary_and_audit_never_bulk_load_task_journal(self):
        e=self.engine([("a",[1,0,0],0),("b",[1,0,0],1),("q",[0,1,0],1200)],
                      options={"concurrency":2})
        e.replay()
        def guarded(memory,table):
            self.assertNotEqual(table,"on_consolidation_tasks","task journal must be streamed or keyed")
            return records(memory,table)
        progress=[]
        with patch("hypermem.online.consolidation.records",side_effect=guarded):
            self.assertEqual(e.run_slow(is_idle=True,progress=lambda **s:progress.append(s))["status"],"complete")
            self.checked(e)
            e.slow_resolver=lambda _:self.fail("resume must reuse persisted results")
            self.assertEqual(e.run_slow(is_idle=True)["status"],"complete")
            self.checked(e)
        self.assertIn("stitch",{s["phase"] for s in progress})
        self.assertIn("summary",{s["phase"] for s in progress})




    def test_low_margin_records_without_defer(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),
                         ("c",[0,1,0],2),("d",[0,1,0],3),("q",[1,1,0],4)])
        e.replay()
        u = records(e.memory,"on_uncertainty")
        self.assertEqual(set(u),{"q"})
        self.assertAlmostEqual(u["q"]["margin"],0)
        self.assertEqual(e.states["q"]["status"],"assigned")
        self.checked(e)

    def test_event_identifier_tier_precedes_closer_active_semantic_candidate(self):
        e = self.engine([("a",[1,0,0],0,"order-a"),("b",[1,0,0],1,"order-a"),
                         ("c",[0,1,0],2),("d",[0,1,0],3),("q",[0,1,0],4,"order-a")])
        e.replay()
        self.assertEqual(e.states["q"]["event_id"],e.states["a"]["event_id"])
        self.checked(e)

    def test_soft_assign_and_revoke_never_change_hard_membership(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),("q",[0,1,0],1200)])
        e.replay()
        before = self.baseline(e)
        target = e.states["a"]["event_id"]
        def resolver(p):
            if p["stage"]=="adjudicate":
                return {"action":"assign","target_event":target,"confidence":.6,"reason":"跨源活动联系。","evidence_ids":["q","a"]}
            return model(p)
        e.slow_resolver = resolver
        e.run_slow(is_idle=True)
        self.assertEqual(before,self.baseline(e))
        self.assertEqual(len(consolidated_view(e.memory)["events"]),1)
        rid = next(iter(records(e.memory,"on_soft_relations")))
        e.revoke_soft_relation(rid,"人工核对后撤销")
        self.assertEqual(len(consolidated_view(e.memory)["events"]),2)
        e.run_slow(is_idle=True)
        self.assertEqual(records(e.memory,"on_soft_relations")[rid]["status"],"revoked")
        self.assertEqual(e.report()["consolidation"]["wiki_pending"],0)
        self.checked(e)

    def test_uncertain_isolate_is_only_a_virtual_event(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),
                         ("c",[0,1,0],2),("d",[0,1,0],3),("q",[1,1,0],4)])
        e.replay()
        before = self.baseline(e)
        e.run_slow(is_idle=True)
        self.assertEqual(before,self.baseline(e))
        groups = consolidated_view(e.memory)["events"]
        self.assertTrue(any(g["id"]=="V-isolate-q" and g["observation_ids"]==["q"] for g in groups))
        self.checked(e)

    def test_history_recall_includes_archived_and_dormant_events(self):
        e = self.engine([("a",[1,0,0],0,"order-a"),("b",[1,0,0],1,"order-a"),
                         ("q",[0,1,0],1200)],options={"archive_days":.001})
        e.replay()
        target = e.states["a"]["event_id"]
        e.run_slow(is_idle=True)
        self.assertEqual(e.runtime[target]["lifecycle"],"archived")
        self.assertNotIn(target,set().union(*e.strong_index.values()) if e.strong_index else set())
        self.assertIn(target,Consolidation(e).candidates("q"))
        for oid,second in (("new1",1201),("new2",1202)):
            o = observation(oid,oid,second,order="order-a")
            e.encoder.vectors[oid] = [1,0,0]
            e.memory.add_observation(o,{"input":o.to_dict()})
            e.process(o)
        self.assertNotEqual(e.states["new2"]["event_id"],target)
        e.finalize()
        self.checked(e)

    def test_stitch_only_keys_and_inactive_gap_no_hard_merge(self):
        e = self.engine([("与小王聊天甲",[1,0,0],0),("与小王聊天乙",[1,0,0],1),
                         ("与小王聊天丙",[0,1,0],1200),("与小王聊天丁",[0,1,0],1201)])
        e.replay()
        before = self.baseline(e)
        e.run_slow(is_idle=True)
        rel = list(records(e.memory,"on_soft_relations").values())
        self.assertEqual([r["kind"] for r in rel],["soft_stitch"])
        self.assertEqual(before,self.baseline(e))
        self.assertEqual(len(consolidated_view(e.memory)["events"]),1)
        self.checked(e)

    def test_archiving_does_not_repeat_completed_stitch_on_resume(self):
        e = self.engine([("与小王聊天甲",[1,0,0],0),("与小王聊天乙",[1,0,0],1),
                         ("与小王聊天丙",[0,1,0],1200),("与小王聊天丁",[0,1,0],1201)],
                        options={"archive_days":.001,"concurrency":2})
        e.replay()
        self.assertEqual(e.run_slow(is_idle=True)["status"],"complete")
        before=copy.deepcopy(records(e.memory,"on_consolidation_tasks"))
        self.assertTrue(any(r["lifecycle"]=="archived" for r in e.runtime.values()))
        e.slow_resolver=lambda _:self.fail("Archiving alone must not repeat a completed model task")
        self.assertEqual(e.run_slow(is_idle=True)["status"],"complete")
        self.assertEqual(before,records(e.memory,"on_consolidation_tasks"))
        self.checked(e)

    def test_task_identity_ignores_only_card_lifecycle(self):
        card={"event_id":"E1","lifecycle":"dormant","member_summaries":[{"obs_id":"o1","text":"source"}]}
        payloads=[{"stage":"stitch","events":[card],"versions":{"E1":"v1"}},
                  {"stage":"adjudicate_batch","items":[{"events":[card]}],"versions":{"E1":"v1"}},
                  {"stage":"wiki_batch","items":[{"event":card,"semantic_version":"s1"}],"versions":{"E1":"v1"}}]
        for payload in payloads:
            with self.subTest(stage=payload["stage"]):
                changed=copy.deepcopy(payload)
                target=(changed["events"][0] if "events" in changed else
                        changed["items"][0].get("event") or changed["items"][0]["events"][0])
                target["lifecycle"]="archived"
                self.assertEqual(Consolidation.task_identity(payload),Consolidation.task_identity(changed))
                target["member_summaries"][0]["text"]="different evidence"
                self.assertNotEqual(Consolidation.task_identity(payload),Consolidation.task_identity(changed))
                changed=copy.deepcopy(payload)
                changed["versions"]["E1"]="v2"
                self.assertNotEqual(Consolidation.task_identity(payload),Consolidation.task_identity(changed))

    def test_archived_failed_batches_keep_budget_and_original_input(self):
        e=self.engine([("a",[1,0,0],0),("b",[0,1,0],1200),
                       ("c",[0,0,1],2400),("d",[-1,0,0],3600)],
                      options={"archive_days":.001,"concurrency":2,"batch_size":2,"max_attempts":1})
        e.replay()
        def batch_model(payload):
            if payload["stage"]=="adjudicate_batch":
                return {"decisions":[{"item_id":item["item_id"],**model({**item,"stage":"adjudicate"})}
                                     for item in payload["items"]]}
            if payload["stage"]=="wiki_batch":
                return {"cards":[{"event_id":item["event_id"],**model({"stage":"wiki","events":[item["event"]]})}
                                 for item in payload["items"]]}
            return model(payload)
        def fail_one_batch(payload):
            if payload["stage"]=="adjudicate_batch" and any(i["item_id"]=="a" for i in payload["items"]):
                raise RuntimeError("temporary model failure")
            return batch_model(payload)
        e.slow_resolver=fail_one_batch
        self.assertEqual(e.run_slow(is_idle=True)["status"],"needs_review")
        before=copy.deepcopy(records(e.memory,"on_consolidation_tasks"))
        failed=[tid for tid,t in before.items() if t["status"]=="failed"]
        self.assertEqual(len(failed),1)
        self.assertTrue(any(r["lifecycle"]=="archived" for r in e.runtime.values()))
        e.slow_resolver=lambda _:self.fail("Resume cannot silently reset exhausted batch attempts")
        self.assertEqual(e.run_slow(is_idle=True)["status"],"needs_review")
        self.assertEqual(before,records(e.memory,"on_consolidation_tasks"))
        calls=[]
        def recovered(payload):
            calls.append(copy.deepcopy(payload))
            return batch_model(payload)
        e.slow_resolver=recovered
        self.assertEqual(e.run_slow(is_idle=True,retry_unresolved=True)["status"],"complete")
        task=records(e.memory,"on_consolidation_tasks")[failed[0]]
        self.assertEqual(task["status"],"complete")
        self.assertEqual(task["attempts"],2)
        self.assertEqual(len(task["retry_authorizations"]),1)
        self.assertEqual(task["input"],before[failed[0]]["input"])
        self.assertEqual([p for p in calls if p["stage"]=="adjudicate_batch"],[task["input"]])
        self.checked(e)

    def test_shared_key_recall_is_not_limited_by_temporal_window(self):
        e = self.engine([("与小王聊天甲",[1,0,0],0),("与小王聊天乙",[1,0,0],1),
                         ("与小王聊天丙",[0,1,0],172800)])
        e.replay()
        target = e.states["与小王聊天甲"]["event_id"]
        self.assertNotIn(target,e.active)
        self.assertIn(target,Consolidation(e).candidates("与小王聊天丙"))
        self.checked(e)

    def test_disjoint_keys_do_not_request_stitch(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),
                         ("c",[0,1,0],1200),("d",[0,1,0],1201)])
        e.replay()
        e.run_slow(is_idle=True)
        self.assertNotIn("stitch",{t["input"]["stage"] for t in records(e.memory,"on_consolidation_tasks").values()})
        self.checked(e)

    def test_resume_idempotence_and_failed_requests_bounded(self):
        e = self.engine([("a",[1,0,0],0)],options={"max_attempts":2})
        e.replay()
        def offline(_):raise RuntimeError("offline")
        e.slow_resolver = offline
        result = e.run_slow(is_idle=True)
        self.assertEqual(result["status"],"needs_review")
        self.assertEqual(result["model_attempts"],4) # one adjudication + one Wiki
        e.run_slow(is_idle=True)
        self.assertEqual(e.report()["consolidation"]["model_attempts"],4)
        e.slow_resolver = model
        result = e.run_slow(is_idle=True,retry_unresolved=True)
        self.assertEqual(result["status"],"complete")
        self.assertTrue(any(t.get("retry_authorizations") for t in records(e.memory,"on_consolidation_tasks").values()))
        self.checked(e)

    def test_successful_resume_does_not_recall_llm(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        e.run_slow(is_idle=True)
        resumed = HyperMem(e.memory,e.settings,encoder=e.encoder,slow_resolver=lambda _:self.fail("repeat call"))
        resumed.run_slow(is_idle=True)
        self.checked(resumed)

    def test_ready_semantic_version_obsoletes_orphaned_equivalent_task(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay(); e.run_slow(is_idle=True)
        wiki = next(iter(records(e.memory,"on_event_wiki").values()))
        complete = records(e.memory,"on_consolidation_tasks")[wiki["task_id"]]
        orphan = {**copy.deepcopy(complete),"status":"interrupted","attempts":3,"response":None}
        orphan["input"]["events"][0]["lifecycle"] = "archived"
        orphan_id = "C-"+digest(orphan["input"])[:24]
        orphan["task_id"] = orphan_id
        with e.memory.transaction():save(e.memory,"on_consolidation_tasks",orphan_id,orphan)
        e.slow_resolver = lambda _:self.fail("Equivalent ready Wiki must not call LLM")
        self.assertEqual(e.run_slow(is_idle=True)["status"],"complete")
        retired = records(e.memory,"on_consolidation_tasks")[orphan_id]
        self.assertEqual(retired["status"],"obsolete")
        self.assertEqual(retired["replaced_by"],wiki["task_id"])
        self.checked(e)

    def test_resume_skips_reviewed_before_rebuilding_candidates(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),("q",[0,1,0],1200)])
        e.replay()
        e.run_slow(is_idle=True)
        resumed = HyperMem(e.memory,e.settings,encoder=e.encoder,slow_resolver=lambda _:self.fail("repeat call"))
        with patch.object(Consolidation,"candidates",side_effect=AssertionError("reviewed item rebuilt candidates")):
            resumed.run_slow(is_idle=True)
        self.checked(resumed)

    def test_invalid_model_cannot_modify_canonical_or_wiki(self):
        e = self.engine([("a",[1,0,0],0)],options={"max_attempts":1})
        e.replay()
        before = self.baseline(e)
        e.slow_resolver = lambda p:{**model(p),"status":"complete"}
        e.run_slow(is_idle=True)
        self.assertEqual(before,self.baseline(e))
        self.assertFalse(records(e.memory,"on_soft_relations"))
        self.assertNotIn("title",next(iter(records(e.memory,"on_event_wiki").values())))
        self.checked(e)

    def test_validation_rejects_unknown_evidence_keys_and_times(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        payload = {"stage":"wiki","events":[Consolidation(e).card(e.states["a"]["event_id"])]}
        for extra in ({"stable_keys":["invented"]},{"start_time":"1900-01-01T00:00:00+00:00"},
                      {"end_time":"2026-09-01"},{"evidence_ids":["unknown"]}):
            with self.assertRaises(ValueError):validate({**model(payload),**extra},payload)

    def test_stage_envelope_is_exact_and_does_not_allow_extra_control_fields(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        payload = {"stage":"wiki","events":[Consolidation(e).card(e.states["a"]["event_id"])]}
        result = model(payload)
        self.assertEqual(validate({"wiki":result},payload),result)
        for value in ({"wiki":result,"status":"complete"},{"adjudicate":result},
                      {"wiki":{**result,"status":"complete"}}):
            with self.assertRaises(ValueError):validate(value,payload)

    def test_known_event_id_is_removed_from_evidence_but_unknown_id_rejected(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        card = Consolidation(e).card(e.states["a"]["event_id"])
        payload = {"stage":"wiki","events":[card]}
        result = model(payload)
        normalized = validate({**result,"evidence_ids":[card["event_id"],*result["evidence_ids"]]},payload)
        self.assertEqual(normalized["evidence_ids"],result["evidence_ids"])
        with self.assertRaises(ValueError):
            validate({**result,"evidence_ids":["E-unknown",*result["evidence_ids"]]},payload)

    def test_wiki_repairs_only_unambiguous_split_scope_key(self):
        key = '["tracking_id","unknown_carrier","JDV021032201226"]'
        card = {"event_id":"E-1","start_time":START.isoformat(),
                "end_time":(START+timedelta(seconds=1)).isoformat(),
                "allowed_keys":[key],"member_summaries":[{"obs_id":"o-1","keys":[key]}]}
        payload = {"stage":"wiki","events":[card]}
        result = {"title":"发货通知","summary":"观测包含运单号。",
                  "stable_keys":["tracking_id","JDV021032201226"],
                  "start_time":card["start_time"],"end_time":card["end_time"],
                  "evidence_ids":["o-1"]}
        self.assertEqual(validate(result,payload)["stable_keys"],[key])

        other = '["tracking_id","explicit_carrier","JDV021032201226"]'
        ambiguous = copy.deepcopy(payload)
        ambiguous["events"][0]["allowed_keys"].append(other)
        ambiguous["events"][0]["member_summaries"][0]["keys"].append(other)
        with self.assertRaises(ValueError):
            validate(result,ambiguous)
        for malformed in (["tracking_id"],["JDV021032201226"],["tracking_id","unknown"]):
            with self.assertRaises(ValueError):
                validate({**result,"stable_keys":malformed},payload)

    def test_wiki_drops_allowed_key_without_cited_support_but_rejects_unknown(self):
        cited,uncited = "entity:candy","entity:郑欣悦在微信"
        card = {"event_id":"E-1","start_time":START.isoformat(),
                "end_time":(START+timedelta(seconds=1)).isoformat(),
                "allowed_keys":[cited,uncited],
                "member_summaries":[{"obs_id":"o-1","keys":[cited]},
                                    {"obs_id":"o-2","keys":[uncited]}]}
        payload = {"stage":"wiki","events":[card]}
        result = {"title":"沟通","summary":"与客服沟通。","stable_keys":[cited,uncited],
                  "start_time":card["start_time"],"end_time":card["end_time"],
                  "evidence_ids":["o-1"]}
        self.assertEqual(validate(result,payload)["stable_keys"],[cited])
        with self.assertRaises(ValueError):
            validate({**result,"stable_keys":[cited,"entity:unknown"]},payload)

    def test_saved_valid_response_recovers_exhausted_format_failure(self):
        e = self.engine([("a",[1,0,0],0)],options={"max_attempts":1})
        e.replay()
        e.slow_resolver = lambda p:{p["stage"]:model(p)}
        # Reproduce a response saved by a previous strict-envelope parser.
        with patch("hypermem.online.consolidation.validate",side_effect=ValueError("old parser")):
            e.run_slow(is_idle=True)
        def resolver(p):
            if p["stage"]=="adjudicate":self.fail("Saved adjudication should be reused")
            return model(p) # its new soft relation changes the Wiki context
        e.slow_resolver = resolver
        result = e.run_slow(is_idle=True)
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["model_attempts"],3)
        tasks = list(records(e.memory,"on_consolidation_tasks").values())
        self.assertTrue(next(t for t in tasks if t["input"]["stage"]=="adjudicate").get("recovery"))
        self.assertEqual(sum(t["status"]=="obsolete" for t in tasks),1)
        self.checked(e)

    def test_append_invalidates_wiki_and_soft_versions(self):
        e = self.engine([("a",[1,0,0],0),("b",[1,0,0],1),("q",[0,1,0],1200)])
        e.replay()
        target = e.states["a"]["event_id"]
        def resolver(p):
            if p["stage"]=="adjudicate":
                return {"action":"assign","target_event":target,"confidence":.9,"reason":"同一活动。","evidence_ids":["q","a"]}
            return model(p)
        e.slow_resolver = resolver
        e.run_slow(is_idle=True)
        o = observation("q2","q2",1201)
        e.encoder.vectors["q2"] = [0,1,0]
        e.memory.add_observation(o,{"input":o.to_dict()})
        # q was isolated and not active: explicit same strong key isn't present;
        # a new pair later forms a separate event, leaving original q intact.
        with e._transaction():
            e.prepare([o])
            e._attach("q2",target,o.start_time,trace={"route":"test_fixture"})
        self.assertEqual(records(e.memory,"on_event_wiki")[target]["semantic_status"],"stale")
        self.assertTrue(all(r["status"]=="stale" for r in records(e.memory,"on_soft_relations").values()))
        e.finalize()
        self.checked(e)



    def test_as_of_must_be_aware_and_not_before_watermark(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        for clock in (START-timedelta(seconds=1),"2026-09-01"):
            with self.assertRaises(ValueError):e.run_slow(is_idle=True,as_of=clock)

    def test_idle_topic_split_partitions_interleaved_members_and_updates_wiki(self):
        vectors = [[1,0,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,1,0]]
        e = self.engine([(str(i),v,i,"one-task") for i,v in enumerate(vectors)],
                        split_enabled=True,split_min_members=4,tau_c=.9)
        for o in e.memory.iter_observations():e.process(o)
        parent = e.states["0"]["event_id"]
        self.assertEqual(len(e.runtime),1)
        e.run_slow(is_idle=False)
        self.assertEqual(len(e.runtime),1)
        e.run_slow(is_idle=True)
        self.assertEqual(e.runtime[parent]["lifecycle"],"superseded")
        groups = {frozenset(ev.observation_ids) for ev in e.memory.iter_events() if e.runtime[ev.event_id]["lifecycle"]!="superseded"}
        self.assertEqual(groups,{frozenset({"0","1","3"}),frozenset({"2","4","5"})})
        self.assertEqual(set(records(e.memory,"on_event_wiki")),set(e.runtime))
        e.finalize()
        self.checked(e)

    def test_response_survives_atomic_apply_failure_without_network_replay(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        with patch.object(Consolidation,"soft",side_effect=RuntimeError("storage failure")):
            with self.assertRaises(RuntimeError):e.run_slow(is_idle=True)
        task = next(iter(records(e.memory,"on_consolidation_tasks").values()))
        self.assertEqual(task["status"],"responded")
        self.assertFalse(records(e.memory,"on_soft_relations"))
        def resolver(p):
            if p["stage"]=="adjudicate":self.fail("Response should be reused")
            return model(p)
        e.slow_resolver = resolver
        e.run_slow(is_idle=True)
        self.checked(e)

    def test_model_reads_text_not_vectors_and_http_json_transport(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        e.slow_resolver = None
        e.settings["llm"].update(model="fixture",base_url="http://fixture.invalid")
        def http(cfg,body):
            self.assertEqual(body["model"],"fixture")
            p = json.loads(body["messages"][1]["content"])
            encoded = json.dumps(p)
            self.assertNotIn('"prototype"',encoded)
            self.assertNotIn('"embedding"',encoded)
            self.assertNotIn('"vector"',encoded)
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(model(p))}}]}
        with patch("hypermem.online.consolidation.request_json",side_effect=http) as calls:
            e.run_slow(is_idle=True)
            self.assertEqual(calls.call_count,2)

    def test_audit_detects_tampered_soft_source_or_target(self):
        e = self.engine([("a",[1,0,0],0)])
        e.replay()
        e.run_slow(is_idle=True)
        from hypermem.online.consolidation import save
        rid,r = next(iter(records(e.memory,"on_soft_relations").items()))
        with e.memory.transaction():save(e.memory,"on_soft_relations",rid,{**r,"source":"fabricated"})
        result = audit(e.memory)
        self.assertEqual(result["status"],"failed")
        self.assertIn("soft_adjudication_projection",{p["code"] for p in result["issues"]})

    def test_fast_writes_during_slow_http_and_stale_reply_is_not_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            m = EventMemory(Path(folder)/"memory.sqlite")
            self.addCleanup(m.close)
            vectors = {"a":[1,0,0],"b":[1,0,0],"q":[0,1,0],"new":[1,0,0]}
            cfg = settings(consolidation={"enabled":True})
            fast = HyperMem(m,cfg,encoder=Encoder(vectors))
            for oid,second in (("a",0),("b",1),("q",1200)):
                o = observation(oid,oid,second,order="task-a" if oid!="q" else None)
                m.add_observation(o,{"input":o.to_dict()})
            fast.replay()
            target = fast.states["a"]["event_id"]
            started,released = threading.Event(),threading.Event()
            errors,results = [],[]
            def resolver(p):
                if p["stage"]=="adjudicate":
                    started.set()
                    if not released.wait(5):raise RuntimeError("Fast did not release test response")
                    return {"action":"assign","target_event":target,"confidence":.9,"reason":"同一活动。","evidence_ids":["q","a"]}
                return model(p)
            def worker():
                try:
                    with EventMemory(m.path) as other:
                        slow = HyperMem(other,cfg,encoder=Encoder(vectors),slow_resolver=resolver)
                        results.append(slow.run_slow(is_idle=True))
                except BaseException as exc:errors.append(exc)
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                o = observation("new","new",1201,order="task-a")
                m.add_observation(o,{"input":o.to_dict()})
                fast.process(o) # must not be blocked by Slow's remote request
            finally:
                released.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors,errors)
            self.assertEqual(results[0]["status"],"needs_review")
            self.assertFalse(records(m,"on_soft_relations"))
            self.assertIn("obsolete",{t["status"] for t in records(m,"on_consolidation_tasks").values()})
            fast._refresh_if_changed()
            fast.finalize()
            self.checked(fast)


if __name__=="__main__":unittest.main()
