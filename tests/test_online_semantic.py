# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from hypermem.config import config
from hypermem.memory import EventMemory
from hypermem.online.engine import OnlineHyperMem as HyperMem
from hypermem.schema import AtomicObservation, Provenance
from hypermem.online.clustering import topic_split, unit
from hypermem.online.io import audit


START = datetime(2026,9,1,tzinfo=timezone.utc)


class Encoder:
    def __init__(self,vectors):
        self.vectors = vectors
        self.calls = 0
    def encode(self,texts):
        self.calls += 1
        return [self.vectors[t] for t in texts]


def settings(**values):
    consolidation = {k:v for k,v in values.pop("consolidation",{}).items() if k!="enabled"}
    if consolidation:
        values["consolidation"] = consolidation
    return config({"online":{"mean_mode":"fixed","fixed_mean":[0.,0.,0.],"split_enabled":False,**values}})


def observation(oid,text,second=0,app="test",order=None):
    data = {"content":text}
    if order:
        data.update(orderId=order,orderPlatform="test-issuer")
    return AtomicObservation(oid,START+timedelta(seconds=second),app,
        json.dumps({"from_app":app,"perception_type":"ScreenPerception","data":data}),{},Provenance("R-"+oid))


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.m = EventMemory()
        self.addCleanup(self.m.close)

    def add(self,o):
        self.m.add_observation(o,{"input":o.to_dict()})
        return o

    def engine(self,vectors,**cfg):
        return HyperMem(self.m,settings(**cfg),encoder=Encoder(vectors))


    def test_topic_split_separates_interleaved_members_and_preserves_history(self):
        e=self.engine({"a":[1,0,0],"b":[0,1,0]},split_enabled=True,split_min_members=6,tau_c=.65)
        # A shared order forces one event by the symbolic route; low cohesion
        # must subsequently reveal the two interleaved semantic threads.
        for i in range(8):
            text="a" if i%2==0 else "b"
            # The first two confirm semantically, then symbolic continuation.
            if i==1:text="a"
            e.process(self.add(observation(str(i),text,i,order="ORDER-ONE")))
        self.assertEqual(e.report()["events"],1)
        e.finalize()
        self.assertEqual(e.report()["events"],2)
        self.assertEqual(e.report()["superseded_events"],1)
        self.assertEqual(self.m.event_for_observation("0"),self.m.event_for_observation("6"))
        self.assertNotEqual(self.m.event_for_observation("0"),self.m.event_for_observation("7"))
        checked=audit(self.m,True)
        self.assertEqual(checked["issues"],[])
    def test_counter_progress_preserves_final_results_and_avoids_database_reads(self):
        texts = ["a1", "b1", "a2", "b2", "a3", "aside"]
        vectors = {t:([0,0,1] if t == "aside" else [1,0,0] if t[0] == "a" else [0,1,0]) for t in texts}
        for consolidation in (False, True):
            results = []
            for detailed in (True, False):
                with self.subTest(consolidation=consolidation, detailed=detailed), EventMemory() as m:
                    for i,t in enumerate(texts):
                        m.add_observation(observation(t,t,i*10), {"input":t})
                    cfg = settings(consolidation={"enabled":consolidation})
                    cfg["replay"]["progress_every"] = 1
                    e = HyperMem(m,cfg,encoder=Encoder(vectors))
                    statements, updates = [], []
                    m._connection.set_trace_callback(statements.append)
                    process = e.process
                    def tracked_process(o):
                        result = process(o)
                        statements.clear()
                        return result
                    def progress(count, report):
                        if not detailed:
                            self.assertEqual(statements, [])
                            self.assertEqual(report["progress_detail"], "counters_only_not_audit")
                        updates.append((count, report["processed"], report["events"], report["pending"]))
                    with patch.object(e,"process",side_effect=tracked_process), patch.object(e,"report",wraps=e.report) as report:
                        final = e.replay(progress=progress,detailed_progress=detailed)
                        self.assertEqual(report.call_count, len(texts)+1 if detailed else 1)
                    self.assertNotIn("progress_detail", final)
                    self.assertEqual(audit(m,True)["status"], "passed")
                    results.append((final,updates,e.states,e.runtime,
                                    [event.to_dict() for event in m.iter_events()],
                                    [m.last_decision(t) for t in texts]))
            self.assertEqual(results[0], results[1])

    def test_reload_reads_vectors_and_grounding_in_one_scan(self):
        for i,t in enumerate(("a","b")):
            self.add(observation(t,t,i))
        e = self.engine({"a":[1,0,0], "b":[1,.1,0]})
        e.replay()
        expected = (e.g, {k:v.tolist() for k,v in e.raw.items()}, {k:v.tolist() for k,v in e.z.items()})
        statements = []
        self.m._connection.set_trace_callback(statements.append)
        e._reload()
        self.assertEqual(sum("FROM on_vectors" in s for s in statements), 1)
        self.assertEqual((e.g, {k:v.tolist() for k,v in e.raw.items()}, {k:v.tolist() for k,v in e.z.items()}), expected)

    def test_counter_progress_resume_reuses_vectors_and_preserves_membership(self):
        texts = ["a1","b1","a2","b2","a3","aside"]
        vectors = {t:([0,0,1] if t == "aside" else [1,0,0] if t[0] == "a" else [0,1,0]) for t in texts}
        results = []
        for resume in (False, True):
            with EventMemory() as m:
                for i,t in enumerate(texts):
                    m.add_observation(observation(t,t,i*10), {"input":t})
                cfg = settings(consolidation={"enabled":True})
                cfg["replay"]["progress_every"] = 1
                e = HyperMem(m,cfg,encoder=Encoder(vectors))
                if resume:
                    def stop(count, report):
                        if count == 3:raise InterruptedError("checkpoint test")
                    with self.assertRaises(InterruptedError):
                        e.replay(progress=stop,detailed_progress=False)
                    encoder = Encoder({})
                    e = HyperMem(m,cfg,encoder=encoder)
                updates = []
                final = e.replay(progress=lambda n,r: updates.append((n,r["processed"])),detailed_progress=False)
                if resume:
                    self.assertEqual(updates[0], (1,3))
                    self.assertEqual(encoder.calls, 0)
                self.assertEqual(audit(m,True)["status"], "passed")
                results.append((final,e.states,e.runtime,[ev.to_dict() for ev in m.iter_events()],
                                [m.last_decision(t) for t in texts]))
        self.assertEqual(results[0],results[1])

    def test_second_topic_evidence_confirms_without_entities(self):
        a,b = self.add(observation("a","A",0)), self.add(observation("b","B",20))
        e = self.engine({"A":[1,0,0],"B":[1,.02,0]})
        self.assertEqual(e.process(a)["status"],"pending")
        e.process(b)
        self.assertEqual(self.m.event_for_observation("a"),self.m.event_for_observation("b"))
        e.finalize()
        self.assertEqual(audit(self.m,True)["status"],"passed")

    def test_interleaved_topics_keep_two_active_lines(self):
        texts = ["a1","b1","a2","b2","a3","b3"]
        e = self.engine({t:[1,0,0] if t[0]=="a" else [0,1,0] for t in texts})
        for i,t in enumerate(texts):
            e.process(self.add(observation(t,t,i*10)))
        self.assertEqual(len(e.active),2)
        self.assertEqual(self.m.event_for_observation("a1"),self.m.event_for_observation("a3"))
        self.assertNotEqual(self.m.event_for_observation("a1"),self.m.event_for_observation("b1"))

    def test_incidental_timeout_does_not_change_prototype_or_wake(self):
        e = self.engine({"a":[1,0,0],"b":[1,0,0],"aside":[0,1,0],"later":[0,0,1]},active_minutes=1)
        for oid,t in (("a",0),("b",5),("aside",10)):
            e.process(self.add(observation(oid,oid,t)))
        eid = self.m.event_for_observation("a")
        before = dict(e.runtime[eid])
        e.process(self.add(observation("later","later",300)))
        self.assertEqual(e.states["aside"]["role"],"incidental")
        self.assertEqual(e.runtime[eid]["prototype"],before["prototype"])
        self.assertEqual(e.runtime[eid]["last_activity"],before["last_activity"])
        self.assertEqual(e.runtime[eid]["lifecycle"],"dormant")
        e.finalize()
        self.assertEqual(audit(self.m,True)["status"],"passed")

    def test_expired_pending_cannot_confirm_with_late_same_topic(self):
        e = self.engine({"a":[1,0,0],"b":[1,0,0]})
        e.process(self.add(observation("a","a",0)))
        e.process(self.add(observation("b","b",181)))
        e.finalize()
        self.assertEqual(e.report()["isolated_events"],2)

    def test_exact_deadline_can_confirm(self):
        e = self.engine({"a":[1,0,0],"b":[1,0,0]})
        e.process(self.add(observation("a","a",0)))
        e.process(self.add(observation("b","b",180)))
        self.assertEqual(e.report()["events"],1)
        self.assertEqual(e.report()["pending"],0)

    def test_symbolic_route_wakes_dormant_despite_semantic_mismatch(self):
        e = self.engine({"a":[1,0,0],"b":[1,0,0],"c":[0,1,0]},active_minutes=1)
        for name,t in (("a",0),("b",1),("c",300)):
            e.process(self.add(observation(name,name,t,order="ORDER-1")))
        self.assertEqual(e.report()["events"],1)
        self.assertEqual(self.m.last_decision("c")["reason"],"event_identifier")
        self.assertTrue(self.m.last_decision("c")["was_dormant"])

    def test_symbolic_candidate_beats_other_higher_semantic_candidate(self):
        texts = ["a1","a2","b1","b2","incoming"]
        e = self.engine({t:[1,0,0] if t.startswith("a") else [0,1,0] for t in texts})
        for i,t in enumerate(texts):
            e.process(self.add(observation(t,t,i*10,order="ORDER-A" if t.startswith("a") or t=="incoming" else None)))
        self.assertEqual(self.m.event_for_observation("incoming"),self.m.event_for_observation("a1"))
        self.assertNotEqual(self.m.event_for_observation("incoming"),self.m.event_for_observation("b1"))

    def test_reusable_entity_key_still_requires_tau_s(self):
        vectors={"联系手机号 13800000000":[1,0,0],"另一活动手机号 13800000000":[0,1,0]}
        e=self.engine(vectors)
        for oid,text,second in (("a","联系手机号 13800000000",0),("b","联系手机号 13800000000",1),
                                ("c","另一活动手机号 13800000000",300)):
            e.process(self.add(observation(oid,text,second)))
        self.assertEqual(e.states["c"]["status"],"pending")
        e.finalize()
        self.assertEqual(e.states["c"]["role"],"incidental")

    def test_chat_and_stream_have_distinct_learning_rates(self):
        for app,alpha in (("微信",.12),("抖音",.5)):
            with self.subTest(app=app),EventMemory() as m:
                e=HyperMem(m,settings(),encoder=Encoder({"a":[1,0,0],"b":[1,0,0],"c":[1,1,0]}))
                for i,t in enumerate(("a","b","c")):
                    o=observation(t,t,i,app=app);m.add_observation(o,{"input":t});e.process(o)
                trace=m.last_decision("c")
                self.assertEqual(trace["alpha"],alpha)
                np.testing.assert_allclose(trace["prototype_after"],unit((1-alpha)*np.array([1,0,0])+alpha*unit([1,1,0])))

    def test_corpus_mean_and_degenerate_residuals(self):
        for t in ("a","b"):
            self.add(observation(t,t))
        cfg=config({"online":{"split_enabled":False}})
        e=HyperMem(self.m,cfg,encoder=Encoder({"a":[1,2,0],"b":[1,2,0],"c":[0,1,0]}))
        e.replay()
        self.assertTrue(all(np.linalg.norm(v)==0 for v in e.z.values()))
        self.assertEqual(e.report()["isolated_events"],2)
        self.assertEqual(audit(self.m,True)["status"],"passed")
        with self.assertRaisesRegex(ValueError,"Corpus mean"):
            e.process(self.add(observation("c","c",10)))

    def test_resume_equivalent_to_uninterrupted(self):
        texts=["a1","b1","a2","b2","a3","aside"]
        vectors={t:[1,0,0] if t[0]=="a" and t!="aside" else [0,1,0] if t[0]=="b" else [0,0,1] for t in texts}
        outputs=[]
        for restart in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/"memory.sqlite"
                m=EventMemory(p);e=HyperMem(m,settings(),encoder=Encoder(vectors))
                try:
                    for i,t in enumerate(texts):
                        o=observation(t,t,i*10);m.add_observation(o,{"input":t});e.process(o)
                        if restart and i==2:
                            m.close();m=EventMemory(p);e=HyperMem(m,settings(),encoder=Encoder(vectors))
                    e.finalize();outputs.append([sorted(tuple(r) for r in m._connection.execute(q)) for q in (
                        "SELECT event_id,observation_id FROM event_observations","SELECT * FROM on_events","SELECT * FROM on_relations")])
                    self.assertEqual(audit(m,True)["status"],"passed")
                finally:m.close()
        self.assertEqual(outputs[0],outputs[1])

    def test_transaction_failure_leaves_no_half_event(self):
        e=self.engine({"a":[1,0,0],"b":[1,0,0]})
        a,b=self.add(observation("a","a")),self.add(observation("b","b",1))
        e.process(a)
        with patch.object(self.m,"add_event",side_effect=RuntimeError("simulated disk failure")):
            with self.assertRaises(RuntimeError):e.process(b)
        self.assertEqual(e.states["a"]["status"],"pending")
        self.assertNotIn("b",e.states)
        self.assertFalse(e.runtime)
        e.process(b)
        self.assertEqual(e.report()["events"],1)


    def test_split_does_not_depend_on_time_order(self):
        cfg=settings(split_min_members=6,tau_c=.65)["online"]
        x=np.array([[1.,0.,0.],[0.,1.,0.]]*8)
        a=topic_split(x,cfg);b=topic_split(x[::-1],cfg)
        self.assertIsNotNone(a)
        self.assertEqual(sorted(map(len,a["groups"])),sorted(map(len,b["groups"])))
        self.assertAlmostEqual(a["cohesion_after"],b["cohesion_after"])

    def test_split_checks_all_members_including_timed_out_asides(self):
        e=self.engine({"a":[1,0,0],"b":[0,1,0]},split_enabled=True,split_min_members=6,tau_c=.65)
        for i,(text,t) in enumerate((("a",0),("a",1),("b",10),("a",200),("b",210),("a",400),("b",410),("a",600))):
            e.process(self.add(observation(str(i),text,t)))
        self.assertEqual(e.report()["events"],1)
        parent=self.m.event_for_observation("0")
        self.assertEqual(e.runtime[parent]["prototype"],[1.,0.,0.])
        self.assertEqual(sum(s["role"]=="incidental" for s in e.states.values()),3)
        e.finalize()
        self.assertEqual(e.report()["events"],2)
        self.assertEqual(self.m.event_for_observation("2"),self.m.event_for_observation("6"))
        self.assertNotEqual(self.m.event_for_observation("0"),self.m.event_for_observation("2"))
        self.assertEqual(e.states["2"]["pre_split_role"],"incidental")
        self.assertEqual(audit(self.m,True)["issues"],[])

    def test_audit_detects_runtime_and_canonical_prototype_joint_tamper(self):
        e=self.engine({"a":[1,0,0],"b":[1,0,0]})
        for i,t in enumerate(("a","b")):e.process(self.add(observation(t,t,i)))
        e.finalize()
        self.m._connection.execute("UPDATE events SET event_json=json_set(event_json,'$.embedding',json('[0,1,0]'))")
        self.m._connection.execute("UPDATE on_events SET runtime_json=json_set(runtime_json,'$.prototype',json('[0,1,0]'))")
        self.assertIn("runtime_prototype_not_latest_trace",[i["code"] for i in audit(self.m,True)["issues"]])

    def test_audit_detects_centered_vector_tampering(self):
        e=self.engine({"a":[1,0,0],"b":[1,0,0]})
        for i,t in enumerate(("a","b")):e.process(self.add(observation(t,t,i)))
        e.finalize()
        self.m._connection.execute("UPDATE observation_features SET features_json=json_set(features_json,'$.embedding',json('[0,1,0]')) WHERE observation_id='a'")
        self.assertEqual(audit(self.m,True)["status"],"failed")

    def test_invalid_mean_and_late_arrival_rejected(self):
        with self.assertRaises(ValueError):settings(fixed_mean=[])
        e=self.engine({"a":[1,0,0],"b":[1,0,0]})
        e.process(self.add(observation("a","a",20)))
        with self.assertRaisesRegex(ValueError,"Out-of-order"):
            e.process(self.add(observation("b","b",10)))





if __name__=="__main__":unittest.main()
