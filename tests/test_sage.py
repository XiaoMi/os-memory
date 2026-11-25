# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import tempfile
import unittest

from test_online_semantic import Encoder, observation, settings
from test_idle_consolidation import model as wiki_model
from hypermem.memory import EventMemory
from hypermem.online.engine import OnlineHyperMem as HyperMem
from hypermem.online.sage import config, rows, save
from hypermem.online.sage_tools import ToolSAGE
from hypermem.online.hyperos_retrieval import HyperOSRetriever


class AnyEncoder:
    def encode(self,texts): return [[.9,.1,.2] for _ in texts]


def sage_model(payload):
    if payload["stage"]=="summarize_event+extract_facts":
        return {"events":[{"event_id":e["event_id"],"type":"other","topic_anchors":["测试活动"],
            "facts":[{"text":"事件中出现了测试活动记录。","tags":["测试"],"time":None,
                      "support":[e["members"][0]["obs_id"]]}]} for e in payload["events"]]}
    return {"groups":[{"group_id":g["group_id"],"memories":[{
        "text":"多次事件均出现测试活动记录。","support_events":[e["event_id"] for e in g["events"]],
        "confidence":"high","evidence_ids":[e["facts"][0]["support"][0] for e in g["events"]],
        "revises":None}]} for g in payload["groups"]]}


class SageTests(unittest.TestCase):
    def setUp(self):
        self.m=EventMemory();self.addCleanup(self.m.close)
        vectors={"a1":[1,0,0],"a2":[1,0,0],"b1":[0,1,0],"b2":[0,1,0]}
        for i,oid in enumerate(vectors):
            o=observation(oid,oid,i);self.m.add_observation(o,{"input":o.to_dict()})
        self.engine=HyperMem(self.m,settings(consolidation={"enabled":True}),encoder=Encoder(vectors),slow_resolver=wiki_model)
        self.engine.replay();self.engine.run_slow(is_idle=True)
        from test_sage_tools import model as tool_model
        self.sage=ToolSAGE(self.m,config({"embedding":{"dimension":3},"event":{"batch_size":2},
            "long_memory":{"batch_size":2}}),tool_resolver=tool_model,resolver=sage_model,encoder=AnyEncoder())

    def test_complete_layered_build_and_retrieval(self):
        with tempfile.TemporaryDirectory() as output:
            report=self.sage.run(output)
            self.assertEqual(report["status"],"complete",report)
        self.assertEqual(self.sage.audit()["status"],"passed")
        graph=self.sage.projection()
        self.assertEqual(graph["summary"]["observations"],4)
        self.assertGreaterEqual(graph["summary"]["events"],2)
        self.assertGreater(graph["summary"]["facts"],0)
        self.assertGreater(graph["summary"]["long_memories"],0)
        result=HyperOSRetriever(self.sage).retrieve("测试活动")
        self.assertTrue(result["events"])

    def test_fact_support_outside_visible_members_is_rejected(self):
        eid=self.sage.current_ids()[0]
        payload={"max_topics":2,"max_facts":2,"events":[{"event_id":eid,"members":self.sage._members(eid)}]}
        bad=sage_model({"stage":"summarize_event+extract_facts",**payload})
        bad["events"][0]["facts"][0]["support"]=["invented"]
        with self.assertRaises(ValueError):self.sage.validate_events(bad,payload)

    def test_anchor_degree_downweight_is_bounded(self):
        self.sage.run()
        result=HyperOSRetriever(self.sage).retrieve("测试")
        self.assertLessEqual(len(result["node_scores"]),self.sage.cfg["retrieval"]["budget_b"])
        self.assertTrue(rows(self.m,"sage_anchors"))

    def test_retrieval_reuses_index_and_invalidates_after_same_connection_write(self):
        self.sage.run();retriever=HyperOSRetriever(self.sage)
        first=retriever.retrieve("测试活动")
        statements=[];self.m._connection.set_trace_callback(statements.append)
        try:second=retriever.retrieve("测试活动")
        finally:self.m._connection.set_trace_callback(None)
        self.assertEqual(first,second)
        self.assertFalse(any("FROM sage_embeddings" in s for s in statements))
        for nid,item in rows(self.m,"sage_embeddings").items():
            save(self.m,"sage_embeddings",nid,{**item,"embedding":[0.,0.,0.]})
        changed=retriever.retrieve("测试活动")
        self.assertTrue(any(v>0 for v in first["node_scores"].values()))
        self.assertTrue(all(v<0 for v in changed["node_scores"].values()))
        self.assertTrue(all(f["support"] for e in changed["events"] for f in e["facts"]))

    def test_retrieval_invalidates_after_external_connection_write(self):
        from pathlib import Path
        from types import SimpleNamespace
        self.sage.run()
        with tempfile.TemporaryDirectory() as directory:
            with EventMemory(Path(directory)/"memory.sqlite") as memory:
                self.m._connection.backup(memory._connection)
                view=SimpleNamespace(m=memory,cfg=self.sage.cfg,encoder=self.sage.encoder,
                                     center=self.sage.center,events=self.sage.events)
                retriever=HyperOSRetriever(view);first=retriever.retrieve("测试活动")
                with EventMemory(memory.path) as other:
                    for nid,item in rows(other,"sage_embeddings").items():
                        save(other,"sage_embeddings",nid,{**item,"embedding":[0.,0.,0.]})
                changed=retriever.retrieve("测试活动")
                self.assertNotEqual(first["node_scores"],changed["node_scores"])
                self.assertTrue(all(v<0 for v in changed["node_scores"].values()))


if __name__=="__main__":unittest.main()
