# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy
from dataclasses import replace
import json
import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

from test_online_semantic import Encoder, observation, settings
from test_idle_consolidation import model as wiki_model
from test_sage import AnyEncoder, sage_model
from hypermem.memory import EventMemory
from hypermem.online.engine import OnlineHyperMem as HyperMem
from hypermem.online.io import audit
from hypermem.online.consolidation import records, save
from hypermem.online.sage import config, rows
from hypermem.online.sage_tools import ToolSAGE, tool_config
from hypermem.online.hyperos_retrieval import HyperOSRetriever


def call(name,args,ident="call-1"):
    return {"role":"assistant","content":None,"tool_calls":[{"id":ident,"type":"function",
            "function":{"name":name,"arguments":json.dumps(args)}}]}


def summarization(context):
    return {"title":"工具生成的事件标题","summary":"成员观测记录了测试活动。","event_type":"other",
            "entity_keys":context["allowed_keys"],"topic_anchors":["测试活动"]}


def model(messages,tools,eid):
    c=json.loads(messages[1]["content"])
    previous=[json.loads(m["content"]) for m in messages if m["role"]=="tool"]
    if not any(r.get("summary_written") for r in previous):return call("SummarizeEvent",summarization(c))
    return call("ExtractFacts",{"facts":[{"text":"事件包含测试活动。","tags":["测试"],"time":None,
                                              "support":[c["members"][-1]["obs_id"]]}]})


class ParallelToolSageTests(unittest.TestCase):
    def runner(self,count=3,concurrency=2):
        memory=EventMemory();self.addCleanup(memory.close)
        vectors={}
        for index in range(count):
            vector=[int(i==index) for i in range(3)]
            for offset in range(2):
                oid=f"event-{index}-obs-{offset}"
                o=observation(oid,oid,index*1200+offset)
                memory.add_observation(o,{"input":o.to_dict()});vectors[oid]=vector
        engine=HyperMem(memory,settings(consolidation={"enabled":True}),encoder=Encoder(vectors),slow_resolver=wiki_model)
        engine.replay();engine.run_slow(is_idle=True)
        runner=ToolSAGE(memory,config({"embedding":{"dimension":3},"event":{"concurrency":concurrency}}),
                        tool_resolver=model,resolver=sage_model,encoder=AnyEncoder())
        self.assertEqual(len(runner.current_ids()),count)
        return runner

    def test_event_card_changed_by_another_event_is_refined_again(self):
        runner=self.runner(count=2,concurrency=1)
        first,second=runner.current_ids()
        touched=False
        def side_effect(messages,tools,eid):
            nonlocal touched
            if eid==second and not touched:
                touched=True
                # What dirty_semantics does to a neighbour when a split invalidates a soft relation.
                with runner.m.transaction():
                    card=records(runner.m,"on_event_wiki")[first]
                    save(runner.m,"on_event_wiki",first,{**card,"semantic_status":"stale"})
            return model(messages,tools,eid)
        runner.tool_resolver=side_effect
        result=runner.run()
        self.assertTrue(touched)
        self.assertEqual(result["status"],"complete",result["audit"]["issues"])
        self.assertEqual(sorted(j["event_id"] for j in runner._jobs().values()),sorted([first,first,second]))

    def test_bounded_model_overlap_with_owner_thread_commits(self):
        runner=self.runner()
        owner=threading.get_ident();pair=set(runner.current_ids()[:2])
        barrier=threading.Barrier(2,timeout=3);lock=threading.Lock()
        active=peak=0;worker_threads=set()
        def parallel(messages,tools,eid):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active);worker_threads.add(threading.get_ident())
            try:
                if eid in pair and not any(m["role"]=="tool" for m in messages):barrier.wait()
                return model(messages,tools,eid)
            finally:
                with lock:active-=1
        runner.tool_resolver=parallel
        original_execute=runner._execute;original_save=runner._save_job
        def execute(*args):
            self.assertEqual(threading.get_ident(),owner)
            return original_execute(*args)
        def save_job(*args):
            self.assertEqual(threading.get_ident(),owner)
            return original_save(*args)
        with patch.object(runner,"_execute",side_effect=execute),patch.object(runner,"_save_job",side_effect=save_job):
            result=runner.run()
        self.assertEqual(result["status"],"complete",result)
        self.assertEqual(peak,2);self.assertNotIn(owner,worker_threads)
        self.assertEqual(result["execution"]["concurrency"],2)
        for job in runner._jobs().values():
            self.assertEqual(job["rounds"],2)
            self.assertEqual([r["name"] for r in job["results"]],["SummarizeEvent","ExtractFacts"])
        self.assertEqual(runner.m.get_metadata("sage_tool_execution")["context_visibility"],"batch_start_snapshot")
        self.assertEqual(audit(runner.m,True)["status"],"passed")

    def test_anchor_cache_discards_uncommitted_summary_after_rollback(self):
        runner=self.runner(count=2,concurrency=1)
        eid=runner.current_ids()[0]
        original=runner._record_result
        injected=False
        checked_rollback=False
        def record(job,message,tool,result):
            nonlocal injected,checked_rollback
            if result.get("summary_written") and not injected:
                injected=True
                self.assertTrue(runner._anchors())
                raise RuntimeError("Fail after anchor writes, before job commit")
            if result.get("ok") is False:
                checked_rollback=True
                self.assertEqual(runner._anchors(),{})
                self.assertEqual(rows(runner.m,"sage_anchors"),{})
            return original(job,message,tool,result)
        with patch.object(runner,"_record_result",side_effect=record):
            job=runner.process_event(eid)
        self.assertTrue(injected and checked_rollback)
        self.assertEqual(job["status"],"complete")
        self.assertEqual(runner._anchors(),rows(runner.m,"sage_anchors"))
        self.assertTrue(all(a["event_ids"]==[eid] for a in runner._anchors().values()))

    def test_parallel_projection_matches_sequential_for_same_tool_proposals(self):
        runners=[self.runner(concurrency=n) for n in (1,2)]
        for runner in runners:runner.build_events()
        for table in ("sage_events","sage_anchors","sage_facts"):
            self.assertEqual(rows(runners[0].m,table),rows(runners[1].m,table),table)
        for runner in runners:
            self.assertEqual(runner.audit()["unfinished_tool_events"],0)
            self.assertEqual(audit(runner.m,True)["status"],"passed")
        second=runners[0].current_ids()[1]
        context=lambda runner:next(j["context"] for j in runner._jobs().values() if j["event_id"]==second)
        # Snapshot timing is deliberately explicit, not a claim of identical LLM output.
        self.assertTrue(context(runners[0])["nearby_topic_anchors"])
        self.assertFalse(context(runners[1])["nearby_topic_anchors"])

    def test_cached_build_reloads_external_event_changes_on_reuse(self):
        runner=self.runner(count=2)
        with patch.object(runner.m,"iter_events",side_effect=AssertionError("Redundant initial reload")):
            runner.build_events()
        eid=runner.current_ids()[0]
        old_version=runner.version(eid)
        runner.m.update_event(replace(runner.m.get_event(eid),state="Externally changed event"))
        runner.build_events()
        self.assertNotEqual(runner.version(eid),old_version)
        self.assertEqual(rows(runner.m,"sage_events")[eid]["version"],runner.version(eid))
        self.assertEqual(runner.audit()["unfinished_tool_events"],0)

    def test_saved_parallel_replies_resume_without_repeating_requests(self):
        runner=self.runner(count=2)
        original=runner._receive;received=0
        def interrupted(*args):
            nonlocal received
            original(*args);received+=1
            if received==2:raise RuntimeError("stop after saved replies")
        with patch.object(runner,"_receive",side_effect=interrupted),self.assertRaisesRegex(RuntimeError,"saved replies"):
            runner.build_events()
        self.assertTrue(all(j["status"]=="responded" and j["rounds"]==1 for j in runner._jobs().values()))
        calls=[]
        def resume(messages,tools,eid):
            self.assertTrue(any(m["role"]=="tool" and "summary_written" in m["content"] for m in messages))
            calls.append(eid)
            return model(messages,tools,eid)
        resumed=ToolSAGE(runner.m,runner.cfg,tool_resolver=resume,resolver=sage_model,encoder=AnyEncoder())
        resumed.build_events()
        self.assertEqual(len(calls),2)
        self.assertTrue(all(j["status"]=="complete" and j["rounds"]==2 for j in resumed._jobs().values()))
        resumed.tool_resolver=lambda *args:self.fail("Completed batch must not call model again")
        resumed.build_events()

    def test_parallel_failure_preserves_budget_and_does_not_block_sibling(self):
        runner=self.runner(count=2);bad=runner.current_ids()[0]
        progress=[];runner.progress=progress.append
        runner.tool_resolver=lambda messages,tools,eid:call("unknown_tool",{}) if eid==bad else model(messages,tools,eid)
        runner.build_events()
        jobs={j["event_id"]:j for j in runner._jobs().values()}
        self.assertEqual(jobs[bad]["status"],"budget_exhausted")
        self.assertEqual(jobs[bad]["rounds"],12)
        self.assertEqual(sum(j["status"]=="complete" for j in jobs.values()),1)
        self.assertEqual(progress[-1]["processed_in_pass"],2)
        self.assertEqual(progress[-1]["complete_jobs_in_pass"],1)
        self.assertEqual(progress[-1]["exhausted_jobs_in_pass"],1)
        runner.tool_resolver=lambda *args:self.fail("Exhausted budget must not silently reset")
        runner.build_events()
        self.assertEqual(jobs,{j["event_id"]:j for j in runner._jobs().values()})

    def test_parallel_stale_reply_cannot_write_event(self):
        runner=self.runner(count=2);stale=runner.current_ids()[0]
        original=runner._receive;changed=False
        def receive(job,*args):
            nonlocal changed
            original(job,*args)
            if job["event_id"]==stale and not changed:
                runner.m.update_event(replace(runner.m.get_event(stale),state="changed before apply"))
                changed=True
        with patch.object(runner,"_receive",side_effect=receive):jobs=runner._process_batch(runner.current_ids())
        self.assertEqual(next(j for j in jobs if j["event_id"]==stale)["status"],"stale")
        self.assertNotIn(stale,rows(runner.m,"sage_events"))
        self.assertFalse(any(f["event_id"]==stale for f in rows(runner.m,"sage_facts").values()))

    def test_parallel_network_errors_consume_rounds_without_resetting(self):
        runner=self.runner(count=2);bad=runner.current_ids()[0]
        def offline(messages,tools,eid):
            if eid==bad:raise RuntimeError("service temporarily unavailable")
            return model(messages,tools,eid)
        runner.tool_resolver=offline
        runner.build_events()
        jobs={j["event_id"]:j for j in runner._jobs().values()}
        self.assertEqual(jobs[bad]["status"],"budget_exhausted")
        self.assertEqual(jobs[bad]["rounds"],12)
        self.assertEqual(jobs[bad]["calls"],0)
        self.assertEqual(sum(j["status"]=="complete" for j in jobs.values()),1)
        runner.tool_resolver=lambda *args:self.fail("Network failure cannot reset the round budget")
        runner.build_events()
        self.assertEqual(jobs,{j["event_id"]:j for j in runner._jobs().values()})


class ToolSageTests(unittest.TestCase):
    def setUp(self):
        self.m=EventMemory();self.addCleanup(self.m.close)
        vectors={"a1":[1,0,0],"a2":[1,0,0],"b1":[0,1,0],"b2":[0,1,0]}
        for i,oid in enumerate(vectors):
            o=observation(oid,oid,i,order="ORDER-1");self.m.add_observation(o,{"input":o.to_dict()})
        engine=HyperMem(self.m,settings(consolidation={"enabled":True}),encoder=Encoder(vectors),slow_resolver=wiki_model)
        engine.replay();engine.run_slow(is_idle=True)
        self.cfg=config({"embedding":{"dimension":3}})
        self.sage=ToolSAGE(self.m,self.cfg,tool_resolver=model,resolver=sage_model,encoder=AnyEncoder())
        self.root=self.sage.current_ids()[0]
        self.assertEqual(len(self.sage.current_ids()),1)


    def test_summary_facts_complete_export_retrieve_and_idempotence(self):
        before=[o.to_dict() for o in self.m.iter_observations()]
        with tempfile.TemporaryDirectory() as tmp:
            result=self.sage.run(tmp)
            self.assertEqual(result["status"],"complete",result)
            self.assertTrue((Path(tmp)/"layered_memory_graph.html").exists())
        self.assertEqual(rows(self.m,"sage_events")[self.root]["title"],"工具生成的事件标题")
        self.assertEqual(next(iter(rows(self.m,"sage_facts").values()))["support"],["b2"])
        self.assertEqual([o.to_dict() for o in self.m.iter_observations()],before)
        self.assertEqual(audit(self.m,True)["status"],"passed")
        self.assertTrue(HyperOSRetriever(self.sage).retrieve("测试")["events"])
        self.sage.tool_resolver=lambda *a:self.fail("completed tool job called model again")
        self.assertEqual(self.sage.run()["status"],"complete")

    def test_tool_error_feedback_then_correction(self):
        counter=0
        def correcting(messages,tools,eid):
            nonlocal counter
            counter+=1
            if counter==1:return call("ExtractFacts",{"facts":[]})
            if counter==2:
                self.assertIn("SummarizeEvent must succeed",messages[-1]["content"])
                c=json.loads(messages[1]["content"]);a=summarization(c);a["entity_keys"]=["invented"]
                return call("SummarizeEvent",a)
            if counter==3:self.assertIn("supplied event keys",messages[-1]["content"])
            return model(messages,tools,eid)
        self.sage.tool_resolver=correcting
        self.assertEqual(self.sage.run()["status"],"complete")
        self.assertEqual(next(iter(self.sage._jobs().values()))["calls"],4)

    def splitting_model(self,messages,tools,eid):
        if eid==self.root:
            previous=[json.loads(m["content"]) for m in messages if m["role"]=="tool"]
            if not previous:return call("SplitEvent",{"phase":"inspect","groups":[],"retain_parent":False})
            clusters=next(r["clusters"] for r in previous if "clusters" in r)
            return call("SplitEvent",{"phase":"commit","groups":[[c["cluster_id"]] for c in clusters],"retain_parent":True})
        return model(messages,tools,eid)

    def test_two_phase_split_conserves_members_and_refreshes_layers(self):
        self.sage.tool_resolver=self.splitting_model
        result=self.sage.run()
        self.assertEqual(result["status"],"complete",result)
        self.assertEqual(len(self.sage.current_ids()),2)
        groups=[set(self.m.get_event(e).observation_ids) for e in self.sage.current_ids()]
        self.assertCountEqual(groups,[{"a1","a2"},{"b1","b2"}])
        self.assertEqual(self.m._connection.execute("SELECT count(*) FROM event_observations").fetchone()[0],4)
        self.assertEqual(self.sage.runtime[self.root]["lifecycle"],"superseded")
        self.assertEqual(audit(self.m,True)["status"],"passed",audit(self.m,True))
        self.assertTrue(rows(self.m,"sage_long_memory"))
        self.assertNotIn(self.root,rows(self.m,"sage_events"))
        graph=self.sage.projection()
        self.assertEqual(graph["summary"]["parent_events"],1)
        self.assertEqual(graph["summary"]["leaf_events"],2)
        self.assertEqual(sum(e["type"]=="member_of" for e in graph["edges"]),4)
        self.assertEqual(sum(e["type"]=="part-of" and e["target"]==self.root for e in graph["edges"]),2)

    def test_split_children_can_be_processed_in_parallel(self):
        self.sage.cfg["event"]["concurrency"]=2
        self.test_two_phase_split_conserves_members_and_refreshes_layers()
        self.assertEqual(self.m.get_metadata("sage_tool_execution")["concurrency"],2)
        self.assertEqual(len(self.sage._jobs()),3)

    def test_partition_rejects_duplicates_missing_and_foreign_clusters(self):
        j=self.sage._job(self.root)
        with self.m.transaction():self.sage._execute(j,"SplitEvent",{"phase":"inspect","groups":[],"retain_parent":False})
        ids=[c["cluster_id"] for c in j["clusters"]]
        for groups in ([[ids[0]],[ids[0]]],[[ids[0]],["foreign"]],[ids]):
            with self.assertRaises(ValueError):
                with self.m.transaction():self.sage._execute(j,"SplitEvent",{"phase":"commit","groups":groups,"retain_parent":False})
        self.assertEqual(len(self.sage.current_ids()),1)

    def test_split_rollback_has_no_partial_children(self):
        j=self.sage._job(self.root)
        self.sage._execute(j,"SplitEvent",{"phase":"inspect","groups":[],"retain_parent":False})
        original=self.m.add_event
        calls=0
        def fail_second(e):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError("simulated disk failure")
            return original(e)
        with patch.object(self.m,"add_event",side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError,"disk failure"):
                with self.m.transaction():self.sage._execute(j,"SplitEvent",{"phase":"commit","groups":[[c["cluster_id"]] for c in j["clusters"]],"retain_parent":True})
        self.sage._refresh()
        self.assertEqual(self.sage.current_ids(),[self.root])
        self.assertEqual(audit(self.m,True)["status"],"passed")

    def test_split_cache_refresh_rolls_back_without_full_table_reload(self):
        j=self.sage._job(self.root)
        self.sage._execute(j,"SplitEvent",{"phase":"inspect","groups":[],"retain_parent":False})
        message=call("SplitEvent",{"phase":"commit","groups":[[c["cluster_id"]] for c in j["clusters"]],"retain_parent":True})
        j.update(rounds=2,status="responded",pending_message=message)
        with self.m.transaction():self.sage._save_job(j)
        original=self.sage._record_result;injected=False;checked=False
        def fail_after_refresh(job,assistant,tool,result):
            nonlocal injected,checked
            if result.get("child_events") and not injected:
                injected=True
                self.assertEqual(len(self.sage.current_ids()),2)
                raise RuntimeError("Fail after split caches changed, before job commit")
            if result.get("ok") is False:
                checked=True
                self.assertEqual(self.sage.current_ids(),[self.root])
                self.assertEqual(set(self.sage.events),{self.root})
                self.assertEqual(set(self.sage.wiki),{self.root})
            return original(job,assistant,tool,result)
        with patch.object(self.sage,"_record_result",side_effect=fail_after_refresh),patch.object(
                self.m,"iter_events",side_effect=AssertionError("Unexpected full-table refresh")):
            j,request=self.sage._advance(j)
        self.assertTrue(injected and checked and request)
        self.assertEqual(audit(self.m,True)["status"],"passed")
        j.update(status="responded",pending_message=message)
        j,request=self.sage._advance(j)
        self.assertEqual(j["status"],"complete")
        self.assertFalse(request)
        self.assertEqual(len(self.sage.current_ids()),2)
        self.assertEqual(audit(self.m,True)["status"],"passed")

    def test_response_journal_resumes_without_duplicate_model_call(self):
        j=self.sage._job(self.root)
        message=call("SummarizeEvent",summarization(j["context"]))
        j.update(rounds=1,status="responded",pending_message=message)
        with self.m.transaction():self.sage._save_job(j)
        def resumed(messages,tools,eid):
            self.assertTrue(any(m["role"]=="tool" and "summary_written" in m["content"] for m in messages))
            return model(messages,tools,eid)
        runner=ToolSAGE(self.m,self.cfg,tool_resolver=resumed,resolver=sage_model,encoder=AnyEncoder())
        self.assertEqual(runner.run()["status"],"complete")
        self.assertEqual(next(iter(runner._jobs().values()))["rounds"],2)

    def test_budget_exhaustion_is_skipped_and_listed(self):
        self.sage.tool_resolver=lambda *a:call("delete_database",{})
        result=self.sage.run()
        self.assertEqual(result["status"],"complete_with_skipped_failures")
        skipped=result["skipped_failures"]["sage"]
        self.assertEqual([f["stage"] for f in skipped],["event_tools"])
        self.assertEqual(skipped[0]["event_id"],self.root)
        self.assertEqual(result["completion_gate"]["coverage"]["skipped_event_jobs"],1)
        self.assertFalse(rows(self.m,"sage_events"))
        j=next(iter(self.sage._jobs().values()))
        self.assertEqual(j["calls"],12)
        self.sage.run()
        self.assertEqual(next(iter(self.sage._jobs().values()))["rounds"],12)

    def test_stale_response_cannot_write_event(self):
        def stale(messages,tools,eid):
            self.m.update_event(replace(self.m.get_event(eid),state="changed by another writer"))
            return model(messages,tools,eid)
        self.sage.tool_resolver=stale
        job=self.sage.process_event(self.root)
        self.assertEqual(job["status"],"stale")
        self.assertFalse(rows(self.m,"sage_events"))

    def test_fact_cannot_cite_unknown_evidence(self):
        j=self.sage._job(self.root)
        with self.m.transaction():self.sage._execute(j,"SummarizeEvent",summarization(j["context"]))
        with self.assertRaisesRegex(ValueError,"visible Event observations"):
            with self.m.transaction():self.sage._execute(j,"ExtractFacts",{"facts":[{"text":"x","tags":[],"time":None,"support":["invented"]}]})
        self.assertFalse(rows(self.m,"sage_facts"))

    def test_context_budget_does_not_silently_extract_from_sample(self):
        j=self.sage._job(self.root);j["context"]["truncated"]=True
        with self.assertRaisesRegex(ValueError,"Complete member evidence"):
            self.sage._execute(j,"SummarizeEvent",summarization(j["context"]))

    def test_tool_budgets_are_strict(self):
        for setting in ({"max_rounds":True},{"max_calls":1.5},{"max_rounds":0},{"split_similarity":2},{"unknown":2}):
            with self.assertRaises(ValueError):tool_config(setting)

    def test_native_transport_sends_tools_and_forwards_error_results(self):
        self.sage.tool_resolver=None
        replies=[]
        def endpoint(cfg,payload):
            self.assertEqual({t["function"]["name"] for t in payload["tools"]},{"SplitEvent","SummarizeEvent","ExtractFacts"})
            self.assertFalse(payload["parallel_tool_calls"])
            replies.append(payload)
            response=model(payload["messages"],payload["tools"],self.root)
            return {"choices":[{"finish_reason":"tool_calls","message":response}]}
        with patch("hypermem.online.sage_tools.request_json",side_effect=endpoint):
            self.assertEqual(self.sage.run()["status"],"complete")
        self.assertEqual(len(replies),2)
        self.assertEqual(replies[1]["messages"][-1]["role"],"tool")

    def test_malformed_arguments_and_plain_text_cannot_complete_job(self):
        responses=iter([{"role":"assistant","content":"done"},call("SummarizeEvent",{}),
            {"role":"assistant","tool_calls":[{"id":"bad","type":"function","function":{"name":"SummarizeEvent","arguments":"{broken"}}]}])
        self.sage.tool_resolver=lambda *a:next(responses,call("ExtractFacts",{"facts":[]}))
        result=self.sage.run()
        self.assertEqual(result["status"],"complete_with_skipped_failures")
        self.assertEqual(next(iter(self.sage._jobs().values()))["status"],"budget_exhausted")
        self.assertFalse(rows(self.m,"sage_events"))


if __name__=="__main__":unittest.main()
