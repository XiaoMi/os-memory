# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Budgeted four-index retrieval over SAGE events, anchors, facts and memory."""
from collections import defaultdict
import json
import numpy as np

from hypermem.online.clustering import similarity, unit
from hypermem.online.sage import row


class HyperOSRetriever:
    def __init__(self,sage):
        self.s=sage; self.m=sage.m; self.cfg=sage.cfg["retrieval"]
        self._index=None;self._index_revision=None

    def _revision(self):
        return (self.m._connection.total_changes,
                self.m._connection.execute("PRAGMA data_version").fetchone()[0])

    def _load_index(self):
        revision=self._revision()
        if self._index is not None and revision==self._index_revision:return self._index
        db=self.m._connection
        # Stream vectors directly into compact float64 arrays. A frozen QA
        # snapshot reuses these arrays, with the exact original score function.
        embeddings={}
        for nid,raw in db.execute("SELECT node_id,record_json FROM sage_embeddings"):
            item=json.loads(raw)
            embeddings[nid]={"kind":item["kind"],"embedding":np.asarray(item["embedding"],dtype=float)}
        event_to_anchors=defaultdict(set);anchor_to_events=defaultdict(set)
        for eid,aid in db.execute("SELECT event_id,anchor_id FROM sage_event_anchors"):
            event_to_anchors[eid].add(aid);anchor_to_events[aid].add(eid)
        fact_owner=dict(db.execute("SELECT fact_id,json_extract(record_json,'$.event_id') FROM sage_facts"))
        fact_by_event=defaultdict(set)
        for fid,eid in fact_owner.items():fact_by_event[eid].add(fid)
        support={mid:set(json.loads(raw)) for mid,raw in db.execute(
            "SELECT memory_id,json_extract(record_json,'$.support_events') FROM sage_long_memory")}
        events={r[0] for r in db.execute("SELECT event_id FROM sage_events")}
        if self._revision()!=revision:raise RuntimeError("Memory changed while loading retrieval index; retry with a stable snapshot")
        self._index={"embeddings":embeddings,"event_to_anchors":event_to_anchors,
                     "anchor_to_events":anchor_to_events,"fact_owner":fact_owner,
                     "fact_by_event":fact_by_event,"support":support,"events":events}
        self._index_revision=revision
        return self._index

    def retrieve(self,query):
        raw=self.s.encoder.encode([query])
        if len(raw)!=1:raise ValueError("Query encoder returned wrong count")
        q=unit(np.asarray(raw[0])-np.asarray(self.s.center["mean"]))
        index=self._load_index();embeddings=index["embeddings"]
        by_kind=defaultdict(list)
        for nid,item in embeddings.items():by_kind[item["kind"]].append((similarity(q,item["embedding"]),nid))
        scores={}
        seed_k=self.cfg["seed_k"]
        for kind in ("fact","event","anchor","longmem"):
            for score,nid in sorted(by_kind[kind],reverse=True)[:seed_k]:scores[nid]=max(scores.get(nid,-1),score)
        event_to_anchors=index["event_to_anchors"];anchor_to_events=index["anchor_to_events"]
        fact_by_event=index["fact_by_event"];fact_owner=index["fact_owner"];support=index["support"]
        frontier=dict(sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:self.cfg["budget_b"]])
        for _ in range(self.cfg["hop_h"]):
            expanded=dict(frontier)
            for nid,score in frontier.items():
                kind=embeddings[nid]["kind"]
                neighbors=[]
                if kind=="anchor":neighbors=[(eid,score/max(1,np.log1p(len(anchor_to_events[nid])))) for eid in anchor_to_events[nid]]
                elif kind=="event":neighbors=[(x,score) for x in fact_by_event[nid]|event_to_anchors[nid]]
                elif kind=="fact":neighbors=[(fact_owner[nid],score)]
                elif kind=="longmem":neighbors=[(eid,score) for eid in support.get(nid,set())]
                for other,value in neighbors:
                    if other in embeddings:expanded[other]=max(expanded.get(other,-1),value)
            frontier=dict(sorted(expanded.items(),key=lambda x:(-x[1],x[0]))[:self.cfg["budget_b"]])
        events=index["events"]
        selected=sorted((nid for nid in frontier if nid in events),key=lambda eid:(self.s.events[eid].start_time,eid))
        return {"query":query,"node_scores":frontier,"events":[{"event_id":eid,**row(self.m,"sage_events",eid),
            "span":[self.s.events[eid].start_time.isoformat(),self.s.events[eid].end_time.isoformat()],
            "facts":[row(self.m,"sage_facts",fid) for fid in sorted(fact_by_event[eid])]} for eid in selected],
            "long_memories":[row(self.m,"sage_long_memory",mid) for mid in frontier if mid in support],
            "note":"LongMemory → Event → Fact, with observation support retained in every Fact."}
