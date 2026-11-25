# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""DRED: dual-rate, evidence-deferred online event attribution (HyperOS-Mem fast path).

Corpus centering is explicitly a two-pass experiment. Fixed centering accepts
new observations causally; its mean must come from an earlier calibration set.
All event/state/prototype/decision writes for an arrival are one transaction.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import numpy as np

from hypermem.config import check_resumable, config, resolve_services
from hypermem.grounding import ground, digest, identity_anchors
from hypermem.memory.writer_lock import engine_writer
from hypermem.schema import CanonicalEvent, ObservationFeatures
from hypermem.schema.serialization import encode_json
from hypermem.standardization import make_encoder, observation_text
from hypermem.online.clustering import unit, similarity, cohesion, topic_split


DDL = (
    "CREATE TABLE IF NOT EXISTS on_vectors (observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id), vector_json TEXT NOT NULL, grounding_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS on_events (event_id TEXT PRIMARY KEY REFERENCES events(event_id), runtime_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS on_states (observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id), state_json TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS on_relations (source TEXT NOT NULL REFERENCES events(event_id), target TEXT NOT NULL REFERENCES events(event_id), kind TEXT NOT NULL, evidence_json TEXT NOT NULL, PRIMARY KEY(source,target,kind))",
    "CREATE TABLE IF NOT EXISTS on_splits (parent_id TEXT PRIMARY KEY REFERENCES events(event_id), trace_json TEXT NOT NULL)",
)


class OnlineHyperMem:
    """DRED event memory. SAGE (hypermem.online.consolidation + sage_tools) refines it offline."""

    @engine_writer
    def __init__(self, memory, settings=None, *, encoder=None, slow_resolver=None):
        self.memory = memory
        self.settings = resolve_services(config(settings))
        previous = memory.get_metadata("config")
        check_resumable(previous, self.settings)
        if previous is None and memory.counts()["events"]:
            raise ValueError("Online semantic mode requires a new event memory")
        self.cfg = self.settings["online"]
        self.slow_resolver = slow_resolver
        self.encoder = encoder or make_encoder(self.settings["embedding"])
        memory._connection.execute("PRAGMA journal_mode=WAL")
        memory._connection.execute("PRAGMA synchronous=FULL")
        with memory.transaction():
            for statement in DDL:
                memory._connection.execute(statement)
            from hypermem.online.consolidation import DDL as CONSOLIDATION_DDL
            for statement in CONSOLIDATION_DDL:
                memory._connection.execute(statement)
            memory.set_metadata("config",self.settings)
        self._reload()

    def _reload(self):
        db = self.memory._connection
        self.runtime = {r[0]:json.loads(r[1]) for r in db.execute("SELECT * FROM on_events")}
        self.states = {r[0]:json.loads(r[1]) for r in db.execute("SELECT * FROM on_states")}
        self._pending_ids = {oid for oid,s in self.states.items() if s["status"]=="pending"}
        # Each JSON vector can span many SQLite pages. Load grounding alongside
        # it rather than scanning the same multi-GB table a second time.
        self.raw, self.g = {}, {}
        for oid, vector, grounding in db.execute("SELECT * FROM on_vectors"):
            self.raw[oid] = np.asarray(json.loads(vector))
            self.g[oid] = json.loads(grounding)
        self.center = self.memory.get_metadata("online_center")
        self.z = {r[0]:np.asarray(json.loads(r[1])) for r in db.execute("SELECT observation_id,json_extract(features_json,'$.embedding') FROM observation_features")}
        self.watermark = self.memory.get_metadata("online_watermark")
        self.mainline = self.memory.get_metadata("online_mainline")
        self._index_keys = None
        self._reindex()
        self._data_version = db.execute("PRAGMA data_version").fetchone()[0]

    def _refresh_if_changed(self):
        if self.memory._connection.execute("PRAGMA data_version").fetchone()[0]!=self._data_version:
            self._reload()

    def _reindex(self):
        if self._index_keys is None:
            self.event_index = defaultdict(set)
            self.entity_index = defaultdict(set)
            self.strong_index = defaultdict(set)
            self.active = set()
            self._index_keys = {}
            self._dirty_runtime = set(self.runtime)
        for eid in self._dirty_runtime:
            r = self.runtime[eid]
            old_event,old_entity = self._index_keys.pop(eid,((),()))
            for index,keys in ((self.event_index,old_event),(self.entity_index,old_entity),
                               (self.strong_index,set(old_event)|set(old_entity))):
                for key in keys:
                    index[key].discard(eid)
                    if not index[key]:del index[key]
            self.active.discard(eid)
            if r["lifecycle"] in {"superseded", "archived"}:
                continue
            if r["lifecycle"] == "active":
                self.active.add(eid)
            event_keys = tuple(r.get("event_keys",r.get("strong_keys",[])))
            entity_keys = tuple(r.get("entity_keys",[]))
            self._index_keys[eid] = (event_keys,entity_keys)
            for key in event_keys:
                self.event_index[key].add(eid)
                self.strong_index[key].add(eid)
            for key in entity_keys:
                self.entity_index[key].add(eid)
                self.strong_index[key].add(eid)
        self._dirty_runtime.clear()

    @contextmanager
    def _transaction(self):
        try:
            with self.memory.transaction():
                yield
        except BaseException:
            self._reload()
            raise
        self._reindex()

    def _save_runtime(self,eid,r,*,sync_semantics=True):
        self.memory._connection.execute("INSERT INTO on_events VALUES (?,?) ON CONFLICT(event_id) DO UPDATE SET runtime_json=excluded.runtime_json",(eid,encode_json(r)))
        self.runtime[eid] = r
        self._dirty_runtime.add(eid)
        if sync_semantics:
            from hypermem.online.consolidation import sync_entry
            sync_entry(self,eid,r)

    def _save_state(self,oid,s):
        self.memory._connection.execute("INSERT INTO on_states VALUES (?,?) ON CONFLICT(observation_id) DO UPDATE SET state_json=excluded.state_json",(oid,encode_json(s)))
        self.states[oid] = s
        if s["status"]=="pending":self._pending_ids.add(oid)
        else:self._pending_ids.discard(oid)

    def _key_tiers(self,oid):
        """Separate unique event IDs from reusable entity-level identifiers."""
        grounded=self.g[oid]
        event=sorted(encode_json([a["kind"],a["scope"],a["value"]])
                     for a in identity_anchors(grounded).values())
        entity=set()
        for item in grounded.get("anchor_evidence",[]):
            # A chat session_id is scoped to its app, like a phone number: a candidate, not a hard constraint.
            if item.get("kind") in {"phone","account_id","contact_id","session_id"} and item.get("quality",0)>=0.8:
                entity.add(encode_json([item["kind"],item.get("scope") or grounded.get("app","unknown"),item["value"]]))
        for item in grounded.get("entities",[]):
            if item.get("quality",0)>=0.8 and item.get("role") in {"participant","merchant","organizer","contact"}:
                entity.add(encode_json(["entity",item.get("role"),item["value"]]))
        return event,sorted(entity-set(event))

    def _keys(self,oid):
        event,entity=self._key_tiers(oid)
        return event+entity

    def _alpha(self,oid):
        g = self.g[oid]
        if g["app"] in self.cfg["chat_apps"] or g["original_app"] in self.cfg["chat_apps"] or "communicate" in g["actions"]:
            return self.cfg["alpha_chat"]
        if g["app"] in self.cfg["stream_apps"] or g["original_app"] in self.cfg["stream_apps"]:
            return self.cfg["alpha_stream"]
        return self.cfg["alpha_default"]

    @engine_writer
    def prepare(self, observations):
        observations = list(observations)
        if self.center and all(o.observation_id in self.z for o in observations):
            return
        missing = [o for o in observations if o.observation_id not in self.raw]
        if missing and self.center and self.cfg["mean_mode"] == "corpus":
            raise ValueError("Corpus mean is frozen; new inputs require a new run or fixed-mean online mode")
        size = self.settings["embedding"]["batch_size"]
        for start in range(0,len(missing),size):
            batch = missing[start:start+size]
            grounded = [ground(o,aliases=self.settings["standardization"]["entity_aliases"],max_chars=self.settings["standardization"]["max_text_chars"]) for o in batch]
            for o,g in zip(batch,grounded):
                if not g["text"]:
                    g["text"] = observation_text(o,self.settings["standardization"]["max_text_chars"])
                    g["online_text_fallback"] = "supplied_activity_label" if g["text"] else "empty"
            vectors = self.encoder.encode([g["text"] for g in grounded])
            if len(vectors) != len(batch):
                raise ValueError("Embedding response count differs from inputs")
            vectors = [unit(v) for v in vectors]
            dimension = len(next(iter(self.raw.values()))) if self.raw else len(vectors[0])
            if any(len(v)!=dimension for v in vectors):
                raise ValueError("Embedding dimension changed")
            with self._transaction():
                for o,g,v in zip(batch,grounded,vectors):
                    raw = self.memory.resolve_provenance(o.observation_id)
                    user = raw.get("payload",raw).get("user_id",o.provenance.metadata.get("user_id"))
                    if user is not None:
                        old = self.memory.get_metadata("user_id")
                        if old is not None and user != old:
                            raise ValueError("Mixed users in one memory")
                        self.memory.set_metadata("user_id",user)
                    self.memory._connection.execute("INSERT INTO on_vectors VALUES (?,?,?)",(o.observation_id,encode_json(v.tolist()),encode_json(g)))
                    self.raw[o.observation_id],self.g[o.observation_id] = v,g
        if self.cfg["mean_mode"] == "fixed":
            if not self.center:
                self._freeze_center(np.asarray(self.cfg["fixed_mean"]), [], "fixed_prior_calibration")
            self._center_features()

    def _freeze_center(self,mean,ids,scope):
        if mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all():
            raise ValueError("Invalid centering mean")
        if self.raw and len(mean) != len(next(iter(self.raw.values()))):
            raise ValueError("Mean and encoder dimensions differ")
        value = {"mean":mean.tolist(),"dimension":len(mean),"mode":self.cfg["mean_mode"],
                 "scope":scope,"calibration_count":len(ids),"calibration_ids_sha256":digest(sorted(ids)),
                 "encoder_output":"unit_normalized_f_before_mean", "strict_online":self.cfg["mean_mode"]=="fixed"}
        with self._transaction():
            self.memory.set_metadata("online_center",value)
            self.center = value

    def _center_features(self):
        mean = np.asarray(self.center["mean"])
        with self._transaction():
            for oid,v in self.raw.items():
                if oid in self.z:
                    continue
                if v.shape != mean.shape:
                    raise ValueError("Mean and encoder dimensions differ")
                z = unit(v-mean) if self.g[oid]["text"] else np.zeros_like(mean)
                self.z[oid] = z
                g = self.g[oid]
                self.memory.set_features(oid,ObservationFeatures(z.tolist(),sorted({e["value"] for e in g["entities"]}),
                    {"grounding":g,"text":g["text"],"anchors":g["anchors"],"encoder":self.settings["embedding"],
                     "centering_digest":digest(self.center),"zero_residual":bool(np.linalg.norm(z)<1e-12)}))

    def _expire_active(self,now):
        cutoff = now-timedelta(minutes=self.cfg["active_minutes"])
        for eid in list(self.active):
            r = self.runtime[eid]
            if datetime.fromisoformat(r["last_activity"]) < cutoff:
                self._save_runtime(eid,{**r,"lifecycle":"dormant"})
        self._reindex()

    def _current_mainline(self):
        if self.mainline in self.active:
            return self.mainline
        return max(self.active,key=lambda eid:(self.runtime[eid]["last_activity"],eid)) if self.active else None

    def _event(self,eid,ids,prototype,confidence):
        observations = sorted((self.memory.get_observation(oid) for oid in ids),key=lambda o:(o.start_time,o.observation_id))
        core = [o for o in observations if self.states.get(o.observation_id,{}).get("role")!="incidental"] or observations
        # A representative excerpt is a display label, not an LLM summary.
        representative = max(core,key=lambda o:(similarity(self.z[o.observation_id],prototype),o.start_time,o.observation_id))
        anchors = defaultdict(set)
        for o in observations:
            for k,v in self.g[o.observation_id]["anchors"].items():
                anchors[k].update(v)
        return CanonicalEvent(eid,min(o.start_time for o in observations),max(o.end_time for o in observations),
            self.g[representative.observation_id]["text"][:180] or "无文本观测",
            sorted({e["value"] for o in observations for e in self.g[o.observation_id]["entities"]}),
            {k:sorted(v) for k,v in anchors.items()},[o.observation_id for o in observations],
            float(np.clip(confidence,0,1)),list(prototype))

    def _new(self,ids,now,*,isolated=False,mainline=None,reason="pending_confirmation",mainlines=None,suggestion_id=None):
        eid = "E-on-"+digest(sorted(ids))[:24]
        prototype = unit(np.sum([self.z[oid] for oid in ids],axis=0)).tolist()
        event_keys=sorted({k for oid in ids for k in self._key_tiers(oid)[0]})
        entity_keys=sorted({k for oid in ids for k in self._key_tiers(oid)[1]})
        r = {"prototype":prototype,"lifecycle":"isolated" if isolated else "active",
             "last_activity":now.isoformat(),"event_keys":event_keys,"entity_keys":entity_keys,
             "strong_keys":sorted(set(event_keys)|set(entity_keys)),
             "core_ids":list(ids),"mainline_id":mainline,"parent_id":None}
        for oid in ids:
            self._save_state(oid,{"status":"assigned","event_id":eid,"role":"isolated" if isolated else "core",
                                 "mainline_id":(mainlines or {}).get(oid,mainline),"reason":reason})
        confidence = max(0,similarity(self.z[ids[0]],self.z[ids[-1]])) if len(ids)>1 else 0.0
        self.memory.add_event(self._event(eid,ids,prototype,confidence))
        self._save_runtime(eid,r)
        if mainline and mainline != eid:
            self._relation(eid,mainline,"branched_from",{"observation_ids":ids,"meaning":"captured_mainline_context_not_causation"})
        if mainlines:
            for parent in sorted({p for p in mainlines.values() if p and p != eid}):
                self._relation(eid,parent,"branched_from",{"observation_ids":[o for o in ids if mainlines.get(o)==parent],
                    "meaning":"captured_mainline_context_not_causation"})
        for oid in ids:
            self.memory.log_decision(oid,{"formation_decision":"NEW","selected_event":eid,"reason":reason,
                "prototype_after":prototype,"member_ids":ids,"mainline_id":mainline,
                **({"suggestion_id":suggestion_id} if suggestion_id else {})})
        return eid

    def _relation(self,source,target,kind,evidence):
        self.memory._connection.execute("INSERT OR REPLACE INTO on_relations VALUES (?,?,?,?)",(source,target,kind,encode_json(evidence)))

    def _attach(self,oid,eid,now,*,incidental=False,trace=None):
        old = self.memory.get_event(eid)
        r = self.runtime[eid]
        before = list(r["prototype"])
        alpha = 0.0 if incidental else self._alpha(oid)
        prototype = before if incidental else unit((1-alpha)*np.asarray(before)+alpha*self.z[oid]).tolist()
        reason = (trace or {}).get("route", "pending_timeout_mainline") if incidental else trace["route"]
        self._save_state(oid,{"status":"assigned","event_id":eid,"role":"incidental" if incidental else "core",
                             "mainline_id":self.states.get(oid,{}).get("mainline_id",self._current_mainline()),"reason":reason})
        self.memory.update_event(self._event(eid,old.observation_ids+[oid],prototype,old.confidence))
        event_keys,entity_keys=self._key_tiers(oid)
        updated_event=(r.get("event_keys",r.get("strong_keys",[])) if incidental else
                       sorted(set(r.get("event_keys",r.get("strong_keys",[])))|set(event_keys)))
        updated_entity=(r.get("entity_keys",[]) if incidental else
                        sorted(set(r.get("entity_keys",[]))|set(entity_keys)))
        self._save_runtime(eid,{**r,"prototype":prototype,
            "lifecycle":r["lifecycle"] if incidental else "active",
            "last_activity":r["last_activity"] if incidental else now.isoformat(),
            "event_keys":updated_event,"entity_keys":updated_entity,
            "strong_keys":sorted(set(updated_event)|set(updated_entity)),
            "core_ids":r["core_ids"] if incidental else r["core_ids"]+[oid]})
        self.memory.log_decision(oid,{"formation_decision":"ATTACH" if incidental else "MATCH","selected_event":eid,
            "reason":reason,"prototype_before":before,"prototype_after":prototype,"alpha":alpha,
            "was_dormant":r["lifecycle"]=="dormant","evidence":trace or {}})

    def _drain_pending(self,now,final=False):
        cutoff = now-timedelta(minutes=self.cfg["pending_minutes"])
        due = sorted((self.states[oid]["arrival"],oid) for oid in self._pending_ids
                     if final or datetime.fromisoformat(self.states[oid]["arrival"])<cutoff)
        for _,oid in due:
            parent = self.states[oid]["mainline_id"]
            if parent and self.runtime[parent]["lifecycle"]!="superseded":
                self._attach(oid,parent,now,incidental=True)
            else:
                self._new([oid],self.memory.get_observation(oid).end_time,isolated=True,reason="pending_timeout_isolated")

    @engine_writer
    def process(self,observation,**kwargs):
        if kwargs:
            raise ValueError("Online mode processes timestamp-ordered arrivals without oracle/retry flags")
        self._refresh_if_changed()
        oid = observation.observation_id
        if oid in self.states:
            return self.states[oid]
        now = observation.start_time
        if self.watermark and now < datetime.fromisoformat(self.watermark):
            raise ValueError("Out-of-order arrival; sort inputs or start a replay run")
        self.prepare([observation])
        if not self.center or oid not in self.z:
            raise ValueError("Fit the corpus center via replay, or supply a fixed prior mean")
        with self._transaction():
            self._expire_active(now)
            mainline = self._current_mainline()
            event_keys,entity_keys = self._key_tiers(oid)
            event_candidates = set().union(*(self.event_index[k] for k in event_keys)) if event_keys else set()
            entity_candidates = ((set().union(*(self.entity_index[k] for k in entity_keys)) if entity_keys else set())
                                 - event_candidates)
            candidates = self.active | event_candidates | entity_candidates
            scores = {eid:similarity(self.z[oid],self.runtime[eid]["prototype"]) for eid in candidates}
            tiers=((event_candidates,float("-inf"),"event_identifier"),
                   (entity_candidates,self.cfg["tau_s"],"entity_identifier"),
                   (set(self.active)-event_candidates-entity_candidates,self.cfg["tau_a"],"semantic"))
            best=route=None
            ranked_all=[]
            for pool,tau,name in tiers:
                ranked=sorted(pool,key=lambda eid:(-scores[eid],-datetime.fromisoformat(self.runtime[eid]["last_activity"]).timestamp(),eid))
                ranked_all.extend(eid for eid in ranked if eid not in ranked_all)
                if ranked and scores[ranked[0]]>=tau:
                    best,route=ranked[0],name
                    break
            if best:
                matched=(set(event_keys)&set(self.runtime[best].get("event_keys",self.runtime[best].get("strong_keys",[]))) |
                         set(entity_keys)&set(self.runtime[best].get("entity_keys",[])))
                trace = {"route":route,"score":scores[best],"candidates":scores,
                         "event_candidates":sorted(event_candidates),"entity_candidates":sorted(entity_candidates),
                         "matched_keys":sorted(matched)}
                self._attach(oid,best,now,trace=trace)
                if len(ranked_all)>1:
                    other=next((eid for eid in ranked_all if eid!=best),None)
                    margin = scores[best]-scores[other] if other else 1.0
                    if margin < self.cfg["consolidation"]["ambiguity_margin"]:
                        self.memory._connection.execute("INSERT INTO on_uncertainty VALUES (?,?)",
                            (oid,encode_json({"observation_id":oid,"fast_event":best,"scores":scores,
                                "margin":margin,"threshold":self.cfg["consolidation"]["ambiguity_margin"],
                                "matched_keys":trace["matched_keys"],"arrival":now.isoformat()})))
                self.mainline = best
            else:
                pending = [(similarity(self.z[oid],self.z[pid]),pid) for pid in self._pending_ids
                           if 0<=(now-datetime.fromisoformat(self.states[pid]["arrival"])).total_seconds()<=self.cfg["pending_minutes"]*60]
                pending.sort(key=lambda p:(-p[0],p[1]))
                if pending and pending[0][0]>=self.cfg["tau_p"]:
                    pid = pending[0][1]
                    parent = self.states[pid]["mainline_id"] or mainline
                    self.mainline = self._new([pid,oid],now,mainline=parent,
                        mainlines={pid:self.states[pid]["mainline_id"],oid:mainline})
                else:
                    self._save_state(oid,{"status":"pending","event_id":None,"role":"pending","mainline_id":mainline,
                                         "arrival":now.isoformat(),"reason":"awaiting_topic_confirmation"})
                    self.memory.log_decision(oid,{"formation_decision":"PENDING","selected_event":None,
                        "mainline_id":mainline,"candidates":scores,"best_pending_score":pending[0][0] if pending else None})
            self._drain_pending(now)
            self.watermark = now.isoformat()
            self.memory.set_metadata("online_watermark",self.watermark)
            self.memory.set_metadata("online_mainline",self.mainline)
            self.memory.set_metadata("online_complete",False)
        return self.states[oid]

    @engine_writer
    def finalize(self):
        """End of the DRED pass: settle the pending set, then topic-split incohesive events."""
        self._refresh_if_changed()
        if self.watermark:
            with self._transaction():
                self._drain_pending(datetime.fromisoformat(self.watermark),final=True)
        splits = self.split_topics() if self.cfg["split_enabled"] else []
        with self.memory.transaction():
            self.memory.set_metadata("online_complete",all(s["status"]=="assigned" for s in self.states.values()))
        return splits

    def run_slow(self,*,retry_unresolved=False,is_idle=False,as_of=None,progress=None):
        """First step of the SAGE slow path: idle boundary consolidation over DRED events."""
        from hypermem.online.consolidation import Consolidation
        from hypermem.memory.writer_lock import writer_lock, WriterBusyError
        # Exclude a second slow worker, but release the actual memory writer
        # lock during HTTP reasoning so DRED can continue on its connection.
        lock = self.memory.path+".idle-worker" if self.memory.path!=":memory:" else ":memory:"
        try:
            with writer_lock(lock):
                return Consolidation(self,progress=progress).run(is_idle=is_idle,as_of=as_of,
                    retry_exhausted=retry_unresolved)
        except WriterBusyError:
            return {"status":"paused_writer_busy"}

    @engine_writer
    def revoke_soft_relation(self,relation_id,reason):
        from hypermem.online.consolidation import revoke
        revoke(self,relation_id,reason)

    @engine_writer
    def split_topics(self):
        self._refresh_if_changed()
        performed = []
        queue = list(self.runtime)
        for eid in queue:
            r = self.runtime[eid]
            if r["lifecycle"] in {"superseded", "archived"}:
                continue
            old = self.memory.get_event(eid)
            eligible = [oid for oid in old.observation_ids if np.linalg.norm(self.z[oid])>1e-12]
            if len(old.observation_ids)<self.cfg["split_min_members"]:
                continue
            # Step 12 examines the complete event, including attached side
            # observations. Two consistent side observations can therefore
            # become a confirmed subtopic during the explicit splitting phase.
            split_cfg = {**self.cfg,"split_min_members":2*self.cfg["split_min_cluster"]}
            result = topic_split([self.z[oid] for oid in eligible],split_cfg) if eligible else None
            if not result:
                continue
            groups = [[eligible[i] for i in indexes] for indexes in result["groups"]]
            prototypes = [unit(np.mean([self.z[oid] for oid in group],axis=0)) for group in groups]
            extra = [oid for oid in old.observation_ids if oid not in eligible]
            members = [list(group) for group in groups]
            for oid in extra:
                label = max(range(len(groups)),key=lambda i:(similarity(self.z[oid],prototypes[i]),-i))
                members[label].append(oid)
            with self._transaction():
                self.memory._connection.execute("DELETE FROM event_observations WHERE event_id=?",(eid,))
                self._save_runtime(eid,{**r,"lifecycle":"superseded"})
                children = []
                for group,ids,prototype in zip(groups,members,prototypes):
                    child = "E-split-"+digest([eid,sorted(ids)])[:24]
                    children.append(child)
                    for oid in ids:
                        self._save_state(oid,{**self.states[oid],"event_id":child,"split_parent":eid,
                            "pre_split_role":self.states[oid]["role"],
                            "role":("isolated" if len(group)==1 else "core") if oid in group else "incidental"})
                    self.memory.add_event(self._event(child,ids,prototype,old.confidence))
                    last = max(self.memory.get_observation(oid).start_time for oid in group)
                    active = self.watermark is not None and (datetime.fromisoformat(self.watermark)-last).total_seconds()<=60*self.cfg["active_minutes"]
                    event_keys=sorted({k for oid in group for k in self._key_tiers(oid)[0]})
                    entity_keys=sorted({k for oid in group for k in self._key_tiers(oid)[1]})
                    self._save_runtime(child,{"prototype":prototype.tolist(),"lifecycle":"isolated" if len(group)==1 else "active" if active else "dormant",
                        "last_activity":last.isoformat(),"event_keys":event_keys,"entity_keys":entity_keys,
                        "strong_keys":sorted(set(event_keys)|set(entity_keys)),
                        "core_ids":group,"mainline_id":r["mainline_id"],"parent_id":eid})
                    self._relation(child,eid,"split_from",{"member_ids":ids,"method":result["method"]})
                    for oid in ids:
                        self.memory.log_decision(oid,{"formation_decision":"SPLIT","selected_event":child,"parent_id":eid,
                            "role":self.states[oid]["role"],"prototype_after":prototype.tolist()})
                trace = {**result,"parent_id":eid,"child_ids":children,"clustered_observation_ids":eligible,
                         "historical_event":old.to_dict(),"scope":"all_event_members_with_nonzero_semantic_representation",
                         "zero_residual_assignment":"nearest_cluster_without_prototype_update; ties go to first cluster"}
                self.memory._connection.execute("INSERT INTO on_splits VALUES (?,?)",(eid,encode_json(trace)))
                if self.mainline==eid:
                    self.mainline = max(children,key=lambda c:(self.runtime[c]["last_activity"],c))
                    self.memory.set_metadata("online_mainline",self.mainline)
                performed.append(trace)
                queue.extend(children)  # Recheck large, still-incohesive children.
        return performed

    @engine_writer
    def replay(self,*,progress=None,oracle_gold=None,detailed_progress=True):
        if oracle_gold is not None:
            raise ValueError("Gold labels cannot enter event construction")
        observations = list(self.memory.iter_observations())
        if not observations:
            raise ValueError("No observations to replay")
        self.prepare(observations)
        if not self.center:
            ids = [o.observation_id for o in observations]
            self._freeze_center(np.mean([self.raw[oid] for oid in ids],axis=0),ids,"all_selected_inputs_two_pass_not_strict_online")
        self._center_features()
        for count,o in enumerate(observations,1):
            self.process(o)
            if progress and count%max(1,self.settings["replay"]["progress_every"])==0:
                if detailed_progress:
                    report = self.report()
                else:
                    # Operational counters only; the full report and audits still
                    # run at EOF. Never reread all event vectors just to log progress.
                    report = {"algorithm":"online-semantic-v1", "observations":len(observations),
                              "processed":len(self.states), "pending":len(self._pending_ids),
                              "events":sum(r["lifecycle"]!="superseded" for r in self.runtime.values()),
                              "active":len(self.active), "progress_detail":"counters_only_not_audit"}
                progress(count,report)
        self.finalize()
        return self.report()

    def report(self):
        current = {eid:r for eid,r in self.runtime.items() if r["lifecycle"]!="superseded"}
        roles = Counter(s["role"] for s in self.states.values())
        sizes = [len(self.memory.get_event(eid).observation_ids) for eid in current]
        result = {"algorithm":"online-semantic-v1","observations":self.memory.counts()["observations"],
                "processed":len(self.states),"pending":roles.get("pending",0),"roles":dict(roles),
                "events":len(current),"active":len(self.active),"isolated_events":sum(r["lifecycle"]=="isolated" for r in current.values()),
                "superseded_events":len(self.runtime)-len(current),"largest_event":max(sizes,default=0),
                "semantic_quality":"unlabeled_experiment_not_accuracy", "center":{k:v for k,v in (self.center or {}).items() if k!="mean"}}
        from hypermem.online.consolidation import summary
        result["consolidation"] = summary(self.memory)
        return result
