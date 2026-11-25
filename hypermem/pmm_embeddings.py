# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Bounded concurrent, exact-text raw embedding cache; observations stay intact."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json

from hypermem.grounding import ground, digest
from hypermem.standardization import make_encoder, observation_text


class TextCacheEncoder:
    def __init__(self, memory, settings):
        self.memory, self.settings = memory, settings
        self.calls = 0

    def encode(self, texts):
        vectors = []
        for text in texts:
            key = hashlib.sha256(text.encode()).hexdigest()
            row = self.memory._connection.execute("SELECT text,vector_json FROM pmm_text_vectors WHERE text_hash=?",(key,)).fetchone()
            if row is None or row[0] != text:
                raise ValueError("Missing exact-text embedding cache entry")
            vectors.append(json.loads(row[1]))
        return vectors


def prefetch(memory, settings, progress=None, workers=8):
    db = memory._connection
    with memory.transaction():
        db.execute("CREATE TABLE IF NOT EXISTS pmm_text_vectors (text_hash TEXT PRIMARY KEY,text TEXT NOT NULL,vector_json TEXT NOT NULL)")
        previous = memory.get_metadata("pmm_cache_encoder")
        if previous is not None and previous != settings["embedding"]:
            raise ValueError("Exact-text cache encoder changed")
        memory.set_metadata("pmm_cache_encoder",settings["embedding"])
    cached = {r[0] for r in db.execute("SELECT text_hash FROM pmm_text_vectors")}
    texts = {}
    total = 0
    for observation in memory.iter_observations():
        total += 1
        g = ground(observation,aliases=settings["standardization"]["entity_aliases"],max_chars=settings["standardization"]["max_text_chars"])
        text = g["text"] or observation_text(observation,settings["standardization"]["max_text_chars"])
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in cached:texts[key] = text
    missing = list(texts.items())
    size = settings["embedding"]["batch_size"]
    batches = iter(missing[i:i+size] for i in range(0,len(missing),size))
    def encode(batch):
        vectors = make_encoder(settings["embedding"]).encode([text for _,text in batch])
        if any(len(v)!=settings["embedding"]["dimension"] for v in vectors):
            raise ValueError("Embedding service dimension differs from Config A")
        return batch,vectors
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(workers):
            batch = next(batches,None)
            if batch is not None:pending.add(pool.submit(encode,batch))
        while pending:
            completed,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                batch,vectors = future.result()
                with memory.transaction():
                    db.executemany("INSERT INTO pmm_text_vectors VALUES (?,?,?)",
                                   [(key,text,json.dumps(v,allow_nan=False)) for (key,text),v in zip(batch,vectors)])
                done += len(batch)
                if progress:progress({"stage":"embedding_cache","observations":total,"cached_texts":len(cached),
                                      "unique_missing_texts":len(missing),"completed_texts":done})
                batch = next(batches,None)
                if batch is not None:pending.add(pool.submit(encode,batch))
    report = {"observations":total,"unique_texts":len(cached)+len(missing),"new_vectors":done,
              "encoder_digest":digest(settings["embedding"]),"workers":workers,
              "policy":"exact grounding text reuse only; retain every original observation and corpus-mean weight"}
    with memory.transaction():memory.set_metadata("pmm_embedding_cache",report)
    return report
