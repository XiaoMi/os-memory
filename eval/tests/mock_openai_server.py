# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Offline OpenAI-compatible stand-in for pipeline checks (never for scores).

    python tests/mock_openai_server.py --port 8765 --dimension 64

Serves /v1/embeddings (hashed character bigrams) and /v1/chat/completions
with schema-valid replies for every stage HyperOS-Mem and pmm_bench call:
idle consolidation, ToolSAGE function calls, LongMemory, answering, judging.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math

DIMENSION = 64


def embed(text: str) -> list[float]:
    v = [0.0] * DIMENSION
    for i in range(max(1, len(text) - 1)):
        v[int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % DIMENSION] += 1.0
    v = [x + 1e-3 for x in v]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def first_obs(card: dict) -> str:
    return card["member_summaries"][0]["obs_id"]


def consolidation(payload: dict) -> dict:
    stage = payload["stage"]
    if stage == "adjudicate":
        return {"action": "isolate", "target_event": None, "confidence": 0.5, "reason": "离线替身",
                "evidence_ids": [payload["observation"]["obs_id"]]}
    if stage == "adjudicate_batch":
        return {"decisions": [{"item_id": i["item_id"], "action": "isolate", "target_event": None, "confidence": 0.5,
                               "reason": "离线替身", "evidence_ids": [i["observation"]["obs_id"]]} for i in payload["items"]]}
    if stage == "stitch":
        return {"same_activity": False, "confidence": 0.5, "reason": "离线替身",
                "evidence_ids": [first_obs(c) for c in payload["events"]]}
    def card(c):
        return {"title": c["member_summaries"][0]["text"][:40] or "事件", "summary": "离线替身摘要",
                "stable_keys": [], "start_time": c["start_time"], "end_time": c["end_time"], "evidence_ids": [first_obs(c)]}
    if stage == "wiki":
        return card(payload["events"][0])
    if stage == "wiki_batch":
        return {"cards": [{"event_id": i["event_id"], **card(i["event"])} for i in payload["items"]]}
    raise ValueError(f"unknown stage {stage}")


def long_memory(group: dict) -> dict:
    """One memory supported by the first two events that have cited facts."""
    events = [e for e in group["events"] if any(f["support"] for f in e["facts"])][:2]
    if len(events) < 2:
        return {"group_id": group["group_id"], "memories": []}
    evidence = [next(f["support"][0] for f in e["facts"] if f["support"]) for e in events]
    return {"group_id": group["group_id"], "memories": [{"text": "离线替身长期记忆：" + events[0]["title"][:20],
            "support_events": [e["event_id"] for e in events], "confidence": "high",
            "evidence_ids": evidence, "revises": None}]}


def tool_call(messages: list[dict]) -> dict:
    context = json.loads(messages[1]["content"])
    done = [m for m in messages if m.get("role") == "tool"]
    if not done:
        name, args = "SummarizeEvent", {"title": context["members"][0]["text"][:40] or "事件", "summary": "离线替身摘要",
                                        "event_type": "other", "entity_keys": [], "topic_anchors": ["日常"]}
    else:
        member = context["members"][0]
        name, args = "ExtractFacts", {"facts": [{"text": member["text"][:80] or "记录", "tags": [], "time": None,
                                                  "support": [member["obs_id"]]}]}
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"call-{len(done)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


FAIL: set[str] = set()


def kind(body: dict) -> str:
    if body.get("tools"):
        return "tools"
    system = body["messages"][0]["content"]
    return "consolidation" if "idle semantic consolidator" in system else \
        "long_memory" if "durable memory" in system else "qa"


def chat(body: dict) -> dict:
    messages = body["messages"]
    if body.get("tools"):
        return tool_call(messages)
    system, user = messages[0]["content"], messages[-1]["content"]
    if "idle semantic consolidator" in system:
        content = json.dumps(consolidation(json.loads(user)), ensure_ascii=False)
    elif "durable memory" in system:
        content = json.dumps({"groups": [long_memory(g) for g in json.loads(user)["groups"]]}, ensure_ascii=False)
    elif "裁判" in system:
        data = json.loads(user)
        hit = any(ref[:4] and ref[:4] in data["prediction"] for ref in data["references"])
        content = json.dumps({"label": "CORRECT" if hit else "WRONG", "reason": "离线替身"}, ensure_ascii=False)
    else:
        history = user.split("历史记录：\n", 1)[-1].split("\n\n问题：", 1)[0].strip()
        content = history.splitlines()[0][:400] if history else "证据不足，无法回答"
    return {"role": "assistant", "content": content}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            out = {"data": [{"index": i, "embedding": embed(t)} for i, t in enumerate(texts)], "model": body.get("model")}
        elif self.path.endswith("/chat/completions") and kind(body) in FAIL:
            self.send_error(400, "injected failure")
            return
        elif self.path.endswith("/chat/completions"):
            out = {"choices": [{"index": 0, "finish_reason": "stop", "message": chat(body)}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "model": body.get("model")}
        else:
            self.send_error(404)
            return
        data = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    global DIMENSION
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--fail", default="", help="comma list of request kinds to answer with HTTP 400: "
                                              "consolidation, tools, long_memory")
    a = p.parse_args()
    DIMENSION = a.dimension
    FAIL.update(k for k in a.fail.split(",") if k)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
