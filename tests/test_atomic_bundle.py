# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import json
from pathlib import Path
import tempfile
import unittest

from hypermem.adapters.atomic_bundle import AtomicBundleAdapter
from hypermem.grounding import ground, identity_anchors
from hypermem.import_observations import import_jsonl
from hypermem.memory import EventMemory
from hypermem.schema import Provenance
from hypermem.source_verify import verify_source


def record(kind="sms_message", **facts):
    return {"observation_id": "fixture-1", "timestamp_ms": 0,
            "timestamp": "1970-01-01T08:00:00+08:00", "observation_type": kind,
            "source_type": "sms", "app": {"name": "测试应用"}, "facts": facts,
            "inference": {"intent": "购买", "summary": "用户已购买虚构商品"},
            "evidence": {"media_available": False}, "provenance": {},
            "aggregation_level": "event", "representation": "text_only"}


def observation(raw):
    return AtomicBundleAdapter().parse(raw, observation_id="O-test",
        provenance=Provenance("R-test", source_type=raw["observation_type"],
            metadata={"input_format": "atomic_bundle", "source_id": "fixture"}))


class AtomicBundleTests(unittest.TestCase):
    def test_incoming_message_not_user_purchase_or_self_report(self):
        g = ground(observation(record(message_body="我喜欢咖啡，订单号: ABC12345 支付成功", direction="received")))
        self.assertEqual(g["subject"], "content_or_third_party")
        self.assertEqual(g["actions"], ["message_record"])
        self.assertFalse(g["standalone_event"])
        self.assertEqual(g["fact_candidates"], [])
        self.assertNotIn("虚构商品", g["text"])
        self.assertEqual(g["fields"][0]["field"], "content.facts.message_body")

    def test_screen_model_evidence_not_hard_identity(self):
        raw = record("screen_session_summary", topic="购物", observed_actions=["订单号: ABC12345 支付成功"],
                     entities=[{"text": "测试商品", "role": "页面展示"}])
        g = ground(observation(raw))
        self.assertEqual(g["quality"], 0.68)
        self.assertFalse(identity_anchors(g))
        self.assertFalse(g["standalone_event"])
        self.assertEqual(g["fields"][1]["field"], "content.facts.observed_actions[0]")
        self.assertEqual(g["entities"][0]["role"], "upstream_页面展示")

    def test_gallery_is_content_not_user_action(self):
        g = ground(observation(record("gallery_asset_group", caption="某人物购买了商品", visible_text=["订单号: ABC12345"])))
        self.assertEqual(g["actions"], [])
        self.assertEqual(g["subject"], "content_or_third_party")
        self.assertFalse(identity_anchors(g))

    def test_notification_lifecycle_not_independent_sources(self):
        lifecycle = [{"event_id": "received-id", "status": "notification_received"}]
        a = ground(observation(record("notification_received", title="标题", body="消息", lifecycle=lifecycle)))
        b = ground(observation(record("notification_opened", title="标题", body="消息", lifecycle=lifecycle)))
        self.assertEqual(a["source_group"], b["source_group"])
        self.assertEqual(b["actions"], ["open_notification"])

    def test_timestamp_disagreement_rejected(self):
        raw = record(message_body="测试")
        raw["timestamp_ms"] = 1
        with self.assertRaises(ValueError):
            observation(raw)

    def test_lossless_import_source_verification_and_duplicate_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "input.jsonl"
            raw = record(message_body="测试消息")
            line = json.dumps(raw, ensure_ascii=False) + "\n"
            path.write_text(line, encoding="utf-8")
            with EventMemory(run / "memory.sqlite") as memory:
                report = import_jsonl(path, source_id="fixture", input_format="atomic_bundle", memory=memory, user_id="fixture-owner")
                o = next(memory.iter_observations())
                self.assertEqual(json.loads(o.content), raw)
                self.assertEqual(o.provenance.metadata["upstream_observation_id"], raw["observation_id"])
                self.assertEqual(o.provenance.metadata["user_id"], "fixture-owner")
            (run / "import_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(verify_source(run)["status"], "passed")
            path.write_text(line * 2, encoding="utf-8")
            with EventMemory() as memory:
                with self.assertRaises(ValueError):
                    import_jsonl(path, source_id="fixture", input_format="atomic_bundle", memory=memory)
                self.assertEqual(memory.counts()["observations"], 0)


if __name__ == "__main__":
    unittest.main()
