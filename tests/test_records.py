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
import unittest

from hypermem import HyperMemory
from hypermem.api import _resolve_config, _translate
from hypermem.grounding import ground
from hypermem.online.engine import OnlineHyperMem
from hypermem.standardization import normalize
from test_online_semantic import Encoder, settings

T0 = 1777770000000


def record(kind, data, timestamp=T0):
    return {"type": kind, "timestamp": timestamp, "data": data}


def events(rec):
    """[(perception_type, ts_ms, collector event), ...] for one record."""
    return _translate(rec)[1]


def unit(rec, index=0):
    return events(rec)[index][2]["data"]


class RecordFormatTests(unittest.TestCase):
    def test_every_type_becomes_a_collector_record(self):
        examples = {
            "screen": ({"action_description": "查询车票", "title": "车票查询", "ocr": "G15", "image_path": "/s.jpg"},
                       ["ScreenPerception"]),
            "gallery": ({"image_description": "车票照片", "ocr": "G15"}, ["GalleryPerception"]),
            "sms": ({"content": "订单已支付", "sender": "12306", "sender_name": "铁路12306"}, ["SmsPerception"]),
            "push": ({"action": "receive", "content": "外卖已送达", "title": "订单提醒"}, ["PushPerception"]),
            "calendar": ({"title": "周三去杭州出差"}, ["CalendarPerception"]),
            "clipboard": ({"content": "补光棒", "action": "cut"}, ["ClipboardPerception"]),
            "position": ({"latitude": 31.23, "longitude": 121.47, "ssid": "Home_5G"},
                         ["GPSPerception", "WifiPerception"]),
            "chat": ({"query": "我不吃香菜", "answer": "已避开香菜"}, ["AppPerception", "AppPerception"]),
            "recording": ({"transcript": [{"text": "下周三去杭州", "offset_ms": 0, "speaker": None}]},
                          ["AudioPerception"]),
            "note": ({"action": "edit", "content": "带上合同"}, ["NotePerception"]),
        }
        for kind, (data, types) in examples.items():
            with self.subTest(kind=kind):
                self.assertEqual([pt for pt, _, _ in events(record(kind, data))], types)

    def test_fields_map_onto_collector_keys(self):
        self.assertEqual(unit(record("screen", {"action_description": "查询车票", "title": "车票查询"})),
                         {"vlm_caption": "查询车票", "title": "车票查询"})
        self.assertEqual(unit(record("screen", {"action_description": "在搜索框输入", "typed_text": "北京南到上海"})),
                         {"vlm_caption": "在搜索框输入", "text_after": "北京南到上海"})
        self.assertEqual(unit(record("push", {"action": "receive", "title": "好物推荐"})), {"title": "好物推荐"})
        self.assertEqual(unit(record("note", {"action": "create", "title": "购物清单"}))["title"], "购物清单")
        self.assertEqual(unit(record("sms", {"content": "x", "sender": "12306", "sender_name": "铁路12306"})),
                         {"content": "x", "phoneNum": "12306", "contactName": "铁路12306"})
        self.assertEqual(unit(record("position", {"ssid": "Home_5G"})),
                         {"connectedWifi": "Home_5G", "content": "连接 WiFi：Home_5G"})
        self.assertEqual(unit(record("position", {"latitude": 31.23, "longitude": 121.47, "location": "人民广场"}))
                         ["content"], "位置：人民广场")

    def test_actions_become_collector_labels_and_text(self):
        def event(kind, data):
            return events(record(kind, data))[0][2]
        opened = event("push", {"action": "open", "content": "外卖已送达"})
        self.assertEqual((opened["active_name"], opened["data"]["description"]), ("PUSH打开", "用户打开了这条通知"))
        received = event("push", {"action": "receive", "content": "外卖已送达"})
        self.assertEqual((received["active_name"], "description" in received["data"]), ("PUSH接收", False))
        self.assertEqual(event("clipboard", {"content": "x", "action": "copy"})["active_name"], "复制")
        self.assertEqual(event("clipboard", {"content": "x"})["active_name"], "剪贴板新增")
        note = event("note", {"action": "create", "content": "带上合同"})
        self.assertEqual((note["active_name"], note["data"]["speaker_role"]), ("笔记新增", "user"))
        self.assertNotIn("speaker_role", event("note", {"action": "delete", "content": "旧笔记"})["data"])

    def test_common_fields_reach_every_observation(self):
        out = events(record("chat", {"query": "订单到了吗", "answer": "已签收", "from_app": "MyAssistant",
                                     "anchors": {"order_id": "E1"}, "entities": {"merchant": "美团"},
                                     "latitude": 39.9, "longitude": 116.4}))
        for _, _, event in out:
            self.assertEqual(event["from_app"], "MyAssistant")
            self.assertEqual({k: event["data"][k] for k in ("order_id", "merchantName", "latitude", "longitude")},
                             {"order_id": "E1", "merchantName": "美团", "latitude": 39.9, "longitude": 116.4})
        sms = unit(record("sms", {"content": "x", "sender_name": "银行", "entities": {"contact": "张三"}}))
        self.assertEqual(sms["contactName"], "银行")

    def test_speakers_are_written_into_the_text(self):
        (_, t1, q), (_, t2, a) = events(record("chat", {"query": "我住在上海", "answer": "好的"}))
        self.assertEqual((q["data"]["content"], a["data"]["content"], t2 - t1), ("用户：我住在上海", "助手：好的", 1))
        rec = events(record("recording", {"transcript": [
            {"text": "下周三去杭州", "offset_ms": 0, "speaker": "user"},
            {"text": "好的", "offset_ms": 5000, "speaker": "张三"},
            {"text": "收到", "offset_ms": 7000, "speaker": None}]}))
        self.assertEqual([ts - T0 for _, ts, _ in rec], [0, 5000, 7000])
        self.assertEqual([e["data"]["transcript"] for _, _, e in rec], ["用户：下周三去杭州", "张三：好的", "收到"])

    def test_content_times_and_places_are_searchable_text(self):
        gallery = unit(record("gallery", {"image_description": "西湖合影", "captured_at": "2025-03-01T15:30:12+08:00",
                                          "location": "杭州西湖断桥"}))
        self.assertEqual(gallery["description"], "拍摄时间：2025-03-01T15:30:12+08:00；拍摄地点：杭州西湖断桥")
        self.assertEqual(gallery["captured_at"],
                         int(datetime.fromisoformat("2025-03-01T15:30:12+08:00").timestamp() * 1000))
        calendar = unit(record("calendar", {"title": "出差", "remind_time": 1778025600000}))
        self.assertEqual(calendar["remind_time"], 1778025600000)
        self.assertTrue(calendar["description"].startswith("提醒时间：2026-05-0"))
        meeting = unit(record("calendar", {"title": "去上海出差", "content": "带上合同",
                                           "start_time": "2026-05-08T09:00:00+08:00",
                                           "end_time": "2026-05-08T18:00:00+08:00",
                                           "remind_time": "2026-05-08T08:00:00+08:00", "calendar_type": "工作"}))
        self.assertEqual(meeting["description"],
                         "开始时间：2026-05-08T09:00:00+08:00；结束时间：2026-05-08T18:00:00+08:00；"
                         "提醒时间：2026-05-08T08:00:00+08:00；日历类型：工作")
        self.assertEqual(meeting["end_time"] - meeting["start_time"], 9 * 3600 * 1000)
        with self.assertRaisesRegex(ValueError, "end_time must not be earlier"):
            _translate(record("calendar", {"title": "x", "start_time": T0 + 2, "end_time": T0 + 1}))

    def test_timestamp_is_stored_as_milliseconds(self):
        stored, out = _translate(record("sms", {"content": "x"}, timestamp="2026-05-03T09:00:00+08:00"))
        expected = int(datetime.fromisoformat("2026-05-03T09:00:00+08:00").timestamp() * 1000)
        self.assertEqual((stored["timestamp"], out[0][1]), (expected, expected))
        for bad in ("yesterday", 1.5, True, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "timestamp must be"):
                _translate(record("sms", {"content": "x"}, timestamp=bad))

    def test_integer_seconds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamp must be integer Unix milliseconds; 1777770000 looks like Unix seconds"):
            _translate(record("sms", {"content": "x"}, timestamp=T0 // 1000))
        with self.assertRaisesRegex(ValueError, "data.start_time .* looks like Unix seconds"):
            _translate(record("calendar", {"title": "x", "start_time": T0 // 1000}))
        stored, _ = _translate(record("sms", {"content": "x"}, timestamp=100_000_000_000))
        self.assertEqual(stored["timestamp"], 100_000_000_000)

    def test_invalid_records(self):
        cases = [
            ("not a dict", "each record must be an object"),
            ({**record("sms", {"content": "x"}), "app": "短信"}, "unknown field"),
            (record("image", {}), "unknown type"),
            ({"type": "sms", "data": {"content": "x"}}, "requires 'timestamp'"),
            ({"type": "sms", "timestamp": T0}, "requires a 'data' object"),
            (record("screen", {"caption": "x"}), "requires data.action_description"),
            (record("push", {"content": "x"}), "requires data.action"),
            (record("push", {"action": "open"}), "requires data.title or data.content"),
            (record("note", {"action": "edit", "content": ""}), "requires data.title or data.content"),
            (record("screen", {"action_description": "x", "typed_text": 3}), "data.typed_text must be"),
            (record("push", {"action": "opened", "content": "x"}), "data.action must be one of receive, open"),
            (record("note", {"action": "update", "content": "x"}), "must be one of create, edit, delete"),
            (record("clipboard", {"content": "x", "action": "paste"}), "must be one of copy, cut"),
            (record("calendar", {"remind_time": T0}), "requires data.title or data.content"),
            (record("chat", {"answer": "好的"}), "requires data.query"),
            (record("position", {"location": "家"}), "requires data.latitude and data.longitude, or data.ssid"),
            (record("position", {"latitude": 31.2}), "given together"),
            (record("sms", {"content": "x", "latitude": 91, "longitude": 0}), "given together"),
            (record("sms", {"content": 3}), "data.content must be a non-empty string"),
            (record("sms", {"content": "x", "anchors": {"order": "E1"}}), "unknown data.anchors key"),
            (record("sms", {"content": "x", "entities": {"contact": ""}}), "data.entities.contact must be"),
            (record("gallery", {"image_description": "x", "captured_at": "去年"}), "data.captured_at must be"),
            (record("recording", {"transcript": [{"text": "x", "offset_ms": 0}]}), "exactly text, offset_ms and speaker"),
            (record("recording", {"transcript": [{"text": "x", "offset_ms": -1, "speaker": None}]}), "offset_ms"),
            (record("recording", {"transcript": []}), "non-empty list"),
        ]
        for rec, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _translate(rec)


class AddTests(unittest.TestCase):
    def setUp(self):
        self.mem = HyperMemory()
        self.addCleanup(self.mem.close)

    def test_add_takes_a_list_or_a_generator(self):
        recs = [record("sms", {"content": f"通知{i}", "record_id": f"n{i}"}) for i in range(3)]
        self.assertEqual(self.mem.add(recs[:2]), {"records": 2, "observations": 2})
        self.assertEqual(self.mem.add(r for r in recs[2:]), {"records": 1, "observations": 1})
        self.assertEqual(self.mem.add([record("chat", {"query": "q", "answer": "a"})]),
                         {"records": 1, "observations": 2})
        self.assertEqual(self.mem.count()["raw_records"], 4)

    def test_stored_record_is_what_was_submitted(self):
        rec = record("sms", {"content": "x", "from_app": "短信"}, timestamp="2026-05-03T09:00:00+08:00")
        self.mem.add([rec])
        stored = json.loads(self.mem.memory._connection.execute("SELECT raw_json FROM raw_records").fetchone()[0])
        self.assertEqual(stored["data"], rec["data"])
        self.assertIsInstance(stored["timestamp"], int)

    def test_a_bad_record_rolls_the_batch_back(self):
        with self.assertRaisesRegex(ValueError, r"records\[1\]: type 'sms' requires data.content"):
            self.mem.add([record("sms", {"content": "x"}), record("sms", {})])
        self.assertEqual(self.mem.count()["raw_records"], 0)

    def test_duplicate_record_ids_are_rejected(self):
        self.mem.add([record("sms", {"content": "x", "record_id": "a"})])
        with self.assertRaisesRegex(ValueError, r"records\[0\]: duplicate record_id 'a'"):
            self.mem.add([record("sms", {"content": "y", "record_id": "a"})])
        self.assertEqual(self.mem.count()["raw_records"], 1)

    def test_a_path_is_not_records(self):
        with self.assertRaisesRegex(TypeError, "to load a JSONL file"):
            self.mem.add("records.jsonl")


class ChatSessionTests(unittest.TestCase):
    """A chat session_id is an entity-level DRED key: a candidate scoped to the app."""

    def turns(self, *turns, session=True):
        mem = HyperMemory()
        self.addCleanup(mem.close)
        vectors, ids = {}, []
        for minute, query, vector in turns:
            data = {"from_app": "MyAssistant", "query": query, **({"session_id": "s1"} if session else {})}
            mem.add([record("chat", data, timestamp=T0 + minute * 60_000)])
            vectors[normalize("用户：" + query)] = vector
        engine = OnlineHyperMem(mem.memory, settings(), encoder=Encoder(vectors))
        engine.replay()
        return [engine.states[o.observation_id]["event_id"] for o in mem.memory.iter_observations()]

    def test_session_id_is_an_app_scoped_entity_key(self):
        mem = HyperMemory()
        self.addCleanup(mem.close)
        mem.add([record("chat", {"from_app": "MyAssistant", "query": "你好", "session_id": "s1"})])
        grounded = ground(next(mem.memory.iter_observations()))
        engine = OnlineHyperMem.__new__(OnlineHyperMem)
        engine.g = {grounded["observation_id"]: grounded}
        event_keys, entity_keys = engine._key_tiers(grounded["observation_id"])
        self.assertEqual(event_keys, [])
        self.assertEqual([json.loads(k) for k in entity_keys], [["session_id", "MyAssistant", "s1"]])

    def test_a_later_related_turn_joins_its_session(self):
        # Two quick turns form an event; 30 minutes later it is no longer active,
        # so only the session key can lead the related third turn back to it.
        turns = ((0, "帮我找家川菜馆", [1, 0, 0]), (1, "要能订位的", [1, 0, 0]),
                 (30, "再推荐一家近一点的", [0.6, 0.8, 0]))
        first, second, third = self.turns(*turns)
        self.assertEqual((first, second), (first, third))
        first, second, third = self.turns(*turns, session=False)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_an_unrelated_turn_in_the_same_session_stays_apart(self):
        first, second, third = self.turns((0, "帮我找家川菜馆", [1, 0, 0]), (1, "要能订位的", [1, 0, 0]),
                                          (30, "明天天气怎么样", [0, 1, 0]))
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class ConfigTests(unittest.TestCase):
    CONFIG = {"llm": {"base_url": "http://llm/v1", "model": "m"},
              "embedding": {"url": "http://emb/v1/embeddings", "model": "e", "dimension": 3}}

    def test_a_missing_key_is_a_value_error_naming_it(self):
        self.assertEqual(_resolve_config(self.CONFIG)[0]["embedding"]["dimension"], 3)
        cases = [
            ({"embedding": {"url": "http://emb/v1/embeddings", "model": "e"}}, "config is missing embedding.dimension"),
            ({"embedding": {"dimension": 3}}, "config is missing embedding.url, embedding.model"),
        ]
        for override, message in cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                _resolve_config({**self.CONFIG, **override})
        _resolve_config({**self.CONFIG, "embedding": {"base_url": "http://emb", "model": "e", "dimension": 3}})

    def test_llm_settings_are_needed_only_for_build(self):
        embedding_only = {"embedding": self.CONFIG["embedding"]}
        cases = [
            (embedding_only, ["llm.base_url", "llm.model"]),
            ({**self.CONFIG, "llm": {"model": "m"}}, ["llm.base_url"]),
            ({**self.CONFIG, "llm": {"base_url": "", "model": "m"}}, ["llm.base_url"]),
            (self.CONFIG, []),
        ]
        for config, missing in cases:
            with self.subTest(config=config):
                self.assertEqual(_resolve_config(config)[2], missing)
        from hypermem import HyperMemory
        with HyperMemory(":memory:", config=embedding_only) as memory:
            memory.add([{"type": "screen", "timestamp": "2026-05-01 10:00",
                         "data": {"from_app": "日历", "action_description": "查看下周会议安排"}}], user="U01")
            self.assertEqual(memory.read()["total"], 0)
            with self.assertRaisesRegex(ValueError, r"config is missing llm\.base_url, llm\.model; build needs an LLM service"):
                memory.build()
            self.assertFalse(memory.is_started)


if __name__ == "__main__":
    unittest.main()
