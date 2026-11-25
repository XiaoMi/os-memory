# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Ground observations in supplied fields, never inventing unobserved media inference.

The deterministic extractor is intentionally versioned and replaceable. Captions
and tags are upstream claims, not verified world facts or independent sources.
All offsets refer to the original field string (before normalization).
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from hypermem.standardization import ANCHOR_PATTERNS, STRONG_ANCHORS, normalize

VERSION = "source-grounded-rules-v1"
APP_ALIASES = {
    "com.tencent.mm": "微信", "com.eg.android.alipaygphone": "支付宝",
    "com.sankuai.meituan": "美团", "com.sankuai.meituan.takeoutnew": "美团外卖",
    "com.ss.android.ugc.aweme": "抖音", "com.xingin.xhs": "小红书",
    "com.taobao.taobao": "淘宝", "com.jingdong.app.mall": "京东",
    "com.tencent.mobileqq": "QQ", "com.baidu.baidumap": "百度地图",
}
GENERIC = frozenset({"用户", "朋友", "同事", "聊天", "工作", "购物", "学习", "视频", "手机", "微信", "抖音", "小红书", "unknown"})
TEXT_FIELDS = ("title", "content", "text", "text_after", "added_text", "ocr", "ocr_text", "asr", "transcript", "vlm_caption", "description")
MEDIA_FIELDS = ("filePath", "image_path", "audio_path", "video_path", "media_uri")
ENTITY_FIELDS = {"contactName": "participant", "merchantName": "merchant", "shopName": "merchant",
                 "productName": "object", "locationName": "place", "organizer": "organizer"}
ANCHOR_FIELDS = {"orderId": "order_id", "transactionId": "transaction_id", "trackingId": "tracking_id",
                 "reservationId": "reservation_id", "phoneNum": "phone", "session_id": "session_id",
                 **{k: k for k in ANCHOR_PATTERNS}}
ACTIVITIES = (
    ("search", r"搜索|检索|查找|search(?:ing)?"),
    ("communicate", r"聊天|讨论|咨询|回复|询问|发送消息|chat(?:ting)?|discuss"),
    ("browse", r"观看|刷视频|看到|浏览|查看|阅读|watch(?:ing)?|brows(?:e|ing)|view(?:ing)?"),
    ("refund", r"退款|退货|refund"),
    ("payment", r"支付|付款|转账|payment|paid"),
    ("reservation", r"预订|预约|订票|订酒店|reservation|booking"),
    ("purchase", r"下单|购买|买了|购物|purchase"),
    ("study", r"学习|做题|复习|作业|课程|study"),
    ("travel", r"导航|乘坐|到达|出发|出行|打车"),
)
NOISE = re.compile(r"^(?:在)?(?:端侧数据采集|数据采集器|采集服务).{0,24}(?:运行|状态|权限|服务)|^(?:查看了?|显示了?)?(?:手机的?)?(?:电池电量和信号强度|状态栏|系统桌面|空白界面)[。.!！\s]*$")
EXPOSURE = re.compile(r"(?:看到|观看|刷到|视频中|影片中|小说中|广告|推广|新闻|朋友圈.{0,8}分享)")
INTENT = re.compile(r"(?:准备|计划|想要|打算|考虑|希望|待支付|未支付|待付款|尚未)")
COMPLETED = re.compile(r"(?:支付成功|付款成功|交易成功|退款成功|退款到账|已付款|已支付|已退款|订单已完成|预订成功|预约成功|payment successful|paid successfully)", re.I)
TOPIC_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,16}")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def unpack(observation):
    try:
        event = json.loads(observation.content)
    except (ValueError, TypeError):
        event = None
    if not isinstance(event, dict):
        return {}, {"content": observation.content}, "content"
    if observation.provenance.metadata.get("input_format") == "pmm":
        event = event["payload"]["event"]
        data = event.get("data", {})
        return event, data if isinstance(data, dict) else {}, "content.payload.event.data"
    data = event.get("data", event)
    return event, data if isinstance(data, dict) else {}, "content.data" if "data" in event else "content"


def ground(observation, *, aliases=None, max_chars=6000):
    event, data, prefix = unpack(observation)
    bundle = observation.provenance.metadata.get("input_format") == "atomic_bundle"
    if bundle:
        from hypermem.adapters.atomic_bundle import grounding_inputs
        bundle_fields = grounding_inputs(event)
        data = {}  # Never treat inference or third-party sender as user-authored text.
    kind = str(event.get("perception_type", observation.provenance.source_type or "text"))
    original_app = str(event.get("from_app") or event.get("from_pkg") or observation.source)
    # _effective_app is an upstream correction, not a new independent inference.
    effective = str(data.get("_effective_app") or original_app)
    app = APP_ALIASES.get(effective, effective)
    fields, media = [], []
    field_inputs = []
    for key in (() if bundle else TEXT_FIELDS):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        method = "upstream_visual_description" if key == "vlm_caption" else (
            "upstream_ocr" if key in {"ocr", "ocr_text"} else "upstream_asr" if key in {"asr", "transcript"} else "supplied_text")
        field_inputs.append((key, value, f"{prefix}.{key}" if prefix != "content" or event else "content", method,
                             0.68 if key == "vlm_caption" else 0.85 if key in {"ocr", "ocr_text", "asr", "transcript"} else 0.9))
    for key, value, path, method, quality in (bundle_fields if bundle else field_inputs):
        fields.append({"field": path, "key": key, "text": value, "method": method, "quality": quality,
                       "origin_observation_id": observation.observation_id})
    for key in MEDIA_FIELDS:
        if isinstance(data.get(key), str) and data[key].strip():
            media.append({"field": f"{prefix}.{key}", "uri": data[key], "opened": False})
    source_key = event.get("capture_id") or data.get("capture_id") or data.get("frame_id")
    source_group = ("capture:" + digest([event.get("from_device"), source_key]) if source_key else
                    "media:" + digest([item["uri"] for item in media]) if media else observation.observation_id)
    if bundle and kind.startswith("notification_"):
        # Lifecycle records are not independent corroboration. Preserve the open
        # action in O, while grouping it with the corresponding received push.
        lifecycle = event["facts"].get("lifecycle", [])
        received = sorted(x["event_id"] for x in lifecycle if x.get("status") == "notification_received" and x.get("event_id"))
        if received:
            source_group = "notification:" + digest([observation.provenance.metadata["source_id"], received[0]])
    text = normalize("\n".join(dict.fromkeys(field["text"] for field in fields)))[:max_chars]
    structured, anchors, entities = [], {}, []

    def add_anchor(key, value, field, quality, start=None, end=None):
        if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
            return
        value = normalize(str(value)).rstrip(".,;，。；")
        anchors.setdefault(key, set()).add(value)
        # Namespace matters. Unknown scopes are not automatically equal across apps.
        issuer = str(data.get("orderPlatform") or data.get("issuer") or app)
        scope = str(data.get("carrier") or "unknown_carrier") if key == "tracking_id" else issuer
        structured.append({"type": "anchor", "kind": key, "value": value,
                           "role": "identifier" if key in STRONG_ANCHORS else "reference",
                           "scope": scope, "scope_explicit": bool(data.get("orderPlatform") or data.get("issuer") or key == "tracking_id" and data.get("carrier")),
                           "field": field, "start": start, "end": end, "quality": quality})

    for key, values in observation.anchors.items():
        for value in values:
            add_anchor(key, value, f"anchors.{key}", 0.95)
    for key, canonical in ANCHOR_FIELDS.items():
        value = data.get(key)
        for item in value if isinstance(value, list) else [value]:
            add_anchor(canonical, item, f"{prefix}.{key}", 0.95)
    for field in fields:
        for key, pattern in ANCHOR_PATTERNS.items():
            for match in re.finditer(pattern, field["text"], re.I):
                add_anchor(key, match.group(1), field["field"], field["quality"], *match.span(1))

    canonical_aliases = {normalize(k).casefold(): normalize(v).casefold() for k, v in (aliases or {}).items()}

    def add_entity(value, role, field, quality, start=None, end=None):
        if not isinstance(value, str) or not value.strip():
            return
        normalized = normalize(value).casefold()
        normalized = canonical_aliases.get(normalized, normalized)
        if normalized in GENERIC or len(normalized) < 2:
            return
        entities.append({"value": normalized, "role": role, "field": field,
                         "quality": quality, "start": start, "end": end})

    for key, role in ENTITY_FIELDS.items():
        add_entity(data.get(key), role, f"{prefix}.{key}", 0.95)
    if bundle and kind == "screen_session_summary":
        for i, entity in enumerate(event["facts"].get("entities", [])):
            if isinstance(entity, dict):
                add_entity(entity.get("text"), "upstream_" + str(entity.get("role") or "mention"),
                           f"content.facts.entities[{i}].text", 0.68)
    for key, role in (("entities", "explicit_entity"),):
        values = event.get(key, [])
        if isinstance(values, list):
            for value in values:
                add_entity(value, role, f"event.{key}", 0.85)
    # Role-bearing mentions are more useful than generic tags. Conservative
    # delimiters avoid pretending that arbitrary entire Chinese sentences are entities.
    patterns = (("participant", r"(?:与|和|向)([^，。！？,!?\s]{2,18}?)(?:聊天|讨论|咨询|询问|交流)"),
                ("merchant", r"(?:商家|店铺|门店|商户)[：:]\s*([^，。！？,!?\n]{2,30})"),
                ("object", r"[《]([^》]{2,40})[》]"))
    for field in fields:
        for role, pattern in patterns:
            for match in re.finditer(pattern, field["text"]):
                add_entity(match.group(1), role, field["field"], min(0.8, field["quality"]), *match.span(1))
        for alias, canonical in canonical_aliases.items():
            if alias in normalize(field["text"]).casefold():
                add_entity(canonical, "configured_alias", field["field"], min(0.85, field["quality"]))

    topics = []
    tags = data.get("vlm_tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and normalize(tag) in text and normalize(tag).casefold() not in GENERIC:
                topics.append({"value": normalize(tag).casefold(), "field": f"{prefix}.vlm_tags",
                               "quality": 0.6, "role": "upstream_topic_not_identity"})
    actions = [name for name, pattern in ACTIVITIES if re.search(pattern, text, re.I)]
    exposure, intent = bool(EXPOSURE.search(text)), bool(INTENT.search(text))
    completed_fields = [f for f in fields if COMPLETED.search(f["text"]) and not INTENT.search(f["text"]) and not EXPOSURE.search(f["text"])]
    completion = bool(completed_fields) and not exposure
    status = "exposure" if exposure else "intention" if intent else "completed_claim" if completion else "observed_activity" if actions else "unknown"
    subject = "content_or_third_party" if exposure or kind in {"GalleryPerception", "FileexplorerPerception"} else (
        "device_user_activity" if actions else "unknown")
    measurements = []
    if kind == "GPSPerception":
        lat, lon = data.get("latitude"), data.get("longitude")
        if (isinstance(lat, (int,float)) and not isinstance(lat, bool) and math.isfinite(lat) and -90 <= lat <= 90 and
                isinstance(lon, (int,float)) and not isinstance(lon, bool) and math.isfinite(lon) and -180 <= lon <= 180):
            measurements.append({"kind": "device_location", "latitude": lat, "longitude": lon,
                                 "fields": [f"{prefix}.latitude", f"{prefix}.longitude"],
                                 "accuracy": data.get("accuracy"), "quality": 0.85,
                                 "claim_scope": "device_not_verified_person_location"})
    if kind == "WifiPerception" and isinstance(data.get("connectedWifi"), str):
        measurements.append({"kind": "device_wifi", "value": data["connectedWifi"],
                             "field": f"{prefix}.connectedWifi", "quality": 0.85})
    noise_reason = "no_semantic_or_structured_evidence" if not text and not structured and not media and not measurements else (
        "collector_or_status_surface" if NOISE.search(text) else None)
    quality = max([f["quality"] for f in fields] + [s["quality"] for s in structured] + [m["quality"] for m in measurements] + [0.0])
    if len(text) < 4 and not structured and not measurements:
        quality = min(quality, 0.35)
    strong = any(a["kind"] in STRONG_ANCHORS and a["quality"] >= 0.8 and len(anchors[a["kind"]]) == 1 for a in structured)
    # A caption saying "bought" is not an independently verified transaction.
    standalone = bool(strong and completion and any(f["quality"] >= 0.8 for f in completed_fields) and subject != "content_or_third_party")
    if bundle:
        if kind in {"sms_message", "notification_received", "notification_opened", "gallery_asset_group"}:
            subject, status, standalone = "content_or_third_party", "exposure", False
            actions = ({"sms_message": ["message_record"], "notification_received": ["receive_notification"],
                        "notification_opened": ["open_notification"], "gallery_asset_group": []}[kind])
        else:
            subject, standalone = "upstream_user_activity_claim", False
    facts = []
    # Only explicitly attributed first-person statements are direct fact candidates.
    # A draft, clipboard, incoming message or caption cannot establish user ownership.
    authored = data.get("speaker_role") == "user" or data.get("direction") == "outgoing" and data.get("sent") is True
    if authored:
        for field in fields:
            if field["key"] not in {"content", "text", "transcript"}:
                continue
            for match in re.finditer(r"我(?:喜欢|不喜欢|偏好|不吃|住在|在.+?工作)[^。！？!?\n]{1,100}", field["text"]):
                facts.append({"content": match.group(0), "kind": "explicit_self_report", "field": field["field"],
                              "start": match.start(), "end": match.end(), "quality": field["quality"],
                              "supporting_observation_ids": [observation.observation_id]})
    return {"version": VERSION, "observation_id": observation.observation_id,
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_group": source_group, "fingerprint": digest([app, text, sorted((k, sorted(v)) for k,v in anchors.items())]),
            "app": app, "original_app": original_app, "app_corrected_upstream": bool(data.get("_effective_app")),
            "perception_type": kind, "fields": fields, "media": media, "measurements": measurements,
            "modality_processing": "supplied_representations_only", "missing_media_inference": bool(media),
            "anchors": {k: sorted(v) for k,v in anchors.items()}, "anchor_evidence": structured,
            "entities": entities, "topics": topics, "actions": actions, "epistemic_status": status,
            "subject": subject, "quality": quality, "noise_reason": noise_reason,
            "upstream_memory_value": data.get("memory_value"), "standalone_event": standalone,
            "fact_candidates": facts, "event_time": observation.start_time.isoformat(),
            **({"upstream_inference": event.get("inference"), "aggregation_level": event.get("aggregation_level"),
                "media_available_upstream": event.get("evidence", {}).get("media_available"),
                "media_reprocessed": False} if bundle else {}),
            "provenance": observation.provenance.to_dict()}


def identity_anchors(grounded):
    """Only reasonably grounded, single-valued IDs can constrain event identity.

    A screen listing several orders is not evidence that those orders are one.
    Caption-only IDs can aid recall but do not create hard constraints.
    """
    result = {}
    for item in grounded["anchor_evidence"]:
        if item["kind"] not in STRONG_ANCHORS or item["quality"] < 0.8:
            continue
        key = (item["kind"], item["scope"])
        result.setdefault(key, {})[item["value"]] = item
    return {key: next(iter(values.values())) for key, values in result.items() if len(values) == 1}
