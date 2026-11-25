# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Generate a tiny, fully synthetic PMM-Bench package in the release layout.

    python examples/make_pmm_sample.py examples/pmm_sample

Every person, number and order below is invented. The package exercises the
loaders, both input files (raw and cleaned) and the evaluation entry point.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

TZ = timezone(timedelta(hours=8))

# (minute offset from day start, app, perception type, data)
DAYS = {
    "2026-05-03": [
        (540, "铁路12306", "ScreenPerception", {"title": "车票查询", "content": "查询 5月8日 北京南→上海虹桥 G15 二等座 余票充足"}),
        (546, "系统", "SmsPerception", {"content": "【铁路12306】订单E2026050301已支付，G15次5车08F，5月8日09:00北京南开。", "order_id": "E2026050301"}),
        (547, "微信", "PushPerception", {"title": "群消息", "content": "周末羽毛球改到周日下午三点"}),
        (552, "相册", "GalleryPerception", {"vlm_caption": "火车票订单截图，G15 北京南至上海虹桥，5车08F"}),
        (700, "美团", "ScreenPerception", {"title": "外卖下单", "content": "老街牛肉面 红烧牛肉面一份 实付 28.5 元"}),
        (701, "美团", "PushPerception", {"title": "订单通知", "content": "您的订单 MT88123 商家已接单，预计 12:05 送达", "order_id": "MT88123"}),
        (722, "美团", "PushPerception", {"title": "订单通知", "content": "订单 MT88123 已送达", "order_id": "MT88123"}),
    ],
    "2026-05-05": [
        (1200, "淘宝", "ScreenPerception", {"title": "搜索", "content": "搜索 电容笔 平板手写笔 按价格排序"}),
        (1210, "拼多多", "ScreenPerception", {"title": "商品详情", "content": "电容笔 百亿补贴价 229 元"}),
        (1222, "京东", "ScreenPerception", {"title": "商品详情", "content": "电容笔 京东自营 279 元 领券满减 30"}),
        (1230, "闲鱼", "ScreenPerception", {"title": "二手", "content": "电容笔 九成新 180 元 卖家在上海"}),
        (1245, "京东", "ScreenPerception", {"title": "支付成功", "content": "电容笔 订单 JD7781 实付 249 元", "order_id": "JD7781"}),
        (1300, "抖音", "AppPerception", {"title": "视频", "content": "浏览 美食探店 短视频"}),
    ],
    "2026-05-08": [
        (520, "铁路12306", "PushPerception", {"title": "出行提醒", "content": "G15 次列车 09:00 北京南开，检票口 5B"}),
        (545, "高德地图", "ScreenPerception", {"title": "导航", "content": "导航至 北京南站 预计 25 分钟"}),
        (840, "高德地图", "ScreenPerception", {"title": "到达", "content": "已到达 上海虹桥站"}),
        (850, "滴滴出行", "PushPerception", {"title": "行程", "content": "打车 上海虹桥站→静安寺 实付 76 元"}),
        (1100, "京东", "PushPerception", {"title": "物流", "content": "订单 JD7781 电容笔 已签收", "order_id": "JD7781"}),
    ],
    "2026-05-12": [
        (1205, "淘宝", "ScreenPerception", {"title": "搜索", "content": "搜索 蓝牙键盘 比较价格"}),
        (1214, "拼多多", "ScreenPerception", {"title": "商品详情", "content": "蓝牙键盘 百亿补贴 159 元"}),
        (1226, "京东", "ScreenPerception", {"title": "支付成功", "content": "蓝牙键盘 订单 JD8890 实付 169 元 使用优惠券", "order_id": "JD8890"}),
        (705, "美团", "ScreenPerception", {"title": "外卖下单", "content": "老街牛肉面 红烧牛肉面一份 实付 28.5 元"}),
    ],
}

QA = [
    ("Q1", "Single", "我5月8日坐的是哪趟车、几车几座？", "G15 次，5 车 08F，北京南开往上海虹桥。"),
    ("Q2", "Single", "电容笔最后在哪个平台买的，实付多少？", "在京东购买，订单 JD7781，实付 249 元。"),
    ("Q3", "Cross", "我买电容笔前在几个平台比过价，各是多少钱？", "四个平台：拼多多 229 元，京东 279 元（可领满减 30），闲鱼二手 180 元，另在淘宝搜索比较；最终京东实付 249 元。"),
    ("Q4", "Temp", "到上海虹桥站之后我接着做了什么？", "打车从上海虹桥站到静安寺，实付 76 元。"),
    ("Q5", "Track", "5月3日那单外卖从下单到送达经过是怎样的？", "在美团点了老街牛肉面红烧牛肉面，实付 28.5 元；商家接单预计 12:05 送达，之后订单 MT88123 已送达。"),
    ("Q6", "Habit", "我在网上买数码配件通常怎么做？", "先在淘宝搜索，再到拼多多、京东等平台比价，领券后在京东下单支付。"),
    ("Q7", "Single", "羽毛球改到什么时候了？", "改到周日下午三点。"),
    ("Q8", "Habit", "我点外卖常点什么？", "常点老街牛肉面的红烧牛肉面，实付 28.5 元。"),
]


# Substrings that identify each question's supporting records (evidence = their line numbers).
EVIDENCE = {
    "Q1": ["E2026050301已支付", "G15 次列车"], "Q2": ["JD7781 实付"],
    "Q3": ["搜索 电容笔", "百亿补贴价 229", "京东自营 279", "九成新 180", "JD7781 实付"],
    "Q4": ["已到达 上海虹桥站", "静安寺"], "Q5": ["红烧牛肉面一份@2026-05-03", "MT88123"],
    "Q6": ["搜索 电容笔", "JD7781 实付", "搜索 蓝牙键盘", "JD8890"], "Q7": ["羽毛球"], "Q8": ["红烧牛肉面一份"],
}


def evidence(qid):
    """Lines whose data contains a key; "key@YYYY-MM-DD" also requires that day."""
    out = []
    for line, r in records():
        text = json.dumps(r["payload"]["event"]["data"], ensure_ascii=False)
        day = datetime.fromtimestamp(r["ts"] / 1000, TZ).strftime("%Y-%m-%d")
        for key in EVIDENCE[qid]:
            needle, _, when = key.partition("@")
            if needle in text and (not when or when == day):
                out.append(f"line:{line}")
                break
    return out


def records():
    line = 0
    for day, items in DAYS.items():
        for minute, app, kind, data in sorted(items):
            line += 1
            ts = int((datetime.fromisoformat(day).replace(tzinfo=TZ) + timedelta(minutes=minute)).timestamp() * 1000)
            yield line, {"ts": ts, "type": "real", "payload": {"user_id": "synthetic-device-01", "src_line": line,
                         "event": {"perception_type": kind, "from_app": app, "from_pkg": "", "data": data}}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


USERS = {"U01": [q[0] for q in QA], "U02": ["Q1", "Q2", "Q4", "Q6", "Q7"]}


def write_user(target: Path, user: str, keep: list[str]) -> dict:
    base = target / "users" / user
    (base / "data").mkdir(parents=True, exist_ok=True)
    raw_lines, cleaned_lines, months = [], [], set()
    for line, record in records():
        raw = json.dumps(record, ensure_ascii=False)
        stamp = datetime.fromtimestamp(record["ts"] / 1000, TZ)
        months.add(stamp.strftime("%Y-%m"))
        raw_lines.append(json.dumps({"event_id": f"raw-line:{line}", "user_id": user, "session_id": "observations",
            "role": "user", "content": raw, "timestamp": stamp.isoformat(timespec="milliseconds"),
            "metadata": {"source_lines": [line], "input_line": line, "input_kind": "raw",
                         "app": record["payload"]["event"]["from_app"],
                         "perception_type": record["payload"]["event"]["perception_type"]}}, ensure_ascii=False))
        data = record["payload"]["event"]["data"]
        text = data.get("vlm_caption") or "\n".join(str(data[k]) for k in ("title", "content") if data.get(k))
        cleaned_lines.append(json.dumps({**record, "_cleaning": {"source_lines": [line], "observation_id": f"obs-{line:04d}",
            "source_text": text, "repeat_count": 1}}, ensure_ascii=False))
    (base / "observations.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    (base / "data" / "raw_source.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for _, r in records()) + "\n", encoding="utf-8")
    (base / "data" / "cleaned_observations.jsonl").write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    qa = [{"query_id": qid, "user_id": user, "session_id": None, "question": q, "answers": [a],
           "evidence_event_ids": evidence(qid), "category": c, "metadata": {"unanswerable": False}}
          for qid, c, q, a in QA if qid in keep]
    (base / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prefix = f"users/{user}/"
    files = [prefix + f for f in ("qa.json", "observations.jsonl", "data/raw_source.jsonl", "data/cleaned_observations.jsonl")]
    checksums = {f: {"sha256": sha(target / f), "bytes": (target / f).stat().st_size} for f in files}
    cleaned = prefix + "data/cleaned_observations.jsonl"
    entry = {"user_id": user, "qa": prefix + "qa.json", "observations": prefix + "observations.jsonl",
             "observation_count": len(raw_lines), "cleaned_observations": cleaned, "cleaned_event_count": len(cleaned_lines),
             "cleaned_sha256": checksums[cleaned]["sha256"], "selected_months": sorted(months), "question_count": len(qa)}
    return entry, checksums


def main(target: Path):
    entries, checksums = [], {}
    for user, keep in USERS.items():
        entry, sums = write_user(target, user, keep)
        entries.append(entry)
        checksums.update(sums)
    manifest = {"schema": "pmm-bench-package-v1", "synthetic": True, "users": entries}
    (target / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}: {len(entries)} users, {sum(e['question_count'] for e in entries)} questions")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/pmm_sample"))
