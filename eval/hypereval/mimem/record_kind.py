# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Three-way typing of device records: occurrence / observation / authored.

The on-device module labels every record with an ``event_type`` and calls
them all "events". But empirically (see device_data_event(24).db, 1526
records) the ten types are NOT the same kind of thing. Some records ARE a
real-world occurrence; most are a *trace* dropped by an occurrence; a few
are content the user deliberately authored.

Mi-Mem's "observation -> event" inference needs this distinction up front:
occurrence rows can be promoted to Event nodes as-is (record == event),
while observation rows must be *aggregated* into latent Event nodes and
kept as evidence beneath them. Treating a photo caption the same as a phone
call is the category error we're fixing.

    RecordKind.OCCURRENCE   record == one real-world happening
    RecordKind.OBSERVATION  record == a trace/snapshot; event is latent
    RecordKind.AUTHORED     record == user-authored content (reference)

Distribution on device_data_event(24).db:
    occurrence   367  (24%)   通话 122 · 短信 11 · 通知 35 · 日程 199
    observation 1065  (69%)   相册 991 · 地理位置 70 · 截屏 4
    authored      94  ( 6%)   联系人 56 · 笔记 31 · 录音 7
"""
from __future__ import annotations

from enum import Enum


class RecordKind(str, Enum):
    #: The record itself is one discrete real-world happening. Time point +
    #: actor + action are all present. Record granularity == event
    #: granularity, so it can seed an Event node directly.
    #:   通话  一次通话 (起止/时长/对象齐全)
    #:   短信  一次收发 (发件方/内容/时间点)
    #:   通知  一次推送 = 一个交易/物流节点
    #:   日程  一个计划中的 occurrence (未来时态, 但仍离散)
    OCCURRENCE = "occurrence"

    #: The record is a trace or state-slice left BY some real-world event;
    #: alone it does not say "what happened". Must be aggregated bottom-up
    #: into a latent Event and retained as its evidence.
    #:   相册  一张图的视觉 caption (零现实语义: "一片开阔的海面")
    #:   截屏  一屏 OCR 快照 (导航页/设置页)
    #:   地理位置  一段停留区间 (端上把 GPS 点聚成 "停留 65h"), 是状态非瞬时发生
    OBSERVATION = "observation"

    #: User-authored content: a note, a recording transcript, a saved
    #: contact. Not a happening and not a passive trace — a reference the
    #: user chose to keep. Feeds Semantic (facts) but should not masquerade
    #: as an Event on the timeline.
    #:   笔记  OKR / 读书清单 / 家庭资料
    #:   录音  转写文本 (发热咨询)
    #:   联系人  新增/更新一个联系人条目
    AUTHORED = "authored"


# --- the mapping the on-device event_type -> RecordKind ----------------

_KIND_BY_EVENT_TYPE: dict[str, RecordKind] = {
    # occurrence — record == event
    "通话事件": RecordKind.OCCURRENCE,
    "短信事件": RecordKind.OCCURRENCE,
    "通知消息事件": RecordKind.OCCURRENCE,
    "日程待办事件": RecordKind.OCCURRENCE,
    # observation — trace/snapshot, event is latent
    "相册事件": RecordKind.OBSERVATION,
    "截屏事件": RecordKind.OBSERVATION,
    "地理位置事件": RecordKind.OBSERVATION,
    # authored — user-authored reference content
    "笔记事件": RecordKind.AUTHORED,
    "录音事件": RecordKind.AUTHORED,
    "联系人事件": RecordKind.AUTHORED,
}

#: Unknown event_types default to OBSERVATION — the conservative choice:
#: an unrecognized record is treated as evidence to be aggregated, never
#: silently promoted to a first-class Event.
_DEFAULT_KIND = RecordKind.OBSERVATION


def kind_of(event_type: str) -> RecordKind:
    """Map an on-device ``event_type`` label to its :class:`RecordKind`."""
    return _KIND_BY_EVENT_TYPE.get(event_type, _DEFAULT_KIND)


def is_occurrence(event_type: str) -> bool:
    return kind_of(event_type) is RecordKind.OCCURRENCE


def is_observation(event_type: str) -> bool:
    return kind_of(event_type) is RecordKind.OBSERVATION


def is_authored(event_type: str) -> bool:
    return kind_of(event_type) is RecordKind.AUTHORED
