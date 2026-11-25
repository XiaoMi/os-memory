# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Lossless adapter for the upstream, text-only atomic observation bundle.

Session/asset-group summaries remain single upstream records, not reconstructed
raw screen frames. Supplied facts from a VLM are still model-produced evidence.
"""
from hypermem.adapters.os_trace import milliseconds
from hypermem.schema import AtomicObservation
from hypermem.schema.serialization import encode_json, parse_datetime, require_text

BUNDLE_TYPES = {"sms_message", "notification_received", "notification_opened",
                "screen_session_summary", "gallery_asset_group"}


class AtomicBundleAdapter:
    def parse(self, raw_record, *, observation_id, provenance):
        require_text(raw_record.get("observation_id"), "upstream observation_id")
        if raw_record.get("observation_type") not in BUNDLE_TYPES:
            raise ValueError("unsupported bundle observation type")
        for key in ("facts", "app", "inference", "evidence", "provenance"):
            if not isinstance(raw_record.get(key), dict):
                raise ValueError(f"bundle {key} must be an object")
        start = milliseconds(raw_record["timestamp_ms"])
        if parse_datetime(raw_record["timestamp"]) != start:
            raise ValueError("bundle timestamp and timestamp_ms disagree")
        app = raw_record["app"]
        source = app.get("name") or app.get("package") or raw_record.get("source_type")
        return AtomicObservation(observation_id, start, source, encode_json(raw_record), {}, provenance)


def grounding_inputs(raw):
    """Return field-level representations without promoting inferred intent.

Paths/offsets point directly into the losslessly stored observation content.
No inference.summary or inference.intent is used to assert observed actions.
"""
    kind, facts = raw["observation_type"], raw["facts"]
    fields = []

    def add(key, value, path, model=False):
        if isinstance(value, str) and value.strip():
            fields.append((key, value, "content." + path,
                           "upstream_model_description" if model else "supplied_text",
                           0.68 if model else 0.9))

    if kind == "sms_message":
        add("text", facts.get("message_body"), "facts.message_body")
    elif kind.startswith("notification_"):
        add("title", facts.get("title"), "facts.title")
        add("content", facts.get("body"), "facts.body")
    elif kind == "screen_session_summary":
        add("title", facts.get("topic"), "facts.topic", True)
        for i, action in enumerate(facts.get("observed_actions", [])):
            add("vlm_caption", action, f"facts.observed_actions[{i}]", True)
    elif kind == "gallery_asset_group":
        add("vlm_caption", facts.get("caption"), "facts.caption", True)
        for i, value in enumerate(facts.get("visible_text", [])):
            add("ocr", value, f"facts.visible_text[{i}]", True)
    return fields
