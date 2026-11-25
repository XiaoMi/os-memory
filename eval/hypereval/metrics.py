# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import re
import string
from collections import Counter
from math import log2
from typing import Iterable


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def exact_match(prediction: str, references: Iterable[str]) -> float:
    value = normalize(prediction)
    return float(any(value == normalize(reference) for reference in references))


def token_f1(prediction: str, references: Iterable[str]) -> float:
    prediction_tokens = normalize(prediction).split()
    if not prediction_tokens:
        return 1.0 if any(not normalize(reference) for reference in references) else 0.0
    best = 0.0
    for reference in references:
        reference_tokens = normalize(reference).split()
        if not reference_tokens:
            continue
        common = sum((min(prediction_tokens.count(token), reference_tokens.count(token)) for token in set(prediction_tokens)))
        if not common:
            continue
        precision = common / len(prediction_tokens)
        recall = common / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def bleu1(prediction: str, references: Iterable[str]) -> float:
    """Compute sentence BLEU-1 with clipped unigram precision.

    The implementation is dependency-free and supports multiple references by
    taking the maximum clipped count for each token and the closest reference
    length for the brevity penalty.
    """
    prediction_tokens = normalize(prediction).split()
    reference_tokens = [normalize(reference).split() for reference in references]
    if not prediction_tokens:
        return 1.0 if any(not tokens for tokens in reference_tokens) else 0.0
    if not reference_tokens or not any(reference_tokens):
        return 0.0
    max_counts: dict[str, int] = {}
    for tokens in reference_tokens:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            max_counts[token] = max(max_counts.get(token, 0), count)
    clipped = sum(
        min(prediction_tokens.count(token), max_counts.get(token, 0))
        for token in set(prediction_tokens)
    )
    precision = clipped / len(prediction_tokens)
    if precision <= 0.0:
        return 0.0
    reference_length = min(
        (len(tokens) for tokens in reference_tokens if tokens),
        key=lambda length: (abs(length - len(prediction_tokens)), length),
    )
    brevity_penalty = 1.0
    if len(prediction_tokens) < reference_length:
        brevity_penalty = pow(2.718281828, 1 - reference_length / len(prediction_tokens))
    return brevity_penalty * precision


def _gallery_normalize(text: str) -> str:
    """Normalize text exactly as Mem-Gallery's official evaluator does."""
    value = str(text).lower()
    value = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", value)
    value = value.replace("_", "UNDERSCOREPLACEHOLDER")
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    value = "".join(
        character if character not in string.punctuation else " "
        for character in value
    )
    return " ".join(
        value.replace("DOTPLACEHOLDER", ".")
        .replace("UNDERSCOREPLACEHOLDER", "_")
        .split()
    )


_PORTER_STEMMER = None


def _porter_stem(token: str) -> str:
    global _PORTER_STEMMER
    if _PORTER_STEMMER is None:
        try:
            from nltk.stem import PorterStemmer
        except ImportError as exc:
            raise RuntimeError(
                "Official LoCoMo/Mem-Gallery F1 requires nltk in the active environment"
            ) from exc
        _PORTER_STEMMER = PorterStemmer()
    return _PORTER_STEMMER.stem(token)


def gallery_exact_match(prediction: str, references: Iterable[str]) -> float:
    normalized = _gallery_normalize(prediction)
    return float(
        any(normalized == _gallery_normalize(reference) for reference in references)
    )


def gallery_f1(prediction: str, references: Iterable[str]) -> float:
    prediction_tokens = [
        _porter_stem(token) for token in _gallery_normalize(prediction).split()
    ]
    if not prediction_tokens:
        return 0.0
    best = 0.0
    for reference in references:
        reference_tokens = [
            _porter_stem(token) for token in _gallery_normalize(reference).split()
        ]
        if not reference_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if not overlap:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def gallery_bleu(
    prediction: str,
    references: Iterable[str],
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:
        raise RuntimeError(
            "Official Mem-Gallery BLEU requires nltk in the active environment"
        ) from exc
    prediction_tokens = _gallery_normalize(prediction).split()
    if not prediction_tokens:
        return 0.0
    scores = []
    for reference in references:
        reference_tokens = _gallery_normalize(reference).split()
        if reference_tokens:
            scores.append(
                sentence_bleu(
                    [reference_tokens],
                    prediction_tokens,
                    weights=weights,
                    smoothing_function=SmoothingFunction().method1,
                )
            )
    return max(scores, default=0.0)


def evidence_recall_at_k(record_ids: list[str], evidence_ids: Iterable[str], k: int) -> float:
    evidence = set(evidence_ids)
    if not evidence:
        return 0.0
    return float(bool(evidence & set(record_ids[:k])))


def reciprocal_rank(record_ids: list[str], evidence_ids: Iterable[str]) -> float:
    evidence = set(evidence_ids)
    for index, record_id in enumerate(record_ids, start=1):
        if record_id in evidence:
            return 1.0 / index
    return 0.0


def evidence_recall_at_k_multi(
    record_source_ids: list[Iterable[str]], evidence_ids: Iterable[str], k: int
) -> float:
    """Recall@k where each retrieved record may map back to several source events.

    A hit means any evidence id appears among the source-event ids of the top-k
    records. Use this when records carry ``source_event_ids`` (traceable back to
    the dataset's native evidence ids) rather than opaque memory ids.
    """
    evidence = set(evidence_ids)
    if not evidence:
        return 0.0
    top = set()
    for source_ids in record_source_ids[:k]:
        top.update(source_ids)
    return float(bool(evidence & top))


def reciprocal_rank_multi(
    record_source_ids: list[Iterable[str]], evidence_ids: Iterable[str]
) -> float:
    """Reciprocal rank of the first record whose source events hit the evidence."""
    evidence = set(evidence_ids)
    if not evidence:
        return 0.0
    for index, source_ids in enumerate(record_source_ids, start=1):
        if evidence & set(source_ids):
            return 1.0 / index
    return 0.0


def locomo_qa_score(prediction: str, references: Iterable[str], category: str | None) -> float:
    """Reproduce LoCoMo's official category-dependent QA scoring."""
    category_id = str(category or "")
    if category_id == "5":
        lowered = prediction.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    answers = [str(reference) for reference in references]
    if not answers:
        return 0.0
    answer = answers[0]
    if category_id == "3":
        answer = answer.split(";", 1)[0].strip()

    def official_f1(candidate: str, gold: str) -> float:
        def tokens(value: str) -> list[str]:
            value = value.replace(",", "").lower()
            value = re.sub(r"\b(a|an|the|and)\b", " ", value)
            value = "".join(
                character for character in value if character not in string.punctuation
            )
            return [_porter_stem(token) for token in value.split()]

        candidate_tokens = tokens(candidate)
        gold_tokens = tokens(gold)
        if not candidate_tokens or not gold_tokens:
            return 0.0
        overlap = sum((Counter(candidate_tokens) & Counter(gold_tokens)).values())
        if not overlap:
            return 0.0
        precision = overlap / len(candidate_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    if category_id == "1":
        candidates = [part.strip() for part in prediction.split(",")]
        golds = [part.strip() for part in answer.split(",")]
        if not golds:
            return 0.0
        return sum(
            max((official_f1(candidate, gold) for candidate in candidates), default=0.0)
            for gold in golds
        ) / len(golds)
    return official_f1(prediction, answer)


def precision_recall_hitrate_at_k(
    record_source_ids: list[Iterable[str]], evidence_ids: Iterable[str], k: int
) -> tuple[float, float, float]:
    """Mem-Gallery's official retrieval Precision/Recall/HitRate@K."""
    ranked: list[str] = []
    for source_ids in record_source_ids:
        for source_id in source_ids:
            ranked.append(str(source_id))
            if len(ranked) >= k:
                break
        if len(ranked) >= k:
            break

    retrieved: list[str] = []
    seen: set[str] = set()
    for value in ranked:
        if value not in seen:
            retrieved.append(value)
            seen.add(value)
    gold = {str(value) for value in evidence_ids}
    hits = sum(value in gold for value in retrieved)
    precision = hits / max(1, len(retrieved))
    recall = hits / len(gold) if gold else 0.0
    return precision, recall, float(hits > 0)


def longmemeval_retrieval_metrics(
    record_source_ids: list[Iterable[str]], evidence_ids: Iterable[str], k: int
) -> tuple[float, float, float]:
    """Return official Recall-any/Recall-all/NDCG-any at turn granularity."""
    ranked: list[str] = []
    seen: set[str] = set()
    for source_ids in record_source_ids:
        for source_id in source_ids:
            value = str(source_id)
            if value not in seen:
                ranked.append(value)
                seen.add(value)
                if len(ranked) >= k:
                    break
        if len(ranked) >= k:
            break
    gold = {str(value) for value in evidence_ids}
    if not gold:
        return 0.0, 0.0, 0.0
    retrieved = set(ranked)
    recall_any = float(bool(gold & retrieved))
    recall_all = float(gold <= retrieved)
    relevance = [float(value in gold) for value in ranked]
    dcg = sum(
        score if index == 0 else score / log2(index + 1)
        for index, score in enumerate(relevance)
    )
    ideal_count = min(k, len(gold))
    ideal = 0.0
    if ideal_count:
        ideal = 1.0 + sum(
            1.0 / log2(index + 1) for index in range(1, ideal_count)
        )
    return recall_any, recall_all, dcg / ideal if ideal else 0.0


def longmemeval_session_retrieval_metrics(
    record_source_ids: list[Iterable[str]],
    evidence_session_ids: Iterable[str],
    k: int,
    event_session_ids: dict[str, str] | None = None,
    corpus_session_ids: Iterable[str] | None = None,
) -> tuple[float, float, float]:
    """Official session metrics derived from a turn-level retrieval ranking."""
    def session_id(source_id: str) -> str:
        if event_session_ids and source_id in event_session_ids:
            return str(event_session_ids[source_id])
        head, marker, _ = source_id.rpartition("-e")
        if not marker:
            return source_id
        _, separator, session = head.rpartition("-")
        return session if separator else head

    ranked_sessions: list[str] = []
    seen_events: set[str] = set()
    for source_ids in record_source_ids:
        for source_id in source_ids:
            event_id = str(source_id)
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            ranked_sessions.append(session_id(event_id))

    effective_k = min(k, len(ranked_sessions))
    while (
        effective_k < len(ranked_sessions)
        and len(set(ranked_sessions[:effective_k])) < k
    ):
        effective_k += 1

    ranked = ranked_sessions[:effective_k]
    gold = {str(value) for value in evidence_session_ids}
    if not gold:
        return 0.0, 0.0, 0.0
    retrieved = set(ranked)
    recall_any = float(bool(gold & retrieved))
    recall_all = float(gold <= retrieved)
    relevance = [float(value in gold) for value in ranked]
    dcg = sum(
        score if index == 0 else score / log2(index + 1)
        for index, score in enumerate(relevance)
    )
    corpus_relevance = [
        float(str(value) in gold) for value in (corpus_session_ids or ())
    ]
    ideal_count = min(effective_k, int(sum(corpus_relevance)))
    ideal = 0.0
    if ideal_count:
        ideal = 1.0 + sum(
            1.0 / log2(index + 1) for index in range(1, ideal_count)
        )
    return recall_any, recall_all, dcg / ideal if ideal else 0.0
