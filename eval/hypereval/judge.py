# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""LLM-based answering and grading for HyperMemory Eval evaluation.

This module makes HyperMemory Eval's scoring comparable to how memory-system papers
(e.g. LightMem/LoCoMo) actually evaluate: an LLM reads the retrieved memories
and produces a short answer, then an LLM judge grades that answer against the
gold answer with generous, semantics-aware criteria (dates in any format, same
topic, etc.). The QA and judge prompts mirror LightMem's official
``locomo-question-answering-flat-memory-system`` and ``locomo-judge`` prompts so
the numbers line up with the official grading口径.

Both calls go through one OpenAI-compatible endpoint configured by env vars
(shared with the systems under test). The model defaults to the value passed by
the runner. All calls are best-effort: on any failure they degrade gracefully
(answer falls back to concatenated memories; judge falls back to token-F1) so a
single flaky request never aborts a whole benchmark run.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from string import Template
from typing import Iterable, Sequence

# Official LightMem LoCoMo QA prompt (flat memory), verbatim except we inline the
# $context / $question substitution here.
_QA_PROMPT = Template(
    "You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n"
    "# CONTEXT:\n"
    "You have access to memories from two speakers in a conversation. These memories contain "
    "timestamped information that may be relevant to answering the question.\n\n"
    "# INSTRUCTIONS:\n"
    "1. Carefully analyze all provided memories from both speakers\n"
    "2. Pay special attention to the timestamps to determine the answer\n"
    "3. If the question asks about a specific event or fact, look for direct evidence in the memories\n"
    "4. If the memories contain contradictory information, prioritize the most recent memory\n"
    "5. If there is a question about time references (like 'last year', 'two months ago', etc.), "
    "calculate the actual date based on the memory timestamp. For example, if a memory from "
    "4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.\n"
    "6. Always convert relative time references to specific dates, months, or years. For example, "
    "convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory "
    "timestamp. Ignore the reference while answering the question.\n"
    "7. Focus only on the content of the memories from both speakers. Do not confuse character "
    "names mentioned in memories with the actual users who created those memories.\n"
    "8. The answer should be less than 5-6 words.\n\n"
    "# APPROACH (Think step by step):\n"
    "1. First, examine all memories that contain information related to the question\n"
    "2. Examine the timestamps and content of these memories carefully\n"
    "3. Look for explicit mentions of dates, times, locations, or events that answer the question\n"
    "4. If the answer requires calculation (e.g., converting relative time references), show your work\n"
    "5. Formulate a precise, concise answer based solely on the evidence in the memories\n"
    "6. Double-check that your answer directly addresses the question asked\n"
    "7. Ensure your final answer is specific and avoids vague time references\n\n"
    "Memories:\n\n"
    "$context\n\n"
    "Question: $question\n\n"
    "Answer:"
)

# PersonaMem multiple-choice prompt. Mirrors the official PersonaMem inference
# (bowen-upenn/PersonaMem): the model reads the (retrieved) interaction memories
# plus the user's latest message, then picks the single most appropriate
# assistant reply from 4 candidates and emits its letter after <final_answer>.
# Here ``$context`` is the RETRIEVED memories (not the full conversation) — that
# is the point of evaluating a memory system.
_MCQ_PROMPT = Template(
    "You are a personalized chatbot assistant. Below are memories retrieved from "
    "your previous interactions with this user, which capture their profile, "
    "shared facts, and evolving preferences.\n\n"
    "# Retrieved memories:\n"
    "$context\n\n"
    "# The user's latest message:\n"
    "$question\n\n"
    "# Candidate assistant responses:\n"
    "$options\n\n"
    "Based ONLY on what the memories tell you about this user, find the most "
    "appropriate assistant response. Think briefly, then give your final answer "
    "as one of (a), (b), (c), or (d) after the special token <final_answer>.\n"
    "For example: <final_answer> (b)"
)

# Official LightMem locomo-judge prompt, verbatim.
_JUDGE_PROMPT = Template(
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: "
    "(1) a question (posed by one user to another user), "
    "(2) a 'gold' (ground truth) answer, "
    "(3) a generated answer "
    "which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
    "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
    "Now it's time for the real question:\n"
    "Question: $question\n"
    "Gold answer: $golden_answers\n"
    "Generated answer: $prediction\n\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
    "Just return the label CORRECT or WRONG in a json format with the key as 'label'."
)

_LOCOMO_CAT5_PROMPT = Template(
    "You are answering a LoCoMo adversarial question using only the retrieved conversation memories.\n\n"
    "Memories:\n$context\n\n"
    "Question: $question\n\n"
    "Choose which answer is supported by the memories:\n"
    "(a) Not mentioned in the conversation\n"
    "(b) $adversarial_answer\n\n"
    "Return exactly the full text of the supported answer, not the option letter."
)

_MEMBENCH_MCQ_PROMPT = Template(
    "Please answer the following question based on $memory_scope.\n"
    "Past memory:\n$context\n"
    "Question: $question\n"
    "Choices:\n$options\n"
    "Please output the correct option for the question, only one corresponding "
    "letter, without any other messages.\n"
    "Example: D"
)

_LONGMEM_QA_PROMPT = Template("""You are answering a long-term memory benchmark question.
Use only the retrieved memory snippets below. Give the shortest complete answer
supported by the memories. Preserve all required names, values, dates, and
update states. For an abstention question, say clearly that the information is
not known when the memories do not support an answer. Do not mention retrieval
or invent facts.

Question type: $question_type
Question date: $question_date

Retrieved memories:
$context

Question: $question

Answer:""")

_GALLERYMEM_QA_PROMPT = Template("""You are an AI assistant evaluated on multimodal long-term conversational memory.
Answer the question concisely, but completely enough to be accurate. If
multiple pieces of information about the same event appear, rely on the most
recent information. The retrieved memories are the relevant parts of the
historical conversation. Historical image captions and image ids are part of
those memories. If a question image is attached, inspect that image directly;
it is part of the current question input and does not need to appear in memory.
Do not hallucinate.

Task type: $question_type
Retrieved memories:
$context

Current question: $question$question_image
$format_constraint

Only provide the answer content, without an introductory phrase.
Answer:""")

_GALLERYMEM_JUDGE_PROMPT = Template("""You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity.
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**
- The answer touches on the topic but misses the core entity or key value required.
- The answer contains a mix of minor correct details and significant hallucinations or wrong associations.
- The answer is excessively vague to the point of being useless.

**Score 0.5 (Partial / Vague):**
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the main entity or concept correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed, or the assistant is uncertain.
- For Open-ended questions: The answer is too general or misses key adjectives or details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**
- The answer is largely accurate and captures the core information confidently.
- It misses only minor details that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary filler that reduces precision.

**Score 1 (Correct / Exact):**
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains all core information and necessary details without hallucinations.

### Input Data

Question: $question
Ground Truth: $golden_answers
Assistant Answer: $prediction

### Output Format

Output strictly in the following JSON format:
{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "<short explanation>"}""")


class LLMEvaluator:
    """Wraps one OpenAI-compatible client for answering and grading.

    Credentials/endpoint come from the same env vars the systems use
    (OPENAI_API_KEY / OPENAI_BASE_URL by default; overridable). Construction is
    lazy-safe: if the openai package or key is missing, ``available`` is False
    and callers fall back to non-LLM behavior.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        longmemeval_judge_model: str | None = None,
    ) -> None:
        self.model = model
        self.longmemeval_judge_model = longmemeval_judge_model or os.getenv(
            "HYPEREVAL_LONGMEMEVAL_JUDGE_MODEL", "azure_openai/gpt-4o"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        key = api_key or os.getenv("OPENAI_API_KEY")
        url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if not key:
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _complete(self, prompt: str, model: str | None = None) -> str:
        return self._complete_content(prompt, model=model)

    def _complete_content(self, content: object, model: str | None = None) -> str:
        attempts = int(os.getenv("HYPEREVAL_EVALUATOR_RETRIES", "3"))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=model or self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=float(os.getenv("HYPEREVAL_EVALUATOR_TIMEOUT", "120")),
                )
                if isinstance(response, str):
                    content = response.strip()
                    if _looks_like_html(content):
                        raise RuntimeError("evaluator gateway returned HTML")
                    if content:
                        return response
                    raise RuntimeError("evaluator gateway returned an empty string")
                choices = getattr(response, "choices", None)
                if not choices:
                    raise RuntimeError("evaluator gateway returned no choices")
                content = choices[0].message.content or ""
                if _looks_like_html(content):
                    raise RuntimeError("evaluator gateway returned HTML")
                if content.strip():
                    return content
                raise RuntimeError("evaluator gateway returned empty content")
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"evaluator request failed after {attempts} attempts: {last_error}"
        )

    def generate_answer(
        self,
        question: str,
        memories: Sequence[str],
        dataset: str | None = None,
        question_type: str | None = None,
        question_date: str | None = None,
        question_metadata: dict[str, object] | None = None,
    ) -> str:
        """Produce a short answer to ``question`` grounded in ``memories``.

        Falls back to the concatenated memories if the LLM is unavailable or
        errors, so downstream metrics still receive a (weaker) prediction.
        """
        context = "\n".join(memories)
        if not self.available:
            return context
        if dataset == "locomo" and str(question_type or "") == "5":
            metadata = question_metadata or {}
            prompt = _LOCOMO_CAT5_PROMPT.substitute(
                context=context or "(no memories retrieved)",
                question=question,
                adversarial_answer=str(metadata.get("adversarial_answer") or "Unknown answer"),
            )
        elif dataset == "longmemeval":
            prompt = _LONGMEM_QA_PROMPT.substitute(
                context=context or "(no memories retrieved)",
                question=question,
                question_type=question_type or "unknown",
                question_date=question_date or "unknown",
            )
        elif dataset == "gallerymem":
            metadata = question_metadata or {}
            caption = str(metadata.get("question_image_caption") or "").strip()
            image_paths = [
                str(value)
                for value in metadata.get("question_image_paths") or []
                if str(value).strip()
            ]
            image_note = (
                f"\nAttached question image caption: {caption}"
                if caption and not image_paths
                else ""
            )
            point = str(question_type or "").upper()
            constraints = {
                "AR": "If absent from the conversation, reply exactly: Not mentioned.",
                "CD": "Reply strictly with either Yes. or No.",
                "VS": "Return only matching image_id values, sorted ascending and comma-separated.",
            }
            prompt = _GALLERYMEM_QA_PROMPT.substitute(
                context=context or "(no memories retrieved)",
                question=question,
                question_type=point or "unknown",
                question_image=image_note,
                format_constraint=constraints.get(point, ""),
            )
        else:
            prompt = _QA_PROMPT.substitute(context=context, question=question)
        try:
            if dataset == "gallerymem" and image_paths:
                content: list[dict[str, object]] = [
                    {"type": "text", "text": prompt}
                ]
                for image_path in image_paths:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        }
                    )
                answer = self._complete_content(content).strip()
            else:
                answer = self._complete(prompt).strip()
            return answer or context
        except Exception:
            return context

    def judge(
        self,
        question: str,
        prediction: str,
        references: Iterable[str],
        dataset: str | None = None,
        question_type: str | None = None,
        question_metadata: dict[str, object] | None = None,
    ) -> float | None:
        """Return 1.0 if the judge labels the prediction CORRECT, else 0.0.

        Returns None (not 0.0) when the LLM judge is unavailable or errors, so
        the caller can distinguish "judge said wrong" from "judge did not run"
        and fall back to another metric.
        """
        if not self.available:
            return None
        gold = "; ".join(str(r) for r in references if str(r).strip())
        if dataset == "longmemeval":
            prompt = _longmemeval_judge_prompt(
                question_type or "unknown",
                question,
                gold,
                prediction,
                bool((question_metadata or {}).get("abstention")),
            )
        elif dataset == "gallerymem":
            prompt = _GALLERYMEM_JUDGE_PROMPT.substitute(
                question=question, golden_answers=gold, prediction=prediction
            )
        else:
            prompt = _JUDGE_PROMPT.substitute(
                question=question, golden_answers=gold, prediction=prediction
            )
        try:
            content = self._complete(
                prompt,
                model=self.longmemeval_judge_model if dataset == "longmemeval" else None,
            )
        except Exception:
            return None
        if dataset == "gallerymem":
            return _parse_gallerymem_score(content)
        if dataset == "longmemeval":
            return _parse_yes_no_label(content)
        return _parse_judge_label(content)

    def answer_mcq(
        self,
        question: str,
        memories: Sequence[str],
        options: Sequence[str],
        dataset: str | None = None,
        question_metadata: dict[str, object] | None = None,
    ) -> str:
        """Pick a multiple-choice option grounded in the retrieved memories.

        Returns the raw model response (expected to contain ``<final_answer>
        (x)``). Falls back to an empty string if the LLM is unavailable or
        errors, so ``score_mcq`` will score it 0 rather than crash.
        """
        if not self.available:
            return ""
        context = "\n".join(memories) if memories else "(no memories retrieved)"
        options_block = "\n".join(str(o) for o in options)
        if dataset == "membench":
            agent = str((question_metadata or {}).get("agent") or "")
            memory_scope = (
                "past memories of the user's messages"
                if agent == "ThirdAgent"
                else "past memories of your conversation with the user"
            )
            prompt = _MEMBENCH_MCQ_PROMPT.substitute(
                context=context,
                question=question,
                options=options_block,
                memory_scope=memory_scope,
            )
        else:
            prompt = _MCQ_PROMPT.substitute(
                context=context, question=question, options=options_block
            )
        try:
            return self._complete(prompt).strip()
        except Exception:
            return ""

    @staticmethod
    def score_mcq(response: str, correct_option: str) -> float:
        """1.0 if the letter parsed from ``response`` == ``correct_option``.

        Mirrors the official PersonaMem parser: prefer text after the
        ``<final_answer>`` token, then match ``(x)`` and fall back to a bare
        letter. Returns 0.0 when nothing parses or the letter differs.
        """
        gold = _option_letter_from_text(correct_option)
        if gold is None:
            return 0.0
        pred = _parse_mcq_letter(response)
        return 1.0 if pred is not None and pred == gold else 0.0


def _option_letter_from_text(text: str) -> str | None:
    """Normalise a correct-option field (``(c)`` / ``c`` / ``C``) to ``c``."""
    if not text:
        return None
    match = re.search(r"\(?\s*([a-dA-D])\s*\)?", text.strip())
    return match.group(1).lower() if match else None


def _looks_like_html(content: str) -> bool:
    normalized = str(content or "").lstrip().lower()
    return normalized.startswith("<!doctype html") or normalized.startswith("<html")


def _image_data_url(path: str) -> str:
    image_path = Path(path)
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{payload}"


def _parse_mcq_letter(response: str) -> str | None:
    """Extract one unambiguous option letter using PersonaMem's parser."""
    if not response:
        return None
    text = response
    if "<final_answer>" in text:
        text = text.split("<final_answer>")[-1].strip()
    matches = re.findall(r"\(([a-dA-D])\)", text)
    if not matches:
        matches = re.findall(r"\b([a-dA-D])\b", text)
    options = {match.lower() for match in matches}
    return next(iter(options)) if len(options) == 1 else None


def _parse_judge_label(content: str) -> float | None:
    """Extract CORRECT/WRONG from a judge response.

    Prefers the JSON ``{"label": ...}`` the prompt asks for, then falls back to
    scanning the text. Mirrors the official口径 (generous CORRECT detection) while
    ensuring an explicit WRONG is never misread as correct.
    """
    if not content:
        return None
    lowered = content.lower()
    # Prefer an explicit label value if present.
    match = re.search(r'"label"\s*:\s*"?\s*(correct|wrong)', lowered)
    if match:
        return 1.0 if match.group(1) == "correct" else 0.0
    has_correct = "correct" in lowered
    has_wrong = "wrong" in lowered
    if has_correct and not has_wrong:
        return 1.0
    if has_wrong and not has_correct:
        return 0.0
    # Ambiguous or unparseable.
    return None


def _parse_gallerymem_score(content: str) -> float | None:
    if not content:
        return None
    match = re.search(
        r'"score"\s*:\s*(0(?:\.0+)?|0\.25|0\.5|0\.75|1(?:\.0+)?)', content
    )
    if not match:
        return None
    value = float(match.group(1))
    return value if value in {0.0, 0.25, 0.5, 0.75, 1.0} else None


def _longmemeval_judge_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
) -> str:
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}"
            "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)
    if question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        extra = ""
        noun = "correct answer"
    elif question_type == "temporal-reasoning":
        extra = (
            " In addition, do not penalize off-by-one errors for the number of days. If the question asks "
            "for the number of days/weeks/months, etc., and the model makes off-by-one errors, the model's "
            "response is still correct."
        )
        noun = "correct answer"
    elif question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes "
            "if the response contains the correct answer. Otherwise, answer no. If the response contains "
            "some previous information along with an updated answer, the response should be considered as "
            "correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
            "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        ).format(question, answer, response)
    elif question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}"
            "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        ).format(question, answer, response)
    else:
        extra = ""
        noun = "correct answer"
    return (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
        "correct answer or contains all the intermediate steps to get the correct answer, you should also "
        "answer yes. If the response only contains a subset of the information required by the answer, answer no."
        + extra
        + "\n\nQuestion: {}\n\n{}: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    ).format(question, noun.title(), answer, response)


def _parse_yes_no_label(content: str) -> float | None:
    if not content:
        return None
    normalized = content.strip().lower()
    if re.search(r"\byes\b", normalized) and not re.search(r"\bno\b", normalized):
        return 1.0
    if re.search(r"\bno\b", normalized) and not re.search(r"\byes\b", normalized):
        return 0.0
    return None
