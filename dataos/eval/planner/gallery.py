"""Curated lm-eval task index used by the NL planner.

Each :class:`GalleryEntry` is a hand-picked, well-supported lm-eval task with
the metadata the planner needs to score / rank it against a user's natural
language goal: capability tags (``topics`` / ``skills``), output type, the
metric we'll surface as the headline, and sane default run knobs.

Adding a new task here makes it visible to the planner. ``output_type`` is the
lm-eval field, NOT the DataOS request type — entries with
``output_type != "generate_until"`` are filtered out by the resolver while
``DataOSLM.loglikelihood`` is unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class GalleryEntry:
    task: str
    description: str
    topics: tuple[str, ...]
    skills: tuple[str, ...]
    output_type: str  # "generate_until" or "multiple_choice"
    headline_metric: str
    default_limit: int = 200
    default_batch_size: int = 32
    default_max_new_tokens: int = 256
    default_num_fewshot: int | None = None
    requires_unsafe_code: bool = False
    extra_pip_packages: tuple[str, ...] = ()
    notes: str = ""


_BUILTIN: tuple[GalleryEntry, ...] = (
    GalleryEntry(
        task="gsm8k",
        description="Grade-school math word problems requiring multi-step arithmetic reasoning.",
        topics=("math", "reasoning", "arithmetic"),
        skills=("multi_step", "word_problem", "numeric_extraction"),
        output_type="generate_until",
        headline_metric="exact_match,flexible-extract",
        default_limit=200,
        default_batch_size=64,
        default_max_new_tokens=256,
        default_num_fewshot=0,
    ),
    GalleryEntry(
        task="ifeval",
        description="Verifiable instruction-following: format, length, content constraints.",
        topics=("instruction_following", "alignment"),
        skills=("constraint_following", "format_compliance"),
        output_type="generate_until",
        headline_metric="prompt_level_strict_acc,none",
        default_limit=100,
        default_batch_size=32,
        default_max_new_tokens=1280,
        default_num_fewshot=0,
        extra_pip_packages=("langdetect", "immutabledict", "nltk"),
    ),
    GalleryEntry(
        task="humaneval",
        description="Python function synthesis from a docstring; pass@1 via unit tests.",
        topics=("code", "code_generation", "programming"),
        skills=("python", "function_synthesis"),
        output_type="generate_until",
        headline_metric="pass@1,create_test",
        default_limit=164,
        default_batch_size=32,
        default_max_new_tokens=768,
        default_num_fewshot=0,
        requires_unsafe_code=True,
        notes="Requires HF_ALLOW_CODE_EVAL=1 and confirm_run_unsafe_code=True.",
    ),
    GalleryEntry(
        task="bbh_cot_zeroshot_logical_deduction_three_objects",
        description="BIG-Bench Hard subtask: zero-shot CoT logical deduction over three objects.",
        topics=("reasoning", "logic", "deduction"),
        skills=("multi_step", "chain_of_thought"),
        output_type="generate_until",
        headline_metric="exact_match,flexible-extract",
        default_limit=100,
        default_batch_size=64,
        default_max_new_tokens=512,
        default_num_fewshot=0,
    ),
    GalleryEntry(
        task="mmlu",
        description="Massive multitask multiple choice across 57 subjects.",
        topics=("knowledge", "general", "multiple_choice"),
        skills=("recall", "subject_knowledge"),
        output_type="multiple_choice",
        headline_metric="acc,none",
        default_limit=200,
        default_batch_size=8,
        default_max_new_tokens=8,
        default_num_fewshot=5,
        notes="Needs DataOSLM.loglikelihood; currently filtered out by resolver.",
    ),
    GalleryEntry(
        task="arc_challenge",
        description="AI2 Reasoning Challenge: science multiple choice (challenge split).",
        topics=("science", "reasoning", "multiple_choice"),
        skills=("subject_knowledge", "elimination"),
        output_type="multiple_choice",
        headline_metric="acc,none",
        default_limit=200,
        default_batch_size=8,
        default_max_new_tokens=8,
        default_num_fewshot=25,
        notes="Needs DataOSLM.loglikelihood; currently filtered out by resolver.",
    ),
    GalleryEntry(
        task="hellaswag",
        description="Commonsense sentence-completion multiple choice.",
        topics=("commonsense", "language", "multiple_choice"),
        skills=("completion", "commonsense_reasoning"),
        output_type="multiple_choice",
        headline_metric="acc_norm,none",
        default_limit=200,
        default_batch_size=8,
        default_max_new_tokens=8,
        default_num_fewshot=10,
        notes="Needs DataOSLM.loglikelihood; currently filtered out by resolver.",
    ),
    GalleryEntry(
        task="truthfulqa_mc2",
        description="Truthfulness multiple-choice (mc2 variant).",
        topics=("factuality", "truthfulness", "hallucination", "multiple_choice"),
        skills=("factual_recall", "calibration"),
        output_type="multiple_choice",
        headline_metric="acc,none",
        default_limit=200,
        default_batch_size=8,
        default_max_new_tokens=8,
        default_num_fewshot=0,
        notes="Needs DataOSLM.loglikelihood; currently filtered out by resolver.",
    ),
)


@dataclass(frozen=True)
class Gallery:
    entries: tuple[GalleryEntry, ...] = field(default_factory=tuple)

    def by_name(self, task: str) -> GalleryEntry | None:
        for entry in self.entries:
            if entry.task == task:
                return entry
        return None

    def filter(
        self,
        *,
        output_types: Iterable[str] | None = None,
        allow_unsafe_code: bool = False,
    ) -> "Gallery":
        allowed_outputs = set(output_types) if output_types is not None else None
        kept: list[GalleryEntry] = []
        for entry in self.entries:
            if allowed_outputs is not None and entry.output_type not in allowed_outputs:
                continue
            if entry.requires_unsafe_code and not allow_unsafe_code:
                continue
            kept.append(entry)
        return Gallery(entries=tuple(kept))


def builtin_gallery() -> Gallery:
    return Gallery(entries=_BUILTIN)
