"""Shared enum definitions for task- and operator-level DataOS concepts."""

from enum import StrEnum, unique

__all__ = [
    "TrainingStage",
    "IntelligenceLevel",
    "TaskType",
    "OperatorCategory",
    "StepStatus",
]


@unique
class TrainingStage(StrEnum):
    """Training lifecycle stages supported by DataOS experiments."""

    PRETRAIN = "pretrain"
    CONTINUAL_PRETRAIN = "continual_pretrain"
    SFT = "sft"
    RL = "rl"
    EVAL = "eval"


@unique
class IntelligenceLevel(StrEnum):
    """Execution sophistication levels for a task or experiment."""

    PROCESS = "process"
    OPTIMIZE_WITH_EVAL = "optimize_with_eval"
    CLOSED_LOOP_AGENTIC = "closed_loop_agentic"


@unique
class TaskType(StrEnum):
    """Task-level intents exposed to planners and experiment configs."""

    CLEAN = "clean"
    SELECT = "select"
    MIX = "mix"
    SYNTHESIZE = "synthesize"
    REWRITE = "rewrite"
    EVALUATE = "evaluate"
    ITERATE = "iterate"


@unique
class OperatorCategory(StrEnum):
    """Operator families used to group lower-level execution units."""

    CLEANING = "cleaning"
    SELECTION = "selection"
    SCORING = "scoring"
    MIXING = "mixing"
    SYNTHESIS = "synthesis"
    REWRITING = "rewriting"
    EVALUATION = "evaluation"
    UTILITY = "utility"


@unique
class StepStatus(StrEnum):
    """Lifecycle states for a pipeline step during execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
