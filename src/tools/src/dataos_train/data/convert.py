"""Canonical SFT data -> LLaMA-Factory alpaca on disk.

Runs INSIDE the training job (it reads the dataset), invoked as
``python -m dataos_train.data.convert --in <path> --informat auto --out <dir>``.
Heavy deps (``datasets``/``pyarrow``) are imported lazily, only for HF/parquet inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataos_launch.io import atomic_write_text


def read_records(path: str, fmt: str = "auto") -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():  # HF datasets save_to_disk dir
        from datasets import load_from_disk

        return [dict(r) for r in load_from_disk(str(p))]
    suffix = p.suffix.lower()
    if suffix == ".jsonl" or fmt == "jsonl":
        return [
            json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return pq.read_table(str(p)).to_pylist()
        except ImportError:
            from datasets import Dataset

            return [dict(r) for r in Dataset.from_parquet(str(p))]
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array of records in {path}")
    return data


def _detect(rec: dict) -> str:
    if "messages" in rec:
        return "messages"
    if "conversations" in rec:
        return "sharegpt"
    if "instruction" in rec or "output" in rec:
        return "alpaca"
    if "question" in rec and "answer" in rec:
        return "qa"
    if "prompt" in rec and any(k in rec for k in ("output", "target", "response")):
        return "prompt"
    return "alpaca"


def _from_messages(msgs: list[dict]) -> dict:
    system = ""
    turns: list[tuple[str, str]] = []
    for m in msgs:
        role, content = m.get("role"), m.get("content", "")
        if role == "system":
            system = content
        elif role:
            turns.append((role, content))
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns):
        if turns[i][0] == "user" and i + 1 < len(turns) and turns[i + 1][0] == "assistant":
            pairs.append((turns[i][1], turns[i + 1][1]))
            i += 2
        else:
            i += 1
    instruction, output, history = "", "", []
    if pairs:
        instruction, output = pairs[-1]
        history = [[u, a] for u, a in pairs[:-1]]
    rec: dict[str, Any] = {"instruction": instruction, "input": "", "output": output}
    if system:
        rec["system"] = system
    if history:
        rec["history"] = history
    return rec


def _from_sharegpt(convs: list[dict]) -> dict:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = [
        {
            "role": role_map.get(c.get("from", ""), c.get("from", "user")),
            "content": c.get("value", ""),
        }
        for c in convs
    ]
    return _from_messages(msgs)


def _prompt_from_messages(msgs: list[dict]) -> dict[str, Any]:
    system = ""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in msgs:
        role, content = m.get("role"), m.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None

    if pending_user is not None:
        instruction = pending_user
        history = pairs
    elif pairs:
        instruction = pairs[-1][0]
        history = pairs[:-1]
    else:
        instruction = ""
        history = []

    rec: dict[str, Any] = {"instruction": instruction, "input": ""}
    if system:
        rec["system"] = system
    if history:
        rec["history"] = [[u, a] for u, a in history]
    return rec


def _prompt_from_sharegpt(convs: list[dict]) -> dict[str, Any]:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = [
        {
            "role": role_map.get(c.get("from", ""), c.get("from", "user")),
            "content": c.get("value", ""),
        }
        for c in convs
    ]
    return _prompt_from_messages(msgs)


def _normalize_history(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("history must be a list of [prompt, response] pairs")
    return [list(turn) if isinstance(turn, tuple) else turn for turn in value]


def _with_lf_optional_prompt_columns(item: dict[str, Any]) -> dict[str, Any]:
    """Materialize columns declared in LF dataset_info.json.

    LLaMA-Factory 0.9.x indexes optional mapped columns such as ``history``
    directly when they are present in dataset_info.json, so rows must carry
    explicit empty values instead of omitting the keys.
    """
    out = dict(item)
    system = out.get("system", "")
    out["system"] = "" if system is None else str(system)
    out["history"] = _normalize_history(out.get("history"))
    return out


def _content_value(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content", value.get("value"))
        if isinstance(content, str):
            return content
    raise ValueError(f"{field} must be a string or message object")


def _bool_value(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("true", "1", "yes", "y", "desirable", "chosen", "positive"):
            return True
        if val in ("false", "0", "no", "n", "undesirable", "rejected", "negative"):
            return False
    raise ValueError(f"{field} must be a boolean or a recognizable boolean string")


def _prompt_base(rec: dict[str, Any], source_fmt: str = "auto") -> dict[str, Any]:
    fmt = source_fmt if source_fmt != "auto" else _detect(rec)
    if fmt == "messages":
        return _with_lf_optional_prompt_columns(_prompt_from_messages(rec.get("messages", [])))
    if fmt == "sharegpt":
        return _with_lf_optional_prompt_columns(_prompt_from_sharegpt(rec.get("conversations", [])))

    item: dict[str, Any] = {
        "instruction": rec.get("instruction", rec.get("prompt", rec.get("question", ""))),
        "input": rec.get("input", rec.get("query", "")),
    }
    if rec.get("system"):
        item["system"] = rec["system"]
    if rec.get("history"):
        item["history"] = rec["history"]
    return _with_lf_optional_prompt_columns(item)


def to_alpaca(records: list[dict], source_fmt: str = "auto") -> list[dict]:
    out: list[dict] = []
    for rec in records:
        fmt = source_fmt if source_fmt != "auto" else _detect(rec)
        if fmt == "messages":
            out.append(_from_messages(rec.get("messages", [])))
        elif fmt == "sharegpt":
            out.append(_from_sharegpt(rec.get("conversations", [])))
        elif fmt == "prompt":
            out.append(
                {
                    "instruction": rec.get("prompt", ""),
                    "input": "",
                    "output": rec.get("output", rec.get("target", rec.get("response", ""))),
                }
            )
        elif fmt == "qa":
            out.append(
                {
                    "instruction": rec.get("question", ""),
                    "input": "",
                    "output": rec.get("answer", ""),
                }
            )
        else:  # alpaca (or fallback)
            item = {
                "instruction": rec.get("instruction", rec.get("prompt", "")),
                "input": rec.get("input", ""),
                "output": rec.get("output", rec.get("target", rec.get("response", ""))),
            }
            if rec.get("system"):
                item["system"] = rec["system"]
            if rec.get("history"):
                item["history"] = rec["history"]
            out.append(item)
    return out


def to_preference(records: list[dict], source_fmt: str = "auto") -> list[dict]:
    """Convert canonical preference rows to LF alpaca pairwise format."""
    out: list[dict] = []
    for rec in records:
        if "chosen" not in rec or "rejected" not in rec:
            raise ValueError("preference records require chosen and rejected fields")
        item = _prompt_base(rec, source_fmt)
        item["chosen"] = _content_value(rec["chosen"], "chosen")
        item["rejected"] = _content_value(rec["rejected"], "rejected")
        out.append(item)
    return out


def to_kto(records: list[dict], source_fmt: str = "auto") -> list[dict]:
    """Convert canonical rows to LF KTO alpaca format with boolean ``kto_tag``."""
    out: list[dict] = []
    for rec in records:
        item = _prompt_base(rec, source_fmt)
        sft = to_alpaca([rec], source_fmt)[0]
        item["output"] = rec.get(
            "output",
            rec.get("target", rec.get("response", rec.get("completion", sft.get("output", "")))),
        )
        tag = rec.get("kto_tag", rec.get("label", rec.get("feedback")))
        item["kto_tag"] = _bool_value(tag, "kto_tag")
        out.append(item)
    return out


def to_ppo(records: list[dict], source_fmt: str = "auto") -> list[dict]:
    """Convert prompt rows to LF unsupervised/PPO alpaca format."""
    out: list[dict] = []
    for rec in records:
        item = _prompt_base(rec, source_fmt)
        item["output"] = rec.get("output", rec.get("target", rec.get("response", "")))
        out.append(item)
    return out


#: LLaMA-Factory only indexes the optional alpaca columns it was TOLD about: its own
#: ``DatasetAttr`` defaults ``system`` and ``history`` to None, so an undeclared column is dropped
#: no matter that the rows carry it. This writer used to omit the map while the DPO/KTO/PPO writers
#: below all declared theirs — and :func:`_with_lf_optional_prompt_columns` deliberately
#: materialises ``system`` into every row, so the field landed on disk and vanished in training.
#: The 2026-08-11
#: bfcl batch paid 33 points for that (85 -> 52/49/51 once the recipe moved its tool schema from
#: ``instruction`` into ``system``), and the leader attributed it to sampling, lr and randomness.
#: A row whose ``system`` is empty still falls back to the template's ``default_system`` — LF does
#: ``system = system or self.default_system``, replace not append, so declaring this cannot
#: double up the system turn.
_ALPACA_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history",
}


def write_alpaca_dataset(records: list[dict], out_dir: str, name: str = "dataos_sft") -> int:
    """Write the alpaca dataset LF will read, guaranteeing every column it is told about.

    THE INVARIANT: whatever is declared here is present in every row. LF's alpaca converter
    indexes an optional column unconditionally once it is mapped — ``if self.dataset_attr.history
    and isinstance(example[self.dataset_attr.history], list)`` short-circuits on the column NAME
    being set, then subscripts the row — so declaring a column the rows may lack turns
    ``dataset_info.json`` into a landmine. Declaring and guaranteeing therefore have to live in one
    function, which is what this is. Getting only the declaring half done cost the Gemma3 bfcl run
    its last round to ``KeyError: 'history'`` 12 seconds into every job.

    Neither column is optional in the sense of "unused": ``to_alpaca`` emits ``history`` for
    multi-turn records and omits it for single-turn ones, so a MIXED dataset is the worst case
    (some rows carry the key, some do not); and ``system`` is where the frozen bfcl scorer puts the
    tool schema, which is why it is declared at all (audit 13.4 — 85 -> 52 when LF silently dropped
    it). Materializing is the only way to keep both without the landmine.
    """
    rows = [_with_lf_optional_prompt_columns(rec) for rec in records]
    return _write_dataset(
        rows, out_dir, name, {"file_name": "train.json", "columns": dict(_ALPACA_COLUMNS)}
    )


def _write_dataset(records: list[dict], out_dir: str, name: str, info: dict[str, Any]) -> int:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / "train.json", json.dumps(records, ensure_ascii=False))
    atomic_write_text(d / "dataset_info.json", json.dumps({name: info}, ensure_ascii=False))
    return len(records)


def write_preference_dataset(records: list[dict], out_dir: str, name: str = "dataos_dpo") -> int:
    return _write_dataset(
        records,
        out_dir,
        name,
        {
            "file_name": "train.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
                "history": "history",
            },
        },
    )


def write_kto_dataset(records: list[dict], out_dir: str, name: str = "dataos_kto") -> int:
    return _write_dataset(
        records,
        out_dir,
        name,
        {
            "file_name": "train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "kto_tag": "kto_tag",
                "system": "system",
                "history": "history",
            },
        },
    )


def write_ppo_dataset(records: list[dict], out_dir: str, name: str = "dataos_ppo") -> int:
    return _write_dataset(
        records,
        out_dir,
        name,
        {
            "file_name": "train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos_train.data.convert")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--informat", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--to", default="alpaca", choices=["alpaca", "preference", "kto", "ppo"])
    ap.add_argument("--name", default="dataos_sft")
    args = ap.parse_args(argv)
    records = read_records(args.inp, args.informat)
    if args.to == "preference":
        converted = to_preference(records, args.informat)
        n = write_preference_dataset(converted, args.out, args.name)
    elif args.to == "kto":
        converted = to_kto(records, args.informat)
        n = write_kto_dataset(converted, args.out, args.name)
    elif args.to == "ppo":
        converted = to_ppo(records, args.informat)
        n = write_ppo_dataset(converted, args.out, args.name)
    else:
        converted = to_alpaca(records, args.informat)
        n = write_alpaca_dataset(converted, args.out, args.name)
    print(json.dumps({"rows": n, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
