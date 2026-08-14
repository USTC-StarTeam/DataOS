"""The data stage's output contract, checked where the data is.

Runs INSIDE the data job, as its last step — the same place and for the same reason
``dataos_train.data.convert`` runs inside the training job: the thing being checked is on that
host, in that venv, and a failure has to fail THAT job.

What it checks is what the 2026-08-14 batch actually got wrong, which is not what one would guess.
A recipe that filters everything away already fails loudly, because ``save_to_disk`` refuses to
write a dataset with no rows to infer a schema from; humaneval's ``length_filter(40..2600)`` took
four sources to zero and the build job died on the spot. There is deliberately no rows>0 check
here: the state is unreachable, and a branch no test can construct is worse than the guarantee
``save_to_disk`` already gives.

The expensive half is content the next stage cannot consume. That same node's NEXT build
completed cleanly, and the training job died 52 seconds later on::

    ValueError: The number of images does not match the number of <image> tokens

Gemma 3 reserves ``<image>`` and ``<video>`` as multimodal placeholders, so a source row that
happens to contain the literal string breaks the chat template — and the batch before this one
lost an arenahardwriting node to the ``<video>`` half of the same thing. Scanning a sample for
those strings here turns "52 seconds into training, wearing a training error" into "the build
refused the data".

The manifest is written here rather than by the recipe because the recipe kept inventing it: the
same batch produced ``data_stats.json`` in one run root and ``data_manifest.json`` in another, for
the same thing. The per-source funnel a recipe records (``in -> after_dedup -> after_length ->
capped``) is genuinely worth having, so it is merged in rather than replaced — this fixes the
name and the load-bearing keys, and leaves the recipe free to say more.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST = "data_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _alpaca_rows(sft_dir: Path) -> tuple[int, str] | None:
    """``(rows, sha256)`` of the alpaca ``train.json``, or None when it is absent."""
    train_json = sft_dir / "train.json"
    if not train_json.is_file():
        return None
    try:
        rows = json.loads(train_json.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"dataos-data: {train_json} is not readable JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise SystemExit(f"dataos-data: {train_json} must be a JSON array, got {type(rows).__name__}")
    return len(rows), _sha256(train_json)


def _hf_rows(sft_dir: Path) -> int:
    """Rows in the HF ``save_to_disk`` form — the one ``dataos_train.data.convert`` reads."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - the exec venv always has it
        raise SystemExit(f"dataos-data: `datasets` is not installed in this venv: {exc}") from exc
    try:
        return len(load_from_disk(str(sft_dir)))
    except Exception as exc:  # noqa: BLE001 — any failure here means the train build will reject it
        raise SystemExit(
            f"dataos-data: {sft_dir} is not a Hugging Face save_to_disk dataset ({exc}). "
            "The training job's converter calls load_from_disk on it, so a recipe that writes "
            "only train.json fails at train time instead of here."
        ) from exc


#: Strings Gemma 3 reserves for multimodal content. A row carrying one as ordinary text makes the
#: chat template demand an image or a video that does not exist, and the error names neither the
#: row nor the recipe — it arrives as a training failure, tens of seconds in.
_RESERVED_TOKENS = ("<image>", "<video>")


def _template_check(rows: list[dict], template: str | None, sample: int) -> str | None:
    """Scan a sample for text the target chat template cannot take literally. First failure, or None.

    A scan, deliberately, and not a render through LLaMA-Factory. Rendering would be the general
    check, and LF is importable exactly where this runs — but it is not importable in a dev
    checkout, so that version could not be tested here, and an untested general check is worth
    less than a tested specific one. What this covers is the failure that has actually happened,
    twice: the ``<image>`` that killed a humaneval training job 52 seconds in, and the ``<video>``
    that cost an arenahardwriting node the batch before.

    Widen ``_RESERVED_TOKENS`` when a third one shows up; swap in a real render when there is a
    venv here that can exercise it.
    """
    if not template:
        return None
    for i, row in enumerate(rows[:sample]):
        text = " ".join(str(row.get(k, "")) for k in ("instruction", "input", "output"))
        for tok in _RESERVED_TOKENS:
            if tok in text:
                return (
                    f"row {i} contains the reserved token {tok!r}. Gemma 3 treats it as a "
                    f"multimodal placeholder, so the chat template will demand content that is "
                    f"not there — strip or escape it in the recipe"
                )
    return None


def verify(
    sft_dir: Path,
    *,
    stats: dict[str, Any] | None = None,
    template: str | None = None,
    sample: int = 200,
) -> dict[str, Any]:
    """Check the contract and return the manifest. Raises SystemExit on any violation."""
    if not sft_dir.is_dir():
        raise SystemExit(f"dataos-data: no data dir at {sft_dir}")

    hf_rows = _hf_rows(sft_dir)

    alpaca = _alpaca_rows(sft_dir)
    if alpaca is None:
        raise SystemExit(
            f"dataos-data: {sft_dir}/train.json is missing. The contract is BOTH forms: the HF "
            "dataset for the converter and the alpaca train.json for LLaMA-Factory."
        )
    alpaca_rows, fingerprint = alpaca
    if alpaca_rows != hf_rows:
        raise SystemExit(
            f"dataos-data: the two forms disagree — {hf_rows} rows in the HF dataset, "
            f"{alpaca_rows} in train.json. They are supposed to be the same rows."
        )

    problem = _template_check(
        json.loads((sft_dir / "train.json").read_text(encoding="utf-8")), template, sample
    )
    if problem:
        raise SystemExit(f"dataos-data: {problem}")

    manifest: dict[str, Any] = {
        "rows": hf_rows,
        "fingerprint": fingerprint,
        "path": str(sft_dir),
    }
    if stats:
        # The recipe's own funnel, kept verbatim under one key so it cannot collide with the
        # fields above and cannot be renamed away.
        manifest["recipe_stats"] = stats
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos-data verify")
    ap.add_argument("--dir", required=True, help="the data/sft dir the recipe wrote")
    ap.add_argument("--stats", default=None,
                    help="optional JSON file of the recipe's own per-source counts; merged in")
    ap.add_argument("--out", default=None, help=f"where to write the manifest (default: <dir>/../{MANIFEST})")
    ap.add_argument("--template", default=None,
                    help="chat template the training job will use; enables the reserved-token scan")
    ap.add_argument("--sample", type=int, default=200, help="rows to render for the template check")
    args = ap.parse_args(argv)

    sft_dir = Path(args.dir).resolve()
    stats = None
    if args.stats and Path(args.stats).is_file():
        try:
            stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SystemExit(f"dataos-data: --stats {args.stats} is not readable JSON: {exc}") from exc

    manifest = verify(sft_dir, stats=stats, template=args.template, sample=args.sample)
    out = Path(args.out) if args.out else sft_dir.parent.parent / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "rows": manifest["rows"], "manifest": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
