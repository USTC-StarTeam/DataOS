"""``llamafactory-cli`` with dataos-train's extra chat templates registered first.

The LF backend renders this as the train/export command instead of ``llamafactory-cli`` when the
spec asks for a template from :data:`~.templates.DATAOS_TEMPLATES`; everything else keeps using
LF's own console script, so this path only carries runs that need it.

    python -m dataos_train.backends.llamafactory.ext.lf_entry train <config.yaml>
    python -m dataos_train.backends.llamafactory.ext.lf_entry export --template smollm3 ...
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    from .templates import register_all

    register_all()

    # LF's launcher re-execs through torchrun for FORCE_TORCHRUN or >1 visible device, and the
    # child imports llamafactory directly — our registration would NOT survive it, and the run
    # would fail with "template smollm3 does not exist" or, worse, fall back silently. Fail here
    # with the reason instead. Single-GPU (every PTB run today) stays in-process.
    if args and args[0] == "train":
        from llamafactory.extras.misc import get_device_count

        if os.environ.get("FORCE_TORCHRUN", "").lower() in {"1", "true"} or get_device_count() > 1:
            raise RuntimeError(
                "dataos-train's extra chat templates are registered in-process, but LF is about "
                "to re-exec through torchrun (FORCE_TORCHRUN set or >1 visible GPU), where the "
                "registration is lost. Run this template single-GPU, or teach LF the template "
                "at package level before using it distributed."
            )

    sys.argv = ["llamafactory-cli", *args]
    from llamafactory.cli import main as lf_main

    lf_main()


if __name__ == "__main__":
    main()
