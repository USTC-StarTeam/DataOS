"""bfcl must send each doc ITS OWN functions, at any concurrency.

The defect this pins (2026-08-11): the task put per-doc ``tools`` into ``gen_kwargs``, and
``gen_kwargs`` is shared by a whole batch. Three layers of lm-eval then conspire:

1. requests are grouped by gen_kwargs; the key builder makes the dict hashable with
   ``tuple(value) if isinstance(value, Iterable)``, which fails for a list of dicts, and the
   ``except (TypeError, AttributeError)`` fallback keys by ``tuple(fn(ob))`` — **the field names
   only, values discarded**. Every doc lands in one group.
2. with ``num_concurrent > 1`` the chunk size is ``n=0``, i.e. the whole group is one chunk.
3. the chunk is sent with ``gen_kwargs=copy.deepcopy(all_gen_kwargs[0])``.

So doc 2..N were asked to call doc 1's functions. bfcl scored 0–16 across two whole batches
(smol3 and 4b2 v5, 23 nodes) while the same bench reached 60–92 elsewhere at concurrency 1 —
the one setting under which a chunk holds a single doc and ``all_gen_kwargs[0]`` is its own.

The fix carries ``tools`` in the CONTEXT, which is per request. This test reproduces step 3
directly: it builds doc 2's payload with doc 1's gen_kwargs and asserts doc 2 still gets its own
functions. It also asserts the gen_kwargs are equal across docs, because that equality is what
keeps them in a single group — i.e. what preserves concurrency instead of trading it for
correctness.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "configs" / "tasks" / "posttrainbench" / "eval"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bfcl_parts():
    # runs in eval_env, where the scorer itself runs; sft_env has lm-eval but not tenacity
    for dep in ("lm_eval", "aiohttp", "tenacity"):
        pytest.importorskip(dep, reason=f"the eval task package needs {dep}")
    task = _load("ptb_task_under_test", EVAL / "lmeval" / "task.py")
    model = _load("ptb_model_under_test", EVAL / "models" / "openai_tool_chat.py")
    return task, model


def _doc(name: str, question: str) -> dict:
    return {
        "id": name,
        "question": [[{"role": "user", "content": question}]],
        "function": [
            {
                "name": name,
                "description": f"call {name}",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "description": "arg"}},
                    "required": ["x"],
                },
            }
        ],
        "ground_truth": [f"{name}(x='1')"],
    }


def _instances(task_module, docs):
    """Two requests as lm-eval would receive them, without instantiating the task class."""
    cls = task_module.PostTrainBenchBFCL
    stub = object.__new__(cls)
    # ConfigurableTask exposes `config` as a read-only property over `_config`
    stub._config = type("C", (), {"generation_kwargs": None})()
    return [cls.construct_requests(stub, doc, None) for doc in docs]


def _payload(model_module, context, shared_gen_kwargs: dict) -> dict:
    """The request body for ``context``, built with the batch's SHARED gen_kwargs.

    This is the exact call the batched path makes: contexts are per request, gen_kwargs is
    ``all_gen_kwargs[0]`` for the whole chunk.
    """
    cls = model_module.DataOSLocalToolChatCompletion
    stub = object.__new__(cls)
    stub.model = "m"
    stub._max_gen_toks = 256
    return cls._create_payload(
        stub,
        json.loads(context.prompt),  # what create_message() hands to _create_payload
        generate=True,
        gen_kwargs=dict(shared_gen_kwargs),
        seed=1234,
        eos=None,
    )


def test_each_doc_gets_its_own_tools_under_a_shared_gen_kwargs(bfcl_parts) -> None:
    task_module, model_module = bfcl_parts
    docs = [_doc("alpha", "do alpha"), _doc("beta", "do beta")]
    first, second = _instances(task_module, docs)

    shared = first.arguments[1]  # all_gen_kwargs[0] — what the whole chunk is sent with
    for name, instance in (("alpha", first), ("beta", second)):
        payload = _payload(model_module, instance.arguments[0], shared)
        got = [t["function"]["name"] for t in payload["tools"]]
        assert got == [name], (
            f"doc {name!r} was sent tools {got} — a per-doc field is travelling in gen_kwargs "
            "again, so every doc in a batch gets the first doc's functions"
        )


def test_gen_kwargs_are_identical_across_docs(bfcl_parts) -> None:
    """Equal gen_kwargs keep all docs in ONE group, which is what keeps concurrency.

    Making the tools merely hashable would also fix correctness, but it would split the docs into
    one group each, and lm-eval runs ``asyncio.run`` once per chunk — so it would silently be as
    slow as num_concurrent=1 (bfcl: ~110s -> tens of minutes).
    """
    task_module, _ = bfcl_parts
    first, second = _instances(task_module, [_doc("alpha", "a"), _doc("beta", "b")])
    assert first.arguments[1] == second.arguments[1]
    assert "tools" not in first.arguments[1]


def test_other_benches_are_byte_for_byte_unchanged(bfcl_parts) -> None:
    """Only bfcl has per-doc body fields. Every other bench must still send a bare message list,
    or this fix would silently move five benches' request bytes and invalidate their baselines."""
    task_module, model_module = bfcl_parts
    instance = task_module._chat_instance(
        doc={"id": "x"},
        messages=[{"role": "user", "content": "hi"}],
        gen_kwargs={"max_gen_toks": 16},
    )
    context = json.loads(instance.arguments[0].prompt)
    assert isinstance(context, list), "a task with no per-doc fields must keep the plain list form"
    payload = _payload(model_module, instance.arguments[0], {"max_gen_toks": 16})
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload


def test_the_grouping_key_no_longer_falls_back(bfcl_parts) -> None:
    """The gen_kwargs must survive lm-eval's hashable-key construction.

    That construction is where the values were being thrown away; if it raises again, the
    fallback keys by field name only and this whole class of bug returns.
    """
    import collections.abc

    task_module, _ = bfcl_parts
    (first,) = _instances(task_module, [_doc("alpha", "a")])
    key = tuple(
        (k, tuple(v) if isinstance(v, collections.abc.Iterable) else v)
        for k, v in sorted(first.arguments[1].items())
    )
    hash(key)  # raises TypeError if any value is still an unhashable container
