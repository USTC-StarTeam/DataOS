from __future__ import annotations

import os
import uuid

import pytest

from dataos_operator import Dataset


# 验证真实 HF 上传流程会在测试结束后删除临时 dataset repo。
def test_dataset_push_to_hf_then_delete() -> None:
    if os.environ.get("RUN_HF_INTEGRATION") != "1":
        pytest.skip("Set RUN_HF_INTEGRATION=1 to run Hugging Face upload test.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("Set HF_TOKEN to run Hugging Face upload test.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    namespace = api.whoami()["name"]
    repo_id = f"{namespace}/dataos-operator-ci-{uuid.uuid4().hex[:12]}"
    dataset = Dataset.from_records(
        [
            {
                "id": "hf-upload-smoke-00000",
                "text": "hello",
                "target": "5",
            }
        ]
    )

    try:
        dataset.push_to_hf(
            repo_id,
            split="train",
            config_name="default",
            token=token,
            private=True,
            commit_message="DataOS Operator CI upload smoke test",
        )
        info = api.dataset_info(repo_id, token=token)
        assert info.id == repo_id
    finally:
        api.delete_repo(
            repo_id,
            repo_type="dataset",
            token=token,
            missing_ok=True,
        )
