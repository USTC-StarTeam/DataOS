# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override

from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...dataos import DataOSController
    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


def _causal_per_sample_losses(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size(0), -1)
    valid_mask = shift_labels.ne(-100)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    return (token_losses * valid_mask).sum(dim=1) / valid_counts


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        dataos_controller: Optional["DataOSController"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.dataos_controller = dataos_controller

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.dataos_controller is not None and hasattr(self.dataos_controller, "sampler"):
            return self.dataos_controller.sampler

        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        dataos_ids = inputs.pop("dataos_id", None)
        if dataos_ids is None:
            inputs.pop("dataos_domain", None)
            inputs.pop("dataos_reference_loss", None)
        if self.dataos_controller is not None and model.training and "labels" in inputs and dataos_ids is not None:
            record_ids = [str(value) for value in dataos_ids]
            domain_values = inputs.pop("dataos_domain", None)
            if domain_values is None:
                domains = ["unknown" for _ in record_ids]
            else:
                domains = [str(value) if value is not None else "unknown" for value in domain_values]
            reference_values = inputs.pop("dataos_reference_loss", None)
            if reference_values is None:
                reference_losses = [None for _ in record_ids]
            else:
                reference_losses = list(reference_values)
            labels = inputs["labels"]
            outputs = model(**inputs)
            per_sample_losses = _causal_per_sample_losses(outputs.logits, labels)
            batch = self.dataos_controller.make_batch(
                record_ids=record_ids,
                domains=domains,
                tensors={"input_ids": inputs.get("input_ids"), "labels": labels},
            )
            self.dataos_controller.record_forward(
                batch=batch,
                per_sample_losses=per_sample_losses.detach().float().cpu().tolist(),
                reference_losses=[
                    float(value) if value is not None else None
                    for value in reference_losses
                ],
            )
            weights = torch.tensor(
                self.dataos_controller.batch_weights(
                    batch=batch,
                    per_sample_losses=per_sample_losses.detach().float().cpu().tolist(),
                ),
                device=per_sample_losses.device,
                dtype=per_sample_losses.dtype,
            )
            weighted_loss = torch.sum(per_sample_losses * weights)
            if kwargs.get("return_outputs", False):
                return weighted_loss, outputs
            return weighted_loss

        return super().compute_loss(model, inputs, *args, **kwargs)
