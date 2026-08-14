"""LLaMA-Factory extension ("魔改") for online operators.

The backend-neutral protocol, registry, and reference ops live in
:mod:`dataos_train.online` (torch-free, unit-testable). This package is the LF-side glue:
the trainer patch (:mod:`trainer`) and launch entrypoint (:mod:`entry`) import torch +
llamafactory and run only inside the training job — they are NOT imported at package
import time, so ``import dataos_train`` stays dependency-light.
"""
