from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import ensure_dataset_dict, find_latest_dataset, restore_dataset_type


_TEMPLATE_PREFIXES = {
    "brief_answer": "Answer briefly: ",
    "answer_only": "Return only the final answer for: ",
    "step_by_step": "Solve step by step: ",
}


class InstructionSynthesisTemplateOperator(Operator):
    name = "instruction_synthesis_template"
    description = "Expand a training split with deterministic instruction rewrites while preserving the original answers."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        split_name = str(kwargs.get("split", "train"))
        templates = list(kwargs.get("templates") or ["brief_answer", "answer_only"])
        max_new_rows = kwargs.get("max_new_rows")
        preserve_original = bool(kwargs.get("preserve_original", True))

        for template_name in templates:
            if template_name not in _TEMPLATE_PREFIXES:
                raise ValueError(f"instruction_synthesis_template: unknown template '{template_name}'")
        if split_name not in dataset_dict:
            raise ValueError(f"instruction_synthesis_template: split '{split_name}' not found")
        split_dataset = dataset_dict[split_name]
        if "instruction" not in split_dataset.column_names or "response" not in split_dataset.column_names:
            raise ValueError("instruction_synthesis_template: requires instruction and response columns")

        synthesized_rows: list[dict] = []
        original_rows = [dict(row) for row in split_dataset]
        for index, row in enumerate(original_rows):
            for template_name in templates:
                new_row = dict(row)
                new_row["instruction"] = _TEMPLATE_PREFIXES[template_name] + str(row.get("instruction", "")).strip()
                new_row["dataos_synth_template"] = template_name
                new_row["dataos_synth_source_index"] = index
                synthesized_rows.append(new_row)
                if max_new_rows is not None and len(synthesized_rows) >= int(max_new_rows):
                    break
            if max_new_rows is not None and len(synthesized_rows) >= int(max_new_rows):
                break

        final_rows = []
        if preserve_original:
            for row in original_rows:
                base_row = dict(row)
                base_row.setdefault("dataos_synth_template", "original")
                base_row.setdefault("dataos_synth_source_index", -1)
                final_rows.append(base_row)
        final_rows.extend(synthesized_rows)
        transformed = dict(dataset_dict)
        transformed[split_name] = Dataset.from_list(final_rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-synth-template",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "instruction_synthesis_split": split_name,
                "instruction_synthesis_templates": templates,
                "instruction_synthesis_added_rows": len(synthesized_rows),
                "instruction_synthesis_preserve_original": preserve_original,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nsplit={split_name}\nadded_rows={len(synthesized_rows)}\n",
        )
        return artifact
