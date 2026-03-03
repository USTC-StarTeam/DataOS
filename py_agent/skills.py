"""Skill discovery and prompt expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SKILL_TOKEN_RE = re.compile(r"/skill:([a-zA-Z0-9_.-]+)")


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    content: str


def discover_skills(workspace: str | Path) -> dict[str, Skill]:
    root = Path(workspace)
    candidate_roots = [root / "skills", root / ".codex" / "skills"]
    skills: dict[str, Skill] = {}

    for base in candidate_roots:
        if not base.exists():
            continue
        for file in base.rglob("SKILL.md"):
            name = file.parent.name.strip()
            if not name or name in skills:
                continue
            content = file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            skills[name] = Skill(name=name, path=file.resolve(), content=content)
    return skills


def apply_skill_invocations(prompt: str, skills: dict[str, Skill]) -> str:
    requested = SKILL_TOKEN_RE.findall(prompt)
    if not requested:
        return prompt

    stripped = SKILL_TOKEN_RE.sub("", prompt)
    stripped = re.sub(r"\s+", " ", stripped).strip()

    selected_contents: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        skill = skills.get(name)
        if skill is not None:
            selected_contents.append(f"Skill '{name}' instructions:\n{skill.content}")

    if not selected_contents:
        return stripped
    return f"{stripped}\n\n" + "\n\n".join(selected_contents)


def build_skills_system_prompt(skills: dict[str, Skill]) -> str:
    if not skills:
        return ""
    names = sorted(skills.keys())
    lines = ["Available skills (invoke with /skill:<name>):"]
    for name in names:
        lines.append(f"- {name}")
    return "\n".join(lines)
