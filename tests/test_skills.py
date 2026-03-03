from __future__ import annotations

from pathlib import Path

from py_agent.skills import apply_skill_invocations, build_skills_system_prompt, discover_skills


def _write_skill(root: Path, name: str, content: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_discover_skills_from_workspace(test_workspace: Path) -> None:
    skills_root = test_workspace / "skills"
    _write_skill(
        skills_root,
        "writer",
        "# writer\nWrite concise markdown.\n\nAlways include headings.",
    )
    _write_skill(
        skills_root,
        "reviewer",
        "# reviewer\nFocus on bugs and regressions.",
    )

    skills = discover_skills(test_workspace)
    assert set(skills.keys()) == {"writer", "reviewer"}
    assert "Write concise markdown." in skills["writer"].content


def test_apply_skill_invocations_expands_requested_skills(test_workspace: Path) -> None:
    skills_root = test_workspace / "skills"
    _write_skill(skills_root, "writer", "# writer\nWrite concise markdown.")
    _write_skill(skills_root, "reviewer", "# reviewer\nFocus on bugs.")
    skills = discover_skills(test_workspace)

    prompt = "/skill:writer /skill:reviewer create release notes"
    expanded = apply_skill_invocations(prompt, skills)

    assert "/skill:writer" not in expanded
    assert "/skill:reviewer" not in expanded
    assert "create release notes" in expanded
    assert "Write concise markdown." in expanded
    assert "Focus on bugs." in expanded


def test_build_skills_system_prompt_contains_available_skills(test_workspace: Path) -> None:
    skills_root = test_workspace / "skills"
    _write_skill(skills_root, "writer", "# writer\nWrite concise markdown.")
    _write_skill(skills_root, "reviewer", "# reviewer\nFocus on bugs.")
    skills = discover_skills(test_workspace)

    system_prompt = build_skills_system_prompt(skills)
    assert "Available skills" in system_prompt
    assert "writer" in system_prompt
    assert "reviewer" in system_prompt


def test_apply_skill_invocations_unknown_skill_keeps_task_text(test_workspace: Path) -> None:
    skills = discover_skills(test_workspace)
    expanded = apply_skill_invocations("/skill:missing implement feature x", skills)
    assert "/skill:missing" not in expanded
    assert "implement feature x" in expanded
