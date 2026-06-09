"""Offline integrity check for relative Markdown links across the repo.

For every committed ``*.md`` file we extract inline links ``[text](target)`` and assert
that each **relative** link (not ``http(s)``/``mailto`` and not a bare ``#anchor``) resolves
to a real path on disk. ``#fragment`` suffixes are stripped before the existence check; we do
not validate heading anchors (that would require a full Markdown renderer), only that the
target *file* exists.
"""

import re
from pathlib import Path

import pytest

# Repo root is two levels up from this file: tests/unit/ -> tests/ -> root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Markdown inline links: [text](target). Capture the target only.
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


# ``docs/plans/`` holds implementation plans: meta-documents full of fenced code blocks
# and *embedded* doc snippets whose relative paths resolve from the target doc's location,
# not the plan's. They are not user-facing cross-linked docs, so the naive inline-link regex
# would flag their code samples as broken; exclude the directory from the integrity check.
_EXCLUDED_DIRS = (".venv", "htmlcov", "plans", ".pytest_cache", ".git", "node_modules")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not any(part in _EXCLUDED_DIRS for part in path.parts)
    )


def _relative_link_targets(md_path: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8")
    targets: list[str] = []
    for raw in _LINK_RE.findall(text):
        target = raw.strip()
        # Skip external schemes and pure in-page anchors.
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target)
    return targets


@pytest.mark.parametrize("md_path", _markdown_files(), ids=lambda p: str(p))
def test_relative_links_resolve(md_path: Path) -> None:
    base = md_path.parent
    broken: list[str] = []
    for target in _relative_link_targets(md_path):
        # Drop any #fragment; we only check that the target file exists.
        path_part = target.split("#", 1)[0]
        if not path_part:  # link was just "#frag" on this file — already skipped above
            continue
        resolved = (base / path_part).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{md_path}: broken relative links: {broken}"


def test_some_markdown_files_were_scanned() -> None:
    # Guard against the glob silently matching nothing (e.g. a path bug).
    assert _markdown_files(), "no markdown files found to check"
