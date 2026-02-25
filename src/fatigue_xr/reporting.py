from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_markdown(path: Path, lines: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines
