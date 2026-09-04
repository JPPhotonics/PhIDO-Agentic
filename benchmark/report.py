"""Shared human-readable report writer for benchmark runs.

Standing convention: EVERY benchmark run driver must emit a Markdown report next to its
results JSON (``results/<name>.md``) so a run's outcome is readable without parsing raw JSON.
The raw JSON stays the machine-readable source of truth; the .md is the human view.

Usage::

    rep = Reporter(OUT.with_suffix(".md"), "Powered ablation N=20xK",
                   meta={"model": MODEL, "N": N, "K": K})
    rep.line("baseline_on: ...")            # prints to stdout AND records to the report
    rep.h("Per-stage rates")                # section header
    rep.table(["stage", "rate"], rows)      # Markdown table
    rep.save()                              # writes the .md (timestamp-stamped)

Route only summary/result content through the Reporter — leave noisy library output (e.g. the
Clingo validator's per-fact diagnostics) on plain ``print`` so it does not pollute the report.
"""

from __future__ import annotations

import datetime
import pathlib


class Reporter:
    def __init__(self, path, title: str, meta: dict | None = None):
        self.path = pathlib.Path(path)
        self._buf: list[str] = [f"# {title}", ""]
        when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self._buf.append(f"_Generated: {when}_")
        self._buf.append("")
        if meta:
            for k, v in meta.items():
                self._buf.append(f"- **{k}**: {v}")
            self._buf.append("")

    def line(self, text: str = "") -> None:
        """Print to stdout and record in the report."""
        print(text, flush=True)
        self._buf.append(text)

    def h(self, text: str) -> None:
        self.line("")
        self.line(f"## {text}")
        self.line("")

    def table(self, headers: list[str], rows: list) -> None:
        self.line("| " + " | ".join(str(h) for h in headers) + " |")
        self.line("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            self.line("| " + " | ".join(str(c) for c in r) + " |")

    def save(self) -> pathlib.Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._buf) + "\n")
        print(f"\n[report] -> {self.path}", flush=True)
        return self.path
