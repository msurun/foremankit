"""Preliminary code review by a second model: build the request, read the findings, decide which ones to act on.

The reviewer is a DIFFERENT source than the author (own local LLM or a paid model, never a free cloud; see
routing.yaml chains.review). It answers one line per problem: `SEVERITY | where | problem | evidence | fix`.
A finding only counts as grounded when its `evidence` quote really appears in the reviewed code: that filters out
the invented problems cheap models are prone to. Only grounded findings of the configured severities are sent back to the author.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SEVERITIES = ("blocker", "major", "minor")
NO_FINDINGS = "NO FINDINGS"
FIELD_SEPARATOR = "|"
EXPECTED_FIELDS = 5                       # severity, where, problem, evidence, fix
BULLET_PREFIX = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
WHITESPACE = re.compile(r"\s+")
# The evidence is quoted code and may itself contain `|`, so when it sits in backticks it is read as one piece.
QUOTED_EVIDENCE_LINE = re.compile(
    r"^(?P<severity>[^|]+)\|(?P<where>[^|]+)\|(?P<problem>[^|]+)\|\s*`(?P<evidence>.+)`\s*\|(?P<fix>.+)$")


@dataclass(frozen=True)
class Finding:
    """One problem the reviewer reported."""
    severity: str
    where: str
    problem: str
    evidence: str
    fix: str
    grounded: bool        # the evidence quote was found in the reviewed code

    def as_line(self) -> str:
        """The finding as one pipe-separated line (what the reviewer is asked to write)."""
        return f"{self.severity.upper()} | {self.where} | {self.problem} | {self.evidence} | {self.fix}"


def build_task(task: str, code: str, target: str, check_summary: str) -> str:
    """The text the reviewer gets: what was asked, the code, and what the automatic checks already established."""
    parts = [f"### What was asked\n{task.strip()}", f"### Code under review ({target or 'answer'})\n```\n{code.rstrip()}\n```"]
    if check_summary:
        parts.append(f"### Result of the automatic checks\n{check_summary.strip()}")
    parts.append("### Your job\nReview the code above as instructed. One line per problem, nothing else.")
    return "\n\n".join(parts)


def _normalise(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("`", "")).strip()


def _split_fields(line: str) -> list[str]:
    """severity, where, problem, evidence and fix of one reviewer line ([] when it is not a finding line)."""
    text = BULLET_PREFIX.sub("", line)
    quoted = QUOTED_EVIDENCE_LINE.match(text)
    if quoted:
        return [quoted[name].strip() for name in ("severity", "where", "problem", "evidence", "fix")]
    fields = [field.strip() for field in text.split(FIELD_SEPARATOR)]
    if len(fields) < EXPECTED_FIELDS:
        return []
    return fields[:4] + [f" {FIELD_SEPARATOR} ".join(fields[4:]).strip()]


def parse_findings(answer: str, code: str) -> list[Finding]:
    """The findings in the reviewer's answer. Lines that do not follow the format are ignored; an answer without
    any valid line (and without NO FINDINGS) simply yields an empty list, never an exception."""
    searchable_code = _normalise(code)
    findings = []
    for line in answer.splitlines():
        fields = _split_fields(line)
        severity = fields[0].strip("*` ").lower() if fields else ""
        if severity not in SEVERITIES:
            continue
        where, problem, evidence, fix = fields[1:]
        quote = _normalise(evidence)
        findings.append(Finding(severity, where, problem, evidence, fix, grounded=bool(quote) and quote in searchable_code))
    return findings


def to_fix(findings: list[Finding], severities: list[str]) -> list[Finding]:
    """The findings the author must act on: grounded, and of one of the configured severities."""
    return [finding for finding in findings if finding.grounded and finding.severity in severities]


def fix_request(findings: list[Finding], target: str) -> str:
    """The message that sends the findings back to the author."""
    lines = "\n".join(f"- {finding.as_line()}" for finding in findings)
    return (f"A reviewer found these problems in your code (SEVERITY | where | problem | evidence | suggested fix):\n\n{lines}\n\n"
            f"Fix them and return the complete corrected {target or 'code'} as one code block. Change nothing else.")
