"""Export persisted research runs into readable Markdown files.

The Web UI keeps a run's timeline in memory only, so once the server stops the
checkpoint database is the single source of truth for what was asked and what
came back. This module reads that database, calls no model, and writes one
Markdown file per run plus an index.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from deepresearch.checkpointing import (
    DEFAULT_CHECKPOINT_DB,
    get_checkpoint_tuple,
    open_sqlite_checkpointer,
)
from deepresearch.reporting import (
    calculate_stats,
    format_evaluation_markdown,
    format_trace,
)
from deepresearch.state import ResearchState


DEFAULT_EXPORT_DIR = Path("reports/exports")
INDEX_FILENAME = "INDEX.md"
SLUG_LIMIT = 40
_QUESTION_PREVIEW_LIMIT = 36
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExportedRun:
    """The facts about one run needed to build the index table."""

    thread_id: str
    question: str
    filename: str
    has_report: bool
    searches: int
    page_reads: int
    sources: int


@dataclass(frozen=True)
class ExportOutcome:
    """What one export invocation read, wrote, and had to skip."""

    export_dir: Path
    index_path: Path | None
    index_markdown: str
    runs: list[ExportedRun]
    written: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]
    missing: list[str]


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(text: str, limit: int = SLUG_LIMIT) -> str:
    """Turn a research question into a short, filename-safe fragment."""

    collapsed = _WHITESPACE.sub(" ", text).strip()
    cleaned = _UNSAFE_FILENAME_CHARS.sub(" ", collapsed)
    cleaned = _WHITESPACE.sub("-", cleaned).strip("-. ")
    if not cleaned:
        return "no-question"
    return cleaned[:limit].rstrip("-. ")


def list_thread_ids(database_path: str | Path) -> list[str]:
    """Return every thread in insertion order without modifying the database."""

    database = Path(database_path)
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT thread_id
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY MAX(rowid)
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def _stats_lines(state: dict[str, Any]) -> list[str]:
    stats = calculate_stats(cast(ResearchState, state))
    return [
        f"- 搜索次数：{stats.total_searches}",
        f"- 成功并返回结果：{stats.successful_searches}",
        f"- 成功但无结果：{stats.empty_searches}",
        f"- 执行失败：{stats.failed_searches}",
        f"- 去重后来源：{stats.unique_sources}",
        f"- Observation 数量：{stats.observation_count}",
        (
            f"- 网页读取：{stats.total_page_reads}"
            f"（成功 {stats.successful_page_reads} / 失败 {stats.failed_page_reads}）"
        ),
    ]


def _missing_report_section(state: dict[str, Any]) -> list[str]:
    """Explain a run that never produced a report using recorded facts only."""

    stats = calculate_stats(cast(ResearchState, state))
    finalization = _mapping(
        _mapping(_mapping(state.get("governance")).get("context")).get("finalization")
    )
    trigger_reason = finalization.get("trigger_reason")
    blocked_names = finalization.get("last_blocked_tool_names")

    lines = ["## 为什么没有生成报告", "", "这次运行没有产出正式报告，程序记录到的原因如下：", ""]
    if isinstance(trigger_reason, str) and trigger_reason:
        lines.append(f"- 收尾触发原因：{trigger_reason}")
    if isinstance(blocked_names, list) and blocked_names:
        lines.append(
            "- 最近被拦截的 Tool：" + ", ".join(str(name) for name in blocked_names)
        )
    lines.extend(
        [
            f"- 搜索：{stats.total_searches} 次"
            f"（返回结果 {stats.successful_searches} 次 / "
            f"无结果 {stats.empty_searches} 次 / "
            f"失败 {stats.failed_searches} 次）",
            f"- 网页读取成功：{stats.successful_page_reads} 次",
            f"- 去重来源：{stats.unique_sources} 个",
            f"- Observation：{stats.observation_count} 条",
        ]
    )

    failed = [
        record
        for record in state.get("search_records", [])
        if isinstance(record, dict) and not record.get("success")
    ]
    if failed:
        lines.extend(["", "搜索失败明细：", ""])
        for index, record in enumerate(failed, start=1):
            lines.append(f"{index}. {record.get('query') or '(空)'}")
            lines.append(f"   - 错误：{record.get('error') or '未记录'}")
    return lines


def _sources_section(state: dict[str, Any]) -> list[str]:
    sources = state.get("sources") or []
    lines = ["## 来源", ""]
    if not sources:
        lines.append("（无）")
        return lines
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        lines.append(f"{index}. {source.get('title') or '（无标题）'}")
        lines.append(f"   - {source.get('url') or ''}")
        lines.append(f"   - 搜索词：{source.get('query') or ''}")
    return lines


def build_markdown(
    thread_id: str,
    state: dict[str, Any],
    created_at: str,
    *,
    exported_at: str,
    with_trace: bool = False,
) -> str:
    """Render one run as a self-contained Markdown document."""

    question = state.get("research_question")
    if not isinstance(question, str) or not question.strip():
        question = "（未记录研究问题）"
    question = question.strip()

    report = state.get("final_report")
    report_text = str(report).strip() if report else ""

    lines = [
        f"# {question}",
        "",
        f"- 任务 ID：`{thread_id}`",
        f"- Checkpoint 时间：{created_at}",
        f"- 导出时间：{exported_at}",
        f"- 状态：{'已生成报告' if report_text else '未生成报告'}",
        "",
    ]
    # 报告正文原样保留（不重排标题层级），便于直接复制成独立文档。
    if report_text:
        lines.extend(["---", "", report_text, ""])
    else:
        lines.extend(_missing_report_section(state))
    lines.extend(["---", "", "## 运行统计", "", *_stats_lines(state)])
    lines.extend(["", *_sources_section(state)])

    evaluation_markdown = format_evaluation_markdown(state)
    if evaluation_markdown:
        lines.extend(["", evaluation_markdown])

    if with_trace:
        lines.extend(["", "---", "", "## 程序附录：完整执行轨迹", "", "```text"])
        lines.extend(format_trace(cast(ResearchState, state)).splitlines())
        lines.append("```")

    lines.append("")
    return "\n".join(lines)


def render_index(runs: Sequence[ExportedRun], *, exported_at: str) -> str:
    """Render the entry point listing every exported run in execution order."""

    lines = [
        "# 研究任务导出索引",
        "",
        f"导出时间：{exported_at}",
        f"任务数量：{len(runs)}",
        "",
        "| # | 问题 | 报告 | 搜索 | 网页读取 | 来源 | 文件 |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, run in enumerate(runs, start=1):
        question = run.question.replace("|", "\\|").replace("\n", " ")
        if len(question) > _QUESTION_PREVIEW_LIMIT:
            question = question[:_QUESTION_PREVIEW_LIMIT] + "…"
        lines.append(
            f"| {index} | {question} | {'已生成' if run.has_report else '未生成'} "
            f"| {run.searches} | {run.page_reads} | {run.sources} "
            f"| [{run.filename}]({run.filename}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_run(checkpointer: Any, thread_id: str) -> tuple[dict[str, Any], str]:
    checkpoint = get_checkpoint_tuple(checkpointer, thread_id)
    values = checkpoint.checkpoint.get("channel_values", {})
    state = dict(values) if isinstance(values, dict) else {}
    created_at = str(checkpoint.checkpoint.get("ts") or "未知")
    return state, created_at


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def export_runs(
    *,
    database_path: str | Path = DEFAULT_CHECKPOINT_DB,
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    thread_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    with_trace: bool = False,
    dry_run: bool = False,
) -> ExportOutcome:
    """Read the checkpoint database and write one Markdown file per run.

    Existing files are skipped unless ``overwrite`` is set, so re-running the
    command after a new research task never destroys earlier exports.
    """

    database = Path(database_path)
    if not database.exists():
        raise ValueError(
            f"找不到 Checkpoint 数据库：{database}\n"
            "请在项目根目录下运行，或用 --db 指定实际路径。"
        )

    available = list_thread_ids(database)
    missing: list[str] = []
    if thread_ids is None:
        selected = available
    else:
        wanted = [str(thread_id) for thread_id in thread_ids]
        wanted_set = set(wanted)
        selected = [thread_id for thread_id in available if thread_id in wanted_set]
        missing = sorted(wanted_set - set(selected))

    exported_at = _now_text()
    destination = Path(output_dir)
    runs: list[ExportedRun] = []
    written: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    with open_sqlite_checkpointer(database) as checkpointer:
        for thread_id in selected:
            try:
                state, created_at = _read_run(checkpointer, thread_id)
                stats = calculate_stats(cast(ResearchState, state))
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                failed.append((thread_id, str(exc)))
                continue

            question = state.get("research_question")
            question = question.strip() if isinstance(question, str) else ""
            filename = f"{thread_id}-{slugify(question)}.md"
            runs.append(
                ExportedRun(
                    thread_id=thread_id,
                    question=question or "（未记录研究问题）",
                    filename=filename,
                    has_report=bool(state.get("final_report")),
                    searches=stats.total_searches,
                    page_reads=stats.total_page_reads,
                    sources=stats.unique_sources,
                )
            )
            if dry_run:
                continue

            target = destination / filename
            if target.exists() and not overwrite:
                skipped.append(filename)
                continue
            _write_text(
                target,
                build_markdown(
                    thread_id,
                    state,
                    created_at,
                    exported_at=exported_at,
                    with_trace=with_trace,
                ),
            )
            written.append(filename)

    index_markdown = render_index(runs, exported_at=exported_at)
    index_path: Path | None = None
    if not dry_run:
        index_path = destination / INDEX_FILENAME
        _write_text(index_path, index_markdown)

    return ExportOutcome(
        export_dir=destination,
        index_path=index_path,
        index_markdown=index_markdown,
        runs=runs,
        written=written,
        skipped=skipped,
        failed=failed,
        missing=missing,
    )
