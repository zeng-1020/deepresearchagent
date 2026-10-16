import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepresearch.agent import run_with_model
from deepresearch.checkpointing import open_sqlite_checkpointer
from deepresearch.export import (
    ExportedRun,
    build_markdown,
    export_runs,
    render_index,
    slugify,
)
from deepresearch.state import ResearchState


def make_state() -> ResearchState:
    return {
        "messages": [],
        "research_question": "workflow 和 agent 工作流有什么区别",
        "plan": None,
        "current_step_id": None,
        "reflection_attempts": 0,
        "research_gaps": [],
        "search_records": [
            {
                "query": "failed query",
                "success": False,
                "result_count": 0,
                "error": "Search failed: No results found.",
            }
        ],
        "page_records": [],
        "sources": [
            {
                "title": "Official documentation",
                "url": "https://example.com/docs",
                "snippet": "Example evidence",
                "query": "successful query",
            }
        ],
        "observations": [],
        "final_report": None,
    }


def seed_database(database_path, thread_id: str, question: str) -> None:  # noqa: ANN001
    """Persist one completed run into a real checkpoint database."""

    with open_sqlite_checkpointer(database_path) as checkpointer:
        run_with_model(
            question,
            FakeMessagesListChatModel(responses=[AIMessage(content="答案")]),
            checkpointer=checkpointer,
            thread_id=thread_id,
        )


def test_build_markdown_keeps_report_verbatim() -> None:
    state = make_state()
    state["final_report"] = "# 报告标题\n\n正文段落"

    markdown = build_markdown(
        "research-001",
        state,
        "2026-10-12T12:00:00+00:00",
        exported_at="2026-10-12 20:00:00",
    )

    assert markdown.startswith("# workflow 和 agent 工作流有什么区别")
    assert "# 报告标题\n\n正文段落" in markdown
    assert "- 状态：已生成报告" in markdown
    assert "## 运行统计" in markdown
    assert "https://example.com/docs" in markdown


def test_build_markdown_explains_missing_report() -> None:
    state = make_state()
    state["governance"] = {
        "context": {
            "finalization": {
                "trigger_reason": "consecutive_unproductive_searches",
                "last_blocked_tool_names": ["web_search"],
            }
        }
    }

    markdown = build_markdown(
        "research-002",
        state,
        "2026-10-12T12:00:00+00:00",
        exported_at="2026-10-12 20:00:00",
    )

    assert "- 状态：未生成报告" in markdown
    assert "## 为什么没有生成报告" in markdown
    assert "consecutive_unproductive_searches" in markdown
    assert "web_search" in markdown
    assert "failed query" in markdown
    assert "Search failed: No results found." in markdown


def test_build_markdown_can_append_full_trace() -> None:
    state = make_state()

    without_trace = build_markdown(
        "research-003", state, "未知", exported_at="2026-10-12 20:00:00"
    )
    with_trace = build_markdown(
        "research-003",
        state,
        "未知",
        exported_at="2026-10-12 20:00:00",
        with_trace=True,
    )

    assert "完整执行轨迹" not in without_trace
    assert "完整执行轨迹" in with_trace
    assert "搜索记录：" in with_trace


def test_build_markdown_falls_back_when_question_is_missing() -> None:
    state = make_state()
    state.pop("research_question")

    markdown = build_markdown(
        "research-004", state, "未知", exported_at="2026-10-12 20:00:00"
    )

    assert markdown.startswith("# （未记录研究问题）")


def test_slugify_removes_unsafe_filename_characters() -> None:
    assert slugify("a/b:c*d?e") == "a-b-c-d-e"
    assert slugify("  多  空格\n换行  ") == "多-空格-换行"
    assert slugify("///") == "no-question"
    assert len(slugify("字" * 100)) == 40


def test_render_index_lists_runs_in_given_order() -> None:
    runs = [
        ExportedRun(
            thread_id="research-001",
            question="第一个问题",
            filename="research-001-a.md",
            has_report=True,
            searches=5,
            page_reads=4,
            sources=12,
        ),
        ExportedRun(
            thread_id="research-002",
            question="第二个问题",
            filename="research-002-b.md",
            has_report=False,
            searches=4,
            page_reads=0,
            sources=0,
        ),
    ]

    index = render_index(runs, exported_at="2026-10-12 20:00:00")

    assert "任务数量：2" in index
    assert index.index("第一个问题") < index.index("第二个问题")
    assert "[research-001-a.md](research-001-a.md)" in index
    assert "| 已生成 |" in index
    assert "| 未生成 |" in index


def test_export_writes_one_file_per_thread_and_an_index(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"
    seed_database(database_path, "research-001", "第一个问题")
    seed_database(database_path, "research-002", "第二个问题")
    output_dir = tmp_path / "exports"

    outcome = export_runs(database_path=database_path, output_dir=output_dir)

    assert [run.thread_id for run in outcome.runs] == ["research-001", "research-002"]
    assert len(outcome.written) == 2
    assert outcome.skipped == []
    first = output_dir / outcome.runs[0].filename
    assert first.exists()
    assert "第一个问题" in first.read_text(encoding="utf-8")
    assert "任务数量：2" in (output_dir / "INDEX.md").read_text(encoding="utf-8")


def test_export_skips_existing_files_until_forced(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"
    seed_database(database_path, "research-001", "第一个问题")
    output_dir = tmp_path / "exports"
    first = export_runs(database_path=database_path, output_dir=output_dir)
    target = output_dir / first.runs[0].filename
    target.write_text("手动修改过的内容", encoding="utf-8")

    second = export_runs(database_path=database_path, output_dir=output_dir)
    assert second.skipped == [first.runs[0].filename]
    assert target.read_text(encoding="utf-8") == "手动修改过的内容"

    third = export_runs(
        database_path=database_path, output_dir=output_dir, overwrite=True
    )
    assert third.written == [first.runs[0].filename]
    assert "第一个问题" in target.read_text(encoding="utf-8")


def test_export_dry_run_writes_nothing(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"
    seed_database(database_path, "research-001", "第一个问题")
    output_dir = tmp_path / "exports"

    outcome = export_runs(
        database_path=database_path, output_dir=output_dir, dry_run=True
    )

    assert len(outcome.runs) == 1
    assert outcome.written == []
    assert outcome.index_path is None
    assert not output_dir.exists()
    assert "第一个问题" in outcome.index_markdown


def test_export_selects_threads_and_reports_unknown_ids(tmp_path) -> None:  # noqa: ANN001
    database_path = tmp_path / "checkpoints.sqlite"
    seed_database(database_path, "research-001", "第一个问题")
    seed_database(database_path, "research-002", "第二个问题")

    outcome = export_runs(
        database_path=database_path,
        output_dir=tmp_path / "exports",
        thread_ids=["research-002", "research-missing"],
    )

    assert [run.thread_id for run in outcome.runs] == ["research-002"]
    assert outcome.missing == ["research-missing"]


def test_export_rejects_missing_database(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="找不到 Checkpoint 数据库"):
        export_runs(
            database_path=tmp_path / "missing.sqlite",
            output_dir=tmp_path / "exports",
        )
