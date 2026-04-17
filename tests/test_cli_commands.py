"""Tests for CLI commands: doctor and demo.

Uses Click's test runner (CliRunner) so no actual terminal interaction occurs.
No Anthropic API calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from causal_hypothesis_engine.cli import cli
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect DB and checkpoint dirs to a tmp directory."""
    import causal_hypothesis_engine.cli as cli_module
    monkeypatch.setattr(cli_module, "_DEFAULT_DB", tmp_path / "test.db")
    monkeypatch.setattr(
        cli_module, "_DEFAULT_CHECKPOINT_DIR", tmp_path / "checkpoints"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    def test_doctor_exits_1_without_api_key(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    def test_doctor_passes_with_api_key(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    def test_doctor_shows_python_version(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert "Python version" in result.output

    def test_doctor_shows_sqlite_info(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert "SQLite" in result.output
        assert "schema_version" in result.output

    def test_doctor_shows_wal_mode(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert "WAL" in result.output

    def test_doctor_shows_packages(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(cli, ["doctor"], env=env, catch_exceptions=False)
        assert "pydantic" in result.output
        assert "pandas" in result.output

    def test_doctor_bigquery_flag_checks_bigquery(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-test-key")
        result = runner.invoke(
            cli, ["doctor", "--bigquery"], env=env, catch_exceptions=False
        )
        assert "BigQuery" in result.output


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


class TestDemoCommand:
    def test_demo_exits_0(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        result = runner.invoke(cli, ["demo"], catch_exceptions=False)
        assert result.exit_code == 0

    def test_demo_creates_two_versions_in_db(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        import causal_hypothesis_engine.cli as cli_module
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        db = Database(cli_module._DEFAULT_DB)
        summaries = db.list_networks()
        assert len(summaries) == 1
        assert summaries[0]["version_count"] == 2

    def test_demo_output_contains_key_sections(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        result = runner.invoke(cli, ["demo"], catch_exceptions=False)
        assert "Demo Mode" in result.output
        assert "Version 1" in result.output
        assert "Version 2" in result.output
        assert "Diff" in result.output

    def test_demo_output_contains_next_steps(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        result = runner.invoke(cli, ["demo"], catch_exceptions=False)
        assert "causal-engine list" in result.output
        assert "causal-engine compare" in result.output

    def test_demo_version_has_nodes_and_edges(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        import causal_hypothesis_engine.cli as cli_module
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        db = Database(cli_module._DEFAULT_DB)
        networks = db.list_networks()
        network = db.get_network(networks[0]["id"])
        v1 = db.get_version(network.version_ids[0])
        assert len(v1.nodes) == 4
        assert len(v1.edges) == 3

    def test_demo_v2_is_child_of_v1(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        import causal_hypothesis_engine.cli as cli_module
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        db = Database(cli_module._DEFAULT_DB)
        networks = db.list_networks()
        network = db.get_network(networks[0]["id"])
        v2 = db.get_version(network.version_ids[1])
        assert v2.parent_version_id == network.version_ids[0]

    def test_demo_idempotent_second_run(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        """Running demo twice creates a second network, not corrupting the first."""
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        import causal_hypothesis_engine.cli as cli_module
        db = Database(cli_module._DEFAULT_DB)
        summaries = db.list_networks()
        assert len(summaries) == 2


# ---------------------------------------------------------------------------
# list command — smoke test
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_list_empty(self, runner: CliRunner, isolated_env: Path) -> None:
        result = runner.invoke(cli, ["list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No networks" in result.output

    def test_list_after_demo(self, runner: CliRunner, isolated_env: Path) -> None:
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        result = runner.invoke(cli, ["list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Insurance" in result.output


# ---------------------------------------------------------------------------
# compare command — smoke test (integration, no API)
# ---------------------------------------------------------------------------


class TestCompareCommand:
    def test_compare_two_versions(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        runner.invoke(cli, ["demo"], catch_exceptions=False)
        import causal_hypothesis_engine.cli as cli_module
        db = Database(cli_module._DEFAULT_DB)
        networks = db.list_networks()
        network = db.get_network(networks[0]["id"])
        v1_id = network.version_ids[0]
        v2_id = network.version_ids[1]
        result = runner.invoke(
            cli, ["compare", v1_id, v2_id], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "DAGVersion Diff" in result.output

    def test_compare_missing_version_exits_1(
        self, runner: CliRunner, isolated_env: Path
    ) -> None:
        result = runner.invoke(
            cli, ["compare", "nonexistent-1", "nonexistent-2"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "not found" in result.output
