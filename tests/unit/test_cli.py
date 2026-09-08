import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from match_engine.cli import app
from match_engine.services.graph.recommendations import (
    FeatureContribution,
    Match,
    PersonNotFound,
)
from match_engine.services.match_run import MatchResult
from match_engine.services.scoring.report import ScoringReport

runner = CliRunner()

ROWS = [
    ("Ada", "Analytical", "engineer", "computing"),
    ("Grace", "Univac", "compiler", "computing"),
    ("Katherine", "NACA", "mathematician", "aerospace"),
]

RESULT = MatchResult(
    nodes=[0, 1],
    names=["Ada", "Grace", "Katherine"],
    density=0.8125,
    row_count=3,
    report=ScoringReport(scorer="embedding", scored_pairs=9),
    info={"size": 2},
)

MATCHES = [
    Match(
        position=1,
        name="Grace",
        company="Univac",
        weight=0.91,
        features=[FeatureContribution("role", 0.8, 0.2)],
    )
]


@pytest.fixture
def roster(tmp_path):
    path = tmp_path / "roster.csv"
    pd.DataFrame(ROWS, columns=["Name", "Company", "Role", "Industry"]).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def stub_run(monkeypatch):
    """Replaces the pipeline, so the CLI surface is tested without a model."""
    seen = {}

    class StubRun:
        def __init__(self, request, progress=None):
            seen["request"] = request
            self.graph_builder = object()

        async def execute(self):
            return RESULT

    monkeypatch.setattr("match_engine.cli.MatchRun", StubRun)
    monkeypatch.setattr(
        "match_engine.cli.recommender_for", lambda builder: StubRanker()
    )
    return seen


class StubRanker:
    def position_of(self, name):
        if name.strip().casefold() != "ada":
            raise PersonNotFound(f"no person named {name!r}")
        return 0

    def top_matches(self, position, k):
        return MATCHES[:k]


class TestMatchCommand:
    def test_a_run_prints_the_group(self, roster, stub_run):
        result = runner.invoke(app, ["match", str(roster)])

        assert result.exit_code == 0
        assert "Ada" in result.stdout
        assert "density 0.812" in result.stdout

    def test_json_output_carries_the_group_and_the_report(self, roster, stub_run):
        result = runner.invoke(app, ["match", str(roster), "--json"])

        document = json.loads(result.stdout)
        assert document["nodes"] == [0, 1]
        assert document["names"] == ["Ada", "Grace"]
        assert document["density"] == 0.8125
        assert document["complementarity"]["scorer"] == "embedding"

    def test_the_flags_reach_the_request(self, roster, stub_run):
        runner.invoke(
            app,
            [
                "match",
                str(roster),
                "--prompt",
                "hiring engineers",
                "--min-density",
                "0.3",
                "--scorer",
                "embedding",
            ],
        )

        request = stub_run["request"]
        assert request.prompt == "hiring engineers"
        assert request.min_density == 0.3
        assert request.scorer_choice == "embedding"

    def test_a_missing_roster_exits_non_zero(self, stub_run):
        result = runner.invoke(app, ["match", "nowhere.csv"])

        assert result.exit_code != 0
        assert "no such file" in result.output

    def test_an_unknown_scorer_is_rejected(self, roster, stub_run):
        result = runner.invoke(app, ["match", str(roster), "--scorer", "magic"])

        assert result.exit_code != 0


class TestRecommendCommand:
    def test_matches_print_with_their_breakdown(self, roster, stub_run):
        result = runner.invoke(app, ["recommend", str(roster), "--person", "Ada"])

        assert result.exit_code == 0
        assert "Grace at Univac" in result.stdout
        assert "role" in result.stdout

    def test_json_output_carries_the_features(self, roster, stub_run):
        result = runner.invoke(
            app, ["recommend", str(roster), "--person", "Ada", "--json"]
        )

        document = json.loads(result.stdout)
        assert document["person"] == "Ada"
        assert document["matches"][0]["name"] == "Grace"
        assert document["matches"][0]["features"][0]["complementarity"] == 0.2

    def test_top_caps_the_list(self, roster, stub_run):
        result = runner.invoke(
            app, ["recommend", str(roster), "--person", "Ada", "--top", "1", "--json"]
        )

        assert len(json.loads(result.stdout)["matches"]) == 1

    def test_an_unknown_person_exits_non_zero(self, roster, stub_run):
        result = runner.invoke(app, ["recommend", str(roster), "--person", "Hedy"])

        assert result.exit_code != 0
        assert "Hedy" in result.output

    def test_the_person_option_is_required(self, roster, stub_run):
        assert runner.invoke(app, ["recommend", str(roster)]).exit_code != 0


class TestServeCommand:
    def test_serve_starts_uvicorn_on_the_app(self, monkeypatch):
        called = {}
        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda target, host, port: called.update(locals())
        )

        result = runner.invoke(app, ["serve", "--port", "9001"])

        assert result.exit_code == 0
        assert called["target"] == "match_engine.presentation.api_controller:app"
        assert called["port"] == 9001
