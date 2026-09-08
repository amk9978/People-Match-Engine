import pandas as pd
import pytest

from match_engine.services.features.schema_mapper import (
    LoadPlan,
    SchemaError,
    SchemaMapper,
)
from match_engine.services.preprocessing.csv_loader import CSVLoader

PRESET = "presets/linkedin_connections.yaml"

NOTES = (
    "Notes:\n"
    '"When exporting your connection data, you may notice that some of the '
    "email addresses are missing. You will only see email addresses for "
    "connections who have allowed their connections to see or download their "
    'email address."\n'
    "\n"
)

HEADER = "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"

PEOPLE = [
    (
        "Ada",
        "Lovelace",
        "ada",
        "",
        "Analytical Engines",
        "Principal Engineer",
        "20 Feb 2022",
    ),
    (
        "Grace",
        "Hopper",
        "grace",
        "grace@univac.example",
        "Univac",
        "Principal Engineer",
        "7 Oct 2021",
    ),
    ("Katherine", "Johnson", "kj", "", "NACA", "Research Mathematician", "3 Jan 2020"),
    ("Dorothy", "Vaughan", "dv", "", "IBM", "Engineering Manager", "9 Jun 2019"),
    ("Mary", "Jackson", "mj", "", "Langley", "Research Mathematician", "1 Mar 2021"),
    (
        "Radia",
        "Perlman",
        "rp",
        "rp@dec.example",
        "DEC",
        "Network Architect",
        "8 Aug 2018",
    ),
]


def _export(path, rows=PEOPLE, preamble=NOTES):
    lines = [f"https://www.linkedin.com/in/{r[2]}" for r in rows]
    body = "".join(
        f"{r[0]},{r[1]},{url},{r[3]},{r[4]},{r[5]},{r[6]}\n"
        for r, url in zip(rows, lines)
    )
    path.write_text(preamble + HEADER + body)
    return str(path)


@pytest.fixture
def export(tmp_path):
    return _export(tmp_path / "Connections.csv")


class TestLoadPlan:
    def test_the_preset_skips_the_notes_preamble(self):
        plan = SchemaMapper().load_plan(SchemaMapper().read_mapping(PRESET))

        assert plan.skip_rows == 3

    def test_the_preset_joins_the_two_name_columns(self):
        plan = SchemaMapper().load_plan(SchemaMapper().read_mapping(PRESET))

        assert plan.name_columns == ("First Name", "Last Name")
        assert plan.name_column == "Person"

    def test_a_mapping_without_load_options_plans_nothing(self):
        plan = SchemaMapper().load_plan({"features": []})

        assert plan == LoadPlan()

    def test_joining_needs_a_target_column_name(self):
        with pytest.raises(SchemaError):
            SchemaMapper().load_plan({"name_columns": ["First Name", "Last Name"]})

    def test_a_negative_skip_is_rejected(self):
        with pytest.raises(SchemaError):
            SchemaMapper().load_plan({"skip_rows": -1})


class TestLinkedInExport:
    def test_the_preamble_does_not_become_the_header(self, export):
        loader = CSVLoader(export, PRESET)

        df = loader.load_data()

        assert "Position" in df.columns
        assert len(df) == len(PEOPLE)

    def test_the_name_column_carries_both_halves(self, export):
        loader = CSVLoader(export, PRESET)

        df = loader.load_data()

        assert loader.feature_set.name_column == "Person"
        assert list(df["Person"])[:2] == ["Ada Lovelace", "Grace Hopper"]

    def test_the_preset_scores_the_position_column(self, export):
        loader = CSVLoader(export, PRESET)
        loader.load_data()

        assert loader.feature_set.names == ("position",)
        assert loader.feature_set.columns["position"] == "Position"
        assert loader.feature_set.company_column == "Company"

    def test_a_free_text_title_is_one_tag_not_a_split_list(self, export):
        loader = CSVLoader(export, PRESET)
        loader.load_data()

        feature = loader.feature_set.feature("position")
        assert feature.split("Principal Engineer, Platform") == [
            "Principal Engineer, Platform"
        ]

    def test_a_row_missing_its_title_is_dropped(self, tmp_path):
        rows = list(PEOPLE)
        rows[2] = (*rows[2][:5], "", rows[2][6])
        loader = CSVLoader(_export(tmp_path / "c.csv", rows), PRESET)

        df = loader.load_data()

        assert len(df) == len(PEOPLE) - 1
        assert "Katherine Johnson" not in list(df["Person"])

    def test_an_export_without_the_position_column_fails_loudly(self, tmp_path):
        path = tmp_path / "old.csv"
        path.write_text(NOTES + "First Name,Last Name,Company\nAda,Lovelace,AE\n")

        with pytest.raises(SchemaError):
            CSVLoader(str(path), PRESET).load_data()

    def test_the_index_stays_positional_after_the_join(self, export):
        df = CSVLoader(export, PRESET).load_data()

        assert list(df.index) == list(range(len(PEOPLE)))


class TestOneFeatureRoster:
    def test_a_lone_feature_takes_all_the_weight(self, export):
        import numpy as np

        from match_engine.services.scoring.intent import Intent
        from match_engine.services.scoring.weight_resolver import WeightResolver

        matrix = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.5], [0.9, 0.5, 0.0]])
        w_s, w_c = WeightResolver().resolve(
            {"position": matrix},
            {"position": matrix},
            Intent.uniform(("position",)),
        )

        assert w_s == {"position": 1.0}
        assert w_c == {"position": 1.0}
