import pandas as pd
import pytest

from match_engine.services.features.feature_set import WHOLE_CELL, Feature, FeatureSet
from match_engine.services.features.schema_mapper import SchemaError, SchemaMapper

VENDOR_PRESET = "presets/vendor_six_column.yaml"


@pytest.fixture
def vendor_frame():
    return pd.read_csv("docs/sample.csv")


@pytest.fixture
def foreign_frame():
    """A roster whose columns share no name with the vendor layout."""
    return pd.DataFrame(
        {
            "Attendee": ["Ada", "Grace", "Katherine", "Dorothy"],
            "Employer": ["Analytical", "Univac", "NACA", "IBM"],
            "Profile URL": ["a", "b", "c", "d"],
            "What they do": [
                "engine design, mathematics",
                "compilers, tooling",
                "orbital mechanics",
                "systems research",
            ],
            "Looking for": ["funding", "hires", "collaborators", "advisors"],
            "Seniority": ["founder", "principal", "principal", "director"],
        }
    )


class TestFeature:
    def test_a_separator_splits_a_cell_into_tags(self):
        feature = Feature(name="role", column="Role", separator="|")
        assert feature.split("Founder | Systems") == ["Founder", "Systems"]

    def test_no_separator_keeps_the_whole_cell(self):
        feature = Feature(name="role", column="Role", separator=WHOLE_CELL)
        assert feature.split("Founder | Systems") == ["Founder | Systems"]

    def test_blank_cells_yield_no_tags(self):
        feature = Feature(name="role", column="Role", separator="|")
        assert feature.split("  |  ") == []

    def test_a_feature_needs_a_column(self):
        with pytest.raises(AssertionError):
            Feature(name="role", column="")


class TestFeatureSet:
    def test_duplicate_names_are_rejected(self):
        with pytest.raises(AssertionError):
            FeatureSet(
                features=(
                    Feature(name="role", column="A"),
                    Feature(name="role", column="B"),
                ),
                name_column="Name",
            )

    def test_an_empty_feature_set_is_rejected(self):
        with pytest.raises(AssertionError):
            FeatureSet(features=(), name_column="Name")

    def test_required_columns_cover_identity_and_features(self):
        feature_set = FeatureSet(
            features=(Feature(name="role", column="Role"),),
            name_column="Name",
            company_column="Company",
        )
        assert feature_set.required_columns() == ("Name", "Company", "Role")


class TestVendorPreset:
    def test_the_preset_maps_the_six_original_columns(self, vendor_frame):
        feature_set = SchemaMapper().from_file(VENDOR_PRESET, vendor_frame)

        assert feature_set.names == (
            "role",
            "experience",
            "industry",
            "market",
            "offering",
            "persona",
        )
        assert feature_set.name_column == "Person Name"

    def test_the_preset_keeps_the_persona_semicolon(self, vendor_frame):
        feature_set = SchemaMapper().from_file(VENDOR_PRESET, vendor_frame)
        assert feature_set.feature("persona").separator == ";"

    def test_a_column_the_dataset_lacks_is_refused(self, foreign_frame):
        with pytest.raises(SchemaError):
            SchemaMapper().from_file(VENDOR_PRESET, foreign_frame)


class TestDetection:
    def test_a_foreign_roster_yields_a_usable_feature_set(self, foreign_frame):
        feature_set = SchemaMapper().detect(foreign_frame)

        assert len(feature_set) >= 1
        assert feature_set.name_column == "Attendee"
        assert feature_set.company_column == "Employer"

    def test_identity_columns_are_never_scored(self, foreign_frame):
        feature_set = SchemaMapper().detect(foreign_frame)

        scored = set(feature_set.columns.values())
        assert "Attendee" not in scored
        assert "Employer" not in scored

    def test_link_columns_are_skipped(self, foreign_frame):
        feature_set = SchemaMapper().detect(foreign_frame)
        assert "Profile URL" not in feature_set.columns.values()

    def test_the_feature_count_is_capped(self, vendor_frame):
        feature_set = SchemaMapper(max_features=2).detect(vendor_frame)
        assert len(feature_set) == 2

    def test_a_declared_mapping_over_the_cap_is_refused(self, vendor_frame):
        with pytest.raises(SchemaError):
            SchemaMapper(max_features=2).from_file(VENDOR_PRESET, vendor_frame)

    def test_a_roster_with_no_name_column_is_refused(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        with pytest.raises(SchemaError):
            SchemaMapper().detect(df)

    def test_a_pipe_separated_column_gets_the_pipe(self):
        df = pd.DataFrame(
            {
                "Person Name": ["Ada", "Grace"],
                "Skills": ["a | b", "c | d"],
            }
        )
        feature_set = SchemaMapper().detect(df)
        assert feature_set.feature("skills").separator == "|"

    def test_a_prose_column_keeps_the_whole_cell(self):
        df = pd.DataFrame(
            {
                "Person Name": ["Ada", "Grace"],
                "Summary": ["builds analytical engines", "writes compilers"],
            }
        )
        feature_set = SchemaMapper().detect(df)
        assert feature_set.feature("summary").separator == WHOLE_CELL

    def test_a_constant_column_is_not_worth_scoring(self):
        df = pd.DataFrame(
            {
                "Person Name": ["Ada", "Grace"],
                "Event": ["Summit", "Summit"],
                "Skills": ["a", "b"],
            }
        )
        feature_set = SchemaMapper().detect(df)
        assert "Event" not in feature_set.columns.values()
