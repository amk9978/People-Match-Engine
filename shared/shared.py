from enum import Enum, unique

FEATURES = ["role", "experience", "industry", "market", "offering", "persona"]


@unique
class FeatureNames(Enum):
    ROLE = "role"
    EXPERIENCE = "experience"
    INDUSTRY = "industry"
    MARKET = "market"
    OFFERING = "offering"
    PERSONA = "persona"


FEATURE_COLUMN_MAPPING = {
    FeatureNames.ROLE.value: "Professional Identity - Role Specification",
    FeatureNames.EXPERIENCE.value: "Professional Identity - Experience Level",
    FeatureNames.PERSONA.value: "All Persona Titles",
    FeatureNames.INDUSTRY.value: "Company Identity - Industry Classification",
    FeatureNames.MARKET.value: "Company Market - Market Traction",
    FeatureNames.OFFERING.value: "Company Offering - Value Proposition",
}

BUSINESS_FEATURES = [
    FeatureNames.INDUSTRY.value,
    FeatureNames.MARKET.value,
    FeatureNames.OFFERING.value,
]
PERSONAL_FEATURES = [
    FeatureNames.ROLE.value,
    FeatureNames.EXPERIENCE.value,
    FeatureNames.PERSONA.value,
]

DEFAULT_FEATURE_WEIGHTS = {
    FeatureNames.ROLE.value: 1.0,
    FeatureNames.EXPERIENCE.value: 1.0,
    FeatureNames.INDUSTRY.value: 1.0,
    FeatureNames.MARKET.value: 1.0,
    FeatureNames.OFFERING.value: 1.0,
    FeatureNames.PERSONA.value: 1.0,
}

OPTIMIZED_FEATURE_WEIGHTS = {
    FeatureNames.ROLE.value: 1.2,
    FeatureNames.EXPERIENCE.value: 1.5,
    FeatureNames.INDUSTRY.value: 1.1,
    FeatureNames.MARKET.value: 1.1,
    FeatureNames.OFFERING.value: 1.2,
    FeatureNames.PERSONA.value: 0.9,
}
