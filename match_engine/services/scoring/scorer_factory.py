import logging

from match_engine import settings
from match_engine.services.scoring.complementarity_scorer import ComplementarityScorer
from match_engine.services.scoring.embedding_scorer import (
    EmbeddingComplementarityScorer,
)
from match_engine.services.scoring.llm_scorer import LLMComplementarityScorer

logger = logging.getLogger(__name__)

AUTO = "auto"
LLM = "llm"
EMBEDDING = "embedding"
CHOICES = (AUTO, LLM, EMBEDDING)


def create_complementarity_scorer(choice: str = None) -> ComplementarityScorer:
    """Pick the scorer this run will use.

    Under auto a key decides, so a stranger with no key still gets real numbers
    and an operator who supplied one gets the faithful signal."""
    selected = choice or settings.COMPLEMENTARITY_SCORER
    if selected not in CHOICES:
        raise ValueError(f"scorer must be one of {CHOICES}, got {selected!r}")

    if selected == AUTO:
        if settings.OPENAI_API_KEY:
            selected = LLM
        else:
            selected = EMBEDDING
        logger.info(f"Complementarity scorer resolved to {selected}")

    if selected == LLM:
        return LLMComplementarityScorer()
    return EmbeddingComplementarityScorer()
