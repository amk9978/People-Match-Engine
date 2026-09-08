import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from match_engine.services.graph.recommendations import (
    Match,
    PersonNotFound,
    recommender_for,
)
from match_engine.services.match_run import MatchRequest, MatchResult, MatchRun
from match_engine.services.scoring.scorer_factory import CHOICES
from match_engine.services.scoring.weights import ExplicitWeights, WeightsError

app = typer.Typer(
    add_completion=False,
    help="Find the group of people most worth putting in a room together.",
)

DEFAULT_TOP = 5


class Scorer(str, Enum):
    auto = "auto"
    llm = "llm"
    embedding = "embedding"


class Progress:
    """Prints each stage to stderr, leaving stdout free to be piped."""

    async def report(self, stage: str) -> None:
        typer.echo(f"{stage}...", err=True)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )


def _request(
    roster: Path, prompt, mapping, min_density, scorer, weights=None
) -> MatchRequest:
    if not roster.is_file():
        raise typer.BadParameter(f"no such file: {roster}")
    return MatchRequest(
        csv_path=str(roster),
        prompt=prompt,
        min_density=min_density,
        mapping_path=str(mapping) if mapping else None,
        scorer_choice=scorer.value,
        weights=_weights(weights),
    )


def _weights(path: Optional[Path]) -> Optional[ExplicitWeights]:
    """Read weights the caller decided, in place of the measured ones."""
    if path is None:
        return None
    if not path.is_file():
        raise typer.BadParameter(f"no such file: {path}")
    try:
        return ExplicitWeights.parse(path.read_text())
    except WeightsError as invalid:
        raise typer.BadParameter(str(invalid))


@app.command()
def match(
    roster: Path = typer.Argument(..., help="CSV of people, one row each."),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="What kind of match you want."
    ),
    mapping: Optional[Path] = typer.Option(
        None, "--map", "-m", help="YAML naming the columns to score."
    ),
    min_density: Optional[float] = typer.Option(
        None, "--min-density", "-d", help="How tight the returned group must be."
    ),
    scorer: Scorer = typer.Option(Scorer.auto, "--scorer", "-s", help=str(CHOICES)),
    weights: Optional[Path] = typer.Option(
        None, "--weights", "-w", help="JSON weights, replacing the measured ones."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the result document."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Return the densest group in the roster."""
    _configure_logging(verbose)
    request = _request(roster, prompt, mapping, min_density, scorer, weights)
    result = asyncio.run(MatchRun(request, progress=Progress()).execute())

    if as_json:
        typer.echo(json.dumps(_as_document(result), indent=2))
        return
    _print_group(result)


@app.command()
def recommend(
    roster: Path = typer.Argument(..., help="CSV of people, one row each."),
    person: str = typer.Option(..., "--person", help="Whose matches to rank."),
    top: int = typer.Option(DEFAULT_TOP, "--top", "-k", min=1),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    mapping: Optional[Path] = typer.Option(None, "--map", "-m"),
    scorer: Scorer = typer.Option(Scorer.auto, "--scorer", "-s"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w"),
    as_json: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Rank one person's best matches in the roster."""
    _configure_logging(verbose)
    request = _request(roster, prompt, mapping, None, scorer, weights)
    run = MatchRun(request, progress=Progress())
    asyncio.run(run.execute())

    recommender = recommender_for(run.graph_builder)
    try:
        position = recommender.position_of(person)
    except PersonNotFound as missing:
        raise typer.BadParameter(str(missing))

    matches = recommender.top_matches(position, top)
    if as_json:
        typer.echo(json.dumps(_as_matches(person, matches), indent=2))
        return
    _print_matches(person, matches)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("match_engine.presentation.api_controller:app", host=host, port=port)


def _as_document(result: MatchResult) -> dict:
    return {
        "nodes": result.nodes,
        "names": [result.names[position] for position in result.nodes],
        "density": round(result.density, 4),
        "row_count": result.row_count,
        "complementarity": result.report.to_dict(),
    }


def _as_matches(person: str, matches: List[Match]) -> dict:
    return {
        "person": person,
        "matches": [
            {
                "position": match.position,
                "name": match.name,
                "company": match.company,
                "weight": round(match.weight, 4),
                "features": [
                    {
                        "feature": feature.feature,
                        "similarity": round(feature.similarity, 4),
                        "complementarity": round(feature.complementarity, 4),
                    }
                    for feature in match.features
                ],
            }
            for match in matches
        ],
    }


def _print_group(result: MatchResult) -> None:
    typer.echo(
        f"\n{len(result.nodes)} of {result.row_count} people, "
        f"density {result.density:.3f}\n"
    )
    for position in result.nodes:
        typer.echo(f"  {position:>4}  {result.names[position]}")
    report = result.report
    typer.echo(
        f"\nComplementarity via {report.scorer}: {report.scored_pairs} scored, "
        f"{report.cached_pairs} cached, {report.fallback_pairs} fell back."
    )


def _print_matches(person: str, matches: List[Match]) -> None:
    typer.echo(f"\nBest matches for {person}\n")
    for rank, match in enumerate(matches, start=1):
        company = f" at {match.company}" if match.company else ""
        typer.echo(f"  {rank}. {match.name}{company}  ({match.weight:.3f})")
        for feature in match.features:
            typer.echo(
                f"       {feature.feature:<14} same {feature.similarity:.2f}  "
                f"complement {feature.complementarity:.2f}"
            )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
