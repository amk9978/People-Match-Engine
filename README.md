# Match Engine

[![CI](https://github.com/amk9978/match-engine/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amk9978/match-engine/actions/workflows/ci-cd.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Point it at a roster of people and it returns the group most worth putting in a
room together. Everyone on the list is a node. Every pair gets an edge weighted
by two signals, how alike the two are and how well they complete each other. The
densest group in that graph is the answer.

![Analysis view](docs/1.png)
![Network view](docs/2.png)

## Quickstart

```bash
git clone https://github.com/amk9978/match-engine && cd match-engine
cp .env.example .env          # set OPENAI_API_KEY
docker compose up -d

curl -X POST http://localhost:8000/analyze -H "X-User-ID: me" \
  -F "file=@docs/sample.csv" \
  -F "prompt=I am hiring engineers for a fintech startup"
```

That returns a job id. Watch `/ws/{client_id}` or poll `/jobs/{job_id}`, then read
`/jobs/{job_id}/result`. Full API at `http://localhost:8000/docs`. Redis is
optional, and without one the engine caches in memory for the process lifetime.
An OpenAI key is required today, because complementarity is the one step that
calls a model. Embeddings run locally through `fastembed`.

## Your data, your weighting

Any CSV works. It needs a column naming each person and at least one column of
text worth comparing. The engine reads the columns off the file, ranks them by
distinct-value count, and infers each one's tag separator. To name them yourself,
write a mapping and set `FEATURE_MAPPING_PATH`, as in
[`presets/vendor_six_column.yaml`](presets/vendor_six_column.yaml), which matches
[`docs/sample.csv`](docs/sample.csv).

```yaml
name_column: Attendee
features:
  - {name: skills, column: What they do, separator: "|"}
  - {name: seeking, column: Looking for}   # separator inferred from the cells
```

Skew comes from the `prompt`, which sets a per-feature importance and direction.
Direction is what lets one run ask for the same industry and a different role.
`min_density` sets how tight the returned group must be and `MAX_FEATURES` caps
how many columns get scored. Going further means Python, since
[`ScoringProfile`](services/scoring/profile.py) and
[`WeightResolver`](services/scoring/weight_resolver.py) are constructor
arguments on `GraphBuilder` rather than config.

## How a run works

1. **Map.** The CSV becomes a feature set, one descriptor per scored column.
2. **Embed.** Each column becomes a vector per person, then a similarity matrix.
3. **Score.** A model rates each pair of profile values for complementarity.
4. **Measure.** Informativeness is a feature's interquartile range over its raw
   pair scores, so a column where all look alike weighs nothing.
5. **Interpret.** Your sentence becomes an importance and a direction per feature.
6. **Calibrate.** Rank normalization makes 0.8 the 80th percentile pair everywhere.
7. **Combine.** The factors multiply into weights, and a power mean sets the edge.

Measuring before calibrating is load-bearing, since rank normalization flattens
the spread step 4 reads. Greedy peeling then drops the lightest node each pass,
keeping the densest working graph it sees.

## What is unusual here

1. **Built for professional networking.** The open source tools here assign
   students to coursework groups.
2. **Complementarity is a first-class signal**, with its own matrix and weight
   vector beside similarity.
3. **Intent picks the match type at run time.** One roster answers "find me an
   investor" and "find me a co-founder" differently.
4. **It installs.** The rest of this space is papers and hosted platforms.

Densest-subgraph team formation is published work, in
[Gajewar and Das Sarma, 2011](https://arxiv.org/abs/1102.3340) and
[Rangapuram, Bühler and Hein, 2015](https://arxiv.org/abs/1505.06661), with
complementarity beside similarity in C3 and TeamUp. The overlap sits at the
objective. Below it the graph comes from embeddings, complementarity gets its own
model-scored matrix, and the weights resolve per run from a sentence.

## Limits

The graph is complete, so memory and peeling time grow as N², and a few hundred
people is comfortable. Complementarity is quadratic in distinct profile values
per feature, and every run reports how many pairs it scored against how many fell
back to a neutral value. `X-User-ID` is a trusted header, so put the API behind
your own auth before exposing it. Next up is a CLI and a `pip install`,
per-person recommendations, and a local scorer that takes the API key off the
default path.

The complementarity design, idea through implementation, is
[Amir Karimi](https://github.com/amk9978)'s own, worked out without knowledge of
the work cited above. Apache 2.0, see [LICENSE](LICENSE).
