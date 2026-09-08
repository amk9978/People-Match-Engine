# Match Engine

[![CI](https://github.com/amk9978/Match-Engine/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amk9978/Match-Engine/actions/workflows/ci-cd.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Point it at a roster of people and it returns the group most worth putting in a
room together. Everyone is a node. Every pair gets an edge weighted by two
signals, how alike the two are and how well they complete each other. The densest
group in that graph is the answer.

![Analysis view](docs/1.png)
![Network view](docs/2.png)

## Quickstart

```bash
pip install git+https://github.com/amk9978/Match-Engine

match-engine match docs/sample.csv --map presets/vendor_six_column.yaml \
  --prompt "I am hiring engineers for a fintech startup"

match-engine recommend docs/sample.csv --map presets/vendor_six_column.yaml \
  --person "Sarah Chen" --top 5
```

No key, no server, no Redis. Embeddings run locally through `fastembed`, whose
130MB model downloads on the first run only. Complementarity comes from those
same vectors unless `OPENAI_API_KEY` is set, which switches on the model that
scores it properly. A run names its scorer.

For the HTTP API and the views above, `cp .env.example .env` then
`docker compose up -d`. `POST /analyze` takes the CSV and returns a job id, and
`/jobs/{job_id}/result` holds the group once it completes. One person's ranking
lives at `/jobs/{job_id}/people/{position}/matches`. Full API at `/docs`.

## Your data, your weighting

Any CSV works. It needs a column naming each person and one column of text worth
comparing. The engine reads the columns off the file, ranks them by distinct-value
count, and infers each one's tag separator. To name them yourself, pass `--map`.

```yaml
name_column: Attendee
skip_rows: 0                               # vendor exports often prepend notes
features:
  - {name: skills, column: What they do, separator: "|"}
  - {name: seeking, column: Looking for}   # separator inferred from the cells
```

Skew comes from the `prompt`, which sets a per-feature importance and direction.
Direction is what lets one run ask for the same industry and a different role.
`--min-density` sets how tight the group must be, and `--weights` takes a JSON
pair of per-feature vectors replacing the measured ones outright. Your own
LinkedIn export works through
[`presets/linkedin_connections.yaml`](presets/linkedin_connections.yaml).

## How a run works

1. **Map.** The CSV becomes a feature set, one descriptor per scored column.
2. **Embed.** Each column becomes a vector per person, then a similarity matrix.
3. **Score.** Each pair of profile values is rated for complementarity.
4. **Measure.** Informativeness is a feature's interquartile range over its raw
   pair scores, so a column where all look alike weighs nothing.
5. **Interpret.** Your sentence becomes an importance and direction per feature.
6. **Calibrate.** Rank normalization makes 0.8 the 80th percentile pair everywhere.
7. **Combine.** The factors multiply into weights, and a power mean sets the edge.

Measuring before calibrating is load-bearing, since rank normalization flattens
the spread step 4 reads. Greedy peeling then drops the lightest node each pass,
keeping the densest graph it sees and ranking everyone's neighbours before it goes.

## What is unusual here

1. **Built for professional networking.** The open source tools here assign
   students to coursework groups.
2. **Complementarity is a first-class signal**, with its own matrix and weight
   vector beside similarity.
3. **Intent picks the match type at run time.** One roster answers "find me an
   investor" and "find me a co-founder" differently.
4. **It installs**, in one pip command. The rest of this space is papers and
   hosted platforms.

Densest-subgraph team formation is published work, in
[Gajewar and Das Sarma, 2011](https://arxiv.org/abs/1102.3340) and
[Rangapuram, Bühler and Hein, 2015](https://arxiv.org/abs/1505.06661), with
complementarity beside similarity in C3 and TeamUp. The overlap sits at the
objective. Below it the graph comes from embeddings, complementarity gets its own
scored matrix, and the weights resolve per run from one sentence.

## Limits

The graph is complete, so memory and peeling time grow as N², and the analysis
that follows peeling is heavier still. `X-User-ID` is a trusted header, so put
the API behind your own auth before exposing it. The keyless scorer reads
complementarity off the same vectors similarity comes from, so the two signals
correlate in a way they do not when a model scores the pairs.

The complementarity design, idea through implementation, is
[Amir Karimi](https://github.com/amk9978)'s own, worked out without knowledge of
the work cited above. Apache 2.0, see [LICENSE](LICENSE).
