# possession-model

Per-frame ball-possession inference (`H` / `A` / `D` / `X`) for More than a Hack 2027, phase 1.
Unsupervised: no label of the provided match is used anywhere. The method is described in
[TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission

Python 3.11 or newer. The match data is **not** in this repository; point `MTAH_DATA_DIR` at
the unpacked bundle (the folder that contains `tactical/`).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock      # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install --no-deps .
```

```bash
export MTAH_DATA_DIR=/path/to/data                             # PowerShell: $env:MTAH_DATA_DIR = "..."
```

One command regenerates the file:

```bash
.venv/Scripts/python -m possession run --config config/default.yaml --out outputs/KLAKSON.csv
```

The run takes about a minute, logs the SHA-256 of the file it wrote and a ground-truth-free
diagnostics report. The submitted file has

```
sha256  34cc2bd6a62af0e6264d5a0d71f28f0621504f403357b621828b8d93e9d75bbd
```

and the same digest comes out with or without the cache. Nothing in the pipeline is random.

Useful flags:

| flag | effect |
|---|---|
| `--validator <official repo>/evaluation/validate_submission.py` | run the organisers' validator on the output (needs `pandas`, which is in the lock file) |
| `--report outputs/report.json` | save the diagnostics |
| `--no-cache` / `--cache-dir DIR` | skip or relocate the cache of the parsed tracking (`.cache/`, git-ignored) |
| `--data-dir DIR` | use this bundle instead of `MTAH_DATA_DIR` |

## Development

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest          # 122 tests, synthetic data only
.venv/Scripts/ruff check src tests scripts && .venv/Scripts/ruff format --check src tests scripts
.venv/Scripts/mypy                      # strict
.venv/Scripts/python scripts/sensitivity.py --config config/default.yaml   # label-free sweeps
```

`scripts/sensitivity.py` takes `--only section.field ...` to sweep selected parameters and
`--set section.field=value ...` to switch a stage on or off first, for example

```bash
.venv/Scripts/python scripts/sensitivity.py --config config/default.yaml --set touch_snap.enabled=true --only touch_snap.touch_radius_m
```

`scripts/validate_on_metrica.py` scores the pipeline with the official scorer on a public match
of another game (Metrica Sports Sample Games 1, 2 and 3, or a match of the open DFL dataset,
downloaded separately, one folder per match, and kept outside the repository; see
TECHNICAL.md, section 4). `--sparse` adds a damage profile with the ball missing on about
40% of frames:

```bash
.venv/Scripts/python scripts/validate_on_metrica.py --config config/default.yaml --data-dir ../external-data/metrica-sample-game-3 --official-repo ../more-than-a-hack-2027-main
```

Every refinement stage has an `enabled` flag in `config/default.yaml`; the stages that are off
there were measured and rejected (TECHNICAL.md, section 2).

## Layout

```
config/default.yaml        every hyper-parameter, with unit and rationale
src/possession/
  config.py                frozen dataclasses, strict YAML validation, fingerprint
  structures.py            typed containers (Tracking, TaggedPhase), label codes
  runs.py                  run-length helpers
  io/                      ATD parser, Smart Tagging parser, YAML loader, tracking cache
  preprocessing/           slot-to-team masks; ball outlier rejection, gap bridging, speed
  features/                ball-player geometry, collective activity and team shape,
                           kick-off line-up and match structure, ball-free shape cue,
                           ball events (exits, restart kicks, reappearances, touches)
  dead_ball.py             live/dead two-state HMM fitted by EM (no labels)
  ownership.py             team-in-control log-odds from ball-player geometry (and from
                           touches, switched off)
  learned_liveness.py      supervised live/dead evidence (features, emissions); used
  learned_ownership.py     supervised alternative (features, fit); switched off, see
                           TECHNICAL.md section 2
  tagging_prior.py         Smart Tagging as self-calibrated soft votes
  decoding.py              Viterbi, forward-backward, minimum durations
  refinement/              boundary snapping after decoding: stoppage edges to ball events,
                           changes of possession to touches
  validation/              readers, event-based labels and tracking degradation for the public
                           validation matches (never used at inference)
  abstention.py            where X is written
  pipeline.py              the whole inference as one pure function
  consistency.py           label-free checks against tagged set pieces and transitions,
                           and boundary-to-event alignment
  diagnostics.py           label-free report (possession and stoppage counts, shares, ...)
  submission.py            CSV writer (byte-reproducible) and official validator call
  cli.py                   `python -m possession run ...`
scripts/sensitivity.py     label-free sensitivity analysis behind the config values
scripts/validate_on_metrica.py   official scorer on a public validation match
scripts/train_ownership.py       trains and cross-validates the ownership model (off)
scripts/train_liveness.py        trains and cross-validates the live/dead model (used)
tests/                     unit tests on synthetic data
```

## Data from other matches

Validation and the one trained component use public data of other matches, credited here as
their licences ask: the DFL / IDSSE open dataset (Bassek, Rein, Weber, Memmert, *Scientific
Data* 2025; data owner DFL; CC-BY 4.0), SkillCorner Open Data (MIT; credit to SkillCorner) and
Metrica Sports sample data. None of it is in this repository; TECHNICAL.md, section 4, says
where to get it and how it was used.

## Data hygiene

`.gitignore` excludes the data bundle, the cache, `outputs/` and every `*.csv` / `*.npz`: a
submission is a full set of labels and the cache would reconstruct the licensed tracking. The
code logs aggregate statistics only, never tracking rows.
