# pir-api

A small web service around [PIR](https://github.com/Qazi-pk/physics-engine), an open-source
symbolic-regression engine (DOI [10.5281/zenodo.23460311](https://doi.org/10.5281/zenodo.23460311)).
Upload a CSV, name the column to explain, and get back a closed-form formula, which stage of
PIR found it, and its accuracy (R²) on rows held out from fitting.

## Run

With Docker:

```bash
docker build -t pir-api .
docker run -p 8000:8000 pir-api
```

Without Docker (Python 3.10+):

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Then open <http://localhost:8000> for the upload page, or <http://localhost:8000/docs> for the API.

The page reads the column names from your file and lets you pick the one to explain; every other
column is used as an input. Four example buttons (pendulum, thin lens, gravity, radioactive decay)
load made-up data generated from a known law, so you can see what a correct answer looks like.

## API

`POST /fit` — multipart form:

| field | meaning |
|---|---|
| `file` | CSV with a header row; numeric columns only |
| `target` | name of the column to explain |
| `gates` | `all` (default), `none`, `rational`, `exp` or `noisy` |

```bash
curl -F file=@data.csv -F target=QT http://localhost:8000/fit
```

Example response (QT interval vs RR interval, synthetic data):

```json
{"formula": "420.043*RR**(1/3)", "found_by": "noisy", "r2_heldout_rows": 0.9616,
 "rows_used": 2000, "fit_seconds": 2.3, "pir_engine": "3.6.0"}
```

`formula` has constants rounded to 6 significant figures; `formula_full_precision` is the unrounded model.

`GET /examples/{name}.csv` returns an example table (`pendulum`, `lens`, `gravity`, `decay`).

`GET /health` returns the service and engine versions.

Limits: 5 MB upload, at most 8 input columns; tables over 20,000 rows are subsampled
(seeded). 25% of rows are held out to report accuracy.

## What the answer means

The formula is a **candidate**. R² on held-out rows says how well it predicts data it was not
fitted on; it does not prove a physical law. `found_by` tells you which part of PIR produced it:
`engine` (the main search), or one of the fallback gates `rational`, `exp`, `noisy`
(see the [PIR README](https://github.com/Qazi-pk/physics-engine#fallback-gates-new-in-36)).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests
```

## License

MIT. Parts of this code were written with AI assistance (Claude, Anthropic); the author reviewed and tested it.
