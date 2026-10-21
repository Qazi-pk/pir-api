"""
pir-api -- a small web service around the PIR symbolic-regression engine.

Upload a CSV, name the target column, get back a closed-form formula, which
stage of PIR produced it, and its accuracy on rows held out from fitting.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
    open http://localhost:8000          (upload page)
    POST /fit  (multipart: file, target, gates)   -> JSON
    GET  /health                                   -> versions
"""
import io
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import physics_engine
from physics_engine.sklearn_adapter import PIRRegressor

__version__ = "0.1.0"

MAX_BYTES = 5 * 1024 * 1024   # 5 MB upload limit
MAX_ROWS = 20_000             # larger files are subsampled (seeded)
MAX_FEATURES = 8              # PIR searches products/powers; keep inputs small
HOLDOUT = 0.25                # share of rows held out for the reported accuracy
GATE_CHOICES = {"all": "all", "none": None, "rational": ("rational",),
                "exp": ("exp",), "noisy": ("noisy",)}

app = FastAPI(title="PIR API", version=__version__,
              description="Find a closed-form formula in tabular data with the PIR engine.")


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    v = float(np.var(y))
    return float(1 - np.mean((y - p) ** 2) / v) if v > 0 else float("nan")


def fit_table(df: pd.DataFrame, target: str, gates: str = "all", seed: int = 0) -> dict:
    """Validate the table, fit PIR on 75% of rows, report accuracy on the other 25%."""
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found; columns are {list(df.columns)}")
    if gates not in GATE_CHOICES:
        raise ValueError(f"gates must be one of {sorted(GATE_CHOICES)}")
    num = df.apply(pd.to_numeric, errors="coerce")
    feats = [c for c in num.columns if c != target]
    bad = [c for c in feats + [target] if num[c].isna().all()]
    if bad:
        raise ValueError(f"non-numeric columns: {bad}")
    if not feats:
        raise ValueError("need at least one input column besides the target")
    if len(feats) > MAX_FEATURES:
        raise ValueError(f"at most {MAX_FEATURES} input columns (got {len(feats)})")
    num = num[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    dropped = int(len(df) - len(num))
    if len(num) < 20:
        raise ValueError(f"need at least 20 complete numeric rows (got {len(num)})")
    if len(num) > MAX_ROWS:
        num = num.sample(MAX_ROWS, random_state=seed)
    perm = np.random.default_rng(seed).permutation(len(num))
    cut = int((1 - HOLDOUT) * len(num))
    tr, te = num.iloc[perm[:cut]], num.iloc[perm[cut:]]

    t0 = time.time()
    m = PIRRegressor(gates=GATE_CHOICES[gates], random_state=seed).fit(tr[feats], tr[target].values)
    secs = time.time() - t0
    p_tr, p_te = m.predict(tr[feats]), m.predict(te[feats])
    info = getattr(m, "gate_info_", {}) or {}
    return {
        "formula": m.model(),
        "target": target,
        "inputs": feats,
        "found_by": getattr(m, "final_source_", "engine"),
        "gate_decision": info.get("decision"),
        "r2_fit_rows": _r2(tr[target], p_tr),
        "r2_heldout_rows": _r2(te[target], p_te),
        "rows_used": int(len(num)),
        "rows_dropped_missing": dropped,
        "heldout_share": HOLDOUT,
        "fit_seconds": round(secs, 2),
        "pir_engine": physics_engine.__version__,
        "note": "R² on held-out rows is the honest accuracy. A formula is a candidate, "
                "not a proven law: check it against domain knowledge.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "pir_api": __version__, "pir_engine": physics_engine.__version__}


@app.post("/fit")
def fit(file: UploadFile = File(...), target: str = Form(...), gates: str = Form("all")):
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"could not read CSV: {e}")
    try:
        return fit_table(df, target.strip(), gates)
    except ValueError as e:
        raise HTTPException(422, str(e))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PIR formula finder</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#5f5f5a;--acc:#1f5f8b;--onacc:#fff;--card:#fff;--bd:#dedcd4}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--acc:#7fb6dd;--onacc:#10202c;--card:#1f1f1d;--bd:#383833}}
a{color:var(--acc)}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:720px;margin:auto}h1{font-size:1.6rem;margin:0 0 4px}p{color:var(--mut)}
form,#out{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-top:16px}
label{display:block;margin:10px 0 4px;font-weight:600}input,select,button{font:inherit;width:100%;box-sizing:border-box;padding:8px;border-radius:6px;border:1px solid var(--bd);background:var(--bg);color:var(--fg)}
button{background:var(--acc);color:var(--onacc);font-weight:600;border:0;margin-top:14px;cursor:pointer}code,pre{font-family:ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
</style></head><body><main>
<h1>PIR formula finder</h1>
<p>Upload a CSV of numbers, choose the column to explain, and PIR looks for a closed-form formula.
Accuracy is reported on rows held out from fitting. Max 5 MB, 8 input columns.</p>
<form id="f"><label for="file">CSV file</label><input id="file" name="file" type="file" accept=".csv" required>
<label for="target">Target column name</label><input id="target" name="target" required placeholder="e.g. y">
<label for="gates">Fallback gates</label><select id="gates" name="gates">
<option value="all">all (recommended)</option><option value="none">none (engine only)</option>
<option value="rational">rational only</option><option value="exp">exp only</option><option value="noisy">noisy power law only</option></select>
<button type="submit">Find formula</button></form>
<div id="out" hidden></div>
<p><a href="/docs">API documentation</a> · <a href="https://github.com/Qazi-pk/physics-engine">PIR engine</a></p>
</main><script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onsubmit=async e=>{e.preventDefault();out.hidden=false;out.textContent='Fitting… (can take a minute)';
 try{const r=await fetch('/fit',{method:'POST',body:new FormData(f)});const j=await r.json();
  if(!r.ok){out.textContent='Error: '+(j.detail||r.status);return;}
  out.innerHTML='<b>Formula</b><pre></pre><b>Details</b><pre></pre>';
  const pre=out.querySelectorAll('pre');pre[0].textContent=j.target+' = '+j.formula;
  pre[1].textContent='found by: '+j.found_by+'\\nR² held-out rows: '+j.r2_heldout_rows.toFixed(4)+
   '\\nrows used: '+j.rows_used+'   fit time: '+j.fit_seconds+' s\\nPIR engine '+j.pir_engine+'\\n\\n'+j.note;
 }catch(err){out.textContent='Error: '+err;}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE
