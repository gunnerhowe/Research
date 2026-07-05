"""Classic econometric reference datasets for the A-E0 faithfulness gate.

- Mroz87 (Mroz 1987, female labour supply): Greene (2002) example 22.8 spec.
- RandHIE (RAND Health Insurance Experiment): Cameron & Trivedi (2005)
  section 16.6 spec; seven-digit Stata reference output shipped in
  data/mma16p3selection.txt (downloaded from cameron.econ.ucdavis.edu).

Both CSVs are committed under data/ (source: vincentarelbundock/Rdatasets,
package sampleSelection).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols: dict[str, list] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)

    def conv(vals):
        out = []
        for v in vals:
            if v in ("", "NA"):
                out.append(np.nan)
            elif v == "TRUE":
                out.append(1.0)
            elif v == "FALSE":
                out.append(0.0)
            else:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(np.nan)
        return np.array(out)

    return {k: conv(v) for k, v in cols.items()}


def load_mroz_greene():
    """Greene (2002) ex. 22.8: selection lfp ~ age + age^2 + faminc + kids +
    educ; outcome wage ~ exper + exper^2 + educ + city."""
    d = _read_csv(DATA / "Mroz87.csv")
    n = len(d["lfp"])
    kids = ((d["kids5"] + d["kids618"]) > 0).astype(float)
    W = np.column_stack([np.ones(n), d["age"], d["age"] ** 2, d["faminc"],
                         kids, d["educ"]])
    X = np.column_stack([np.ones(n), d["exper"], d["exper"] ** 2, d["educ"],
                         d["city"]])
    s = d["lfp"]
    y = np.where(s > 0.5, d["wage"], np.nan)
    names_sel = ["const", "age", "age2", "faminc", "kids", "educ"]
    names_out = ["const", "exper", "exper2", "educ", "city"]
    return y, X, s, W, names_out, names_sel


RANDHIE_XLIST = ["logc", "idp", "lpi", "fmde", "physlm", "disea", "hlthg",
                 "hlthf", "hlthp", "linc", "lfam", "educdec", "xage",
                 "female", "child", "fchild", "black"]


def load_randhie_ct():
    """Cameron & Trivedi (2005) sec. 16.6: subsample year==2 & educdec
    present; selection binexp ~ XLIST; outcome lnmeddol ~ XLIST.

    Note: no exclusion restriction in this spec (identification by
    functional form) -- C&T use it as exactly such an example.
    """
    d = _read_csv(DATA / "RandHIE.csv")
    keep = (d["year"] == 2) & ~np.isnan(d["educdec"])
    cols = [d[c][keep] for c in RANDHIE_XLIST]
    n = int(keep.sum())
    Z = np.column_stack(cols + [np.ones(n)])
    s = d["binexp"][keep]
    y = np.where(s > 0.5, d["lnmeddol"][keep], np.nan)
    names = RANDHIE_XLIST + ["const"]
    return y, Z, s, Z, names, names
