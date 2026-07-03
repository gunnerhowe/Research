"""Generate paper/numbers.tex from results JSONs — every number in the paper comes
from here, never hand-typed. Convention: macros expand to BARE math content (no $);
prose uses them inside $...$, and generated table rows wrap their own cells."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).parent / "numbers.tex"

exp1 = json.loads((RES / "exp1_convergence.json").read_text())
exp2 = json.loads((RES / "exp2_decisive.json").read_text())
exp3 = json.loads((RES / "exp3_multiseed.json").read_text())["aggregate"]
exp4 = json.loads((RES / "exp4_ks.json").read_text())["aggregate"]
exp5 = json.loads((RES / "exp5_esn.json").read_text())
exp5b = json.loads((RES / "exp5b_esn_degraded.json").read_text())

L = []


def cmd(name, body):
    L.append(f"\\newcommand{{\\{name}}}{{{body}}}")


def pm(pair, d=4):
    return f"{pair[0]:.{d}f} \\pm {pair[1]:.{d}f}"


def sci(v):
    if not np.isfinite(v):
        return None
    m, e = f"{v:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


# ---- Lorenz (exp3) -------------------------------------------------------------
S3 = exp3["systems"]
NAMES = [("truth2", "independent truth (neg.\\ ctrl)"),
         ("iaaft", "IAAFT"),
         ("speed2", "speed$\\times2$"),
         ("reversed", "time-reversed"),
         ("rho32", "$\\rho=32$ (pos.\\ ctrl)")]

cmd("lorTruthH", pm(exp3["h_truth"]))
cmd("lorTruthLam", pm(exp3["lambda1_truth"], 3))

rows_a, rows_b = [], []
for key, label in NAMES:
    e = S3[key]
    w3 = ("---" if e["w1_state3d"][0] is None
          else f"${pm(e['w1_state3d'], 3)}$ (${e['w1_state3d_floor'][0]:.3f}$)")
    wd = f"${pm(e['w1_delay'], 3)}$ (${e['w1_delay_floor'][0]:.3f}$)"
    rows_a.append(f"{label} & ${pm(e['w1_marginal_x'])}$ & {w3} & {wd} & "
                  f"${pm(e['psd_logdist_db'], 2)}$ & ${pm(e['acf_rmse'])}$ \\\\")
    lam = f"${pm(e['lambda1'], 3)}$ [${e['lambda1_r2'][0]:.2f}$]"
    db = f"${pm(e['dbar'])}$ (${e['dbar_floor'][0]:.4f}$)"
    rows_b.append(f"{label} & {lam} & ${pm(e['h_block'])}$ & ${pm(e['fano_lb'])}$ & "
                  f"{db} & ${pm(e['dbar_sep'])}$ \\\\")
cmd("lorTableRowsA", "\n".join(rows_a))
cmd("lorTableRowsB", "\n".join(rows_b))

cmd("lorTiiWmarg", pm(S3["truth2"]["w1_marginal_x"]))
cmd("lorIaaftWmarg", pm(S3["iaaft"]["w1_marginal_x"]))
cmd("lorTiiPsd", pm(S3["truth2"]["psd_logdist_db"], 2))
cmd("lorIaaftPsd", pm(S3["iaaft"]["psd_logdist_db"], 2))
cmd("lorIaaftDbarSep", pm(S3["iaaft"]["dbar_sep"]))
cmd("lorIaaftFano", pm(S3["iaaft"]["fano_lb"]))
cmd("lorSpdWstate", pm(S3["speed2"]["w1_state3d"], 3))
cmd("lorSpdWstateFloor", f"{S3['speed2']['w1_state3d_floor'][0]:.3f}")
cmd("lorSpdDbarSep", pm(S3["speed2"]["dbar_sep"]))
cmd("lorRhoWstate", pm(S3["rho32"]["w1_state3d"], 3))
cmd("lorRhoDbarSep", pm(S3["rho32"]["dbar_sep"]))
cmd("lorRhoFano", pm(S3["rho32"]["fano_lb"]))
cmd("lorTiiDbarSep", pm(S3["truth2"]["dbar_sep"]))
cmd("lorRevDbarSep", pm(S3["reversed"]["dbar_sep"]))


def ratio(e):
    return int(round(e["dbar_sep"][0] / e["dbar_floor"][0]))


# sep/floor ratios, macro-protected so no hand-typed multiplier can drift
cmd("lorIaaftRatio", str(ratio(S3["iaaft"])))
cmd("lorSpdRatio", str(ratio(S3["speed2"])))

# ---- scaling (exp1) --------------------------------------------------------------
for name, tag in (("iaaft", "Iaaft"), ("speed2", "Spd")):
    for nsub, suff in ((10_000, "A"), (100_000, "B"), (1_000_000, "C")):
        rows = exp1["scaling"][name][str(nsub)]
        sep = max(r["dbar"] - r["floor"] for r in rows)
        cmd(f"scal{tag}{suff}", f"{sep:.3f}")

# ---- sensitivity (exp2) -----------------------------------------------------------
sens_rows = []
label_map = {"sign(x) m=2": "sign$(x)$, $m{=}2$, $\\tau{=}0.1$ (reference)",
             "quantile4(x) m=4": "quantile-4$(x)$, $m{=}4$",
             "box(x,z) m=6": "box$(x,z)$, $m{=}6$"}
for pname, res in exp2["partition_sensitivity"].items():
    cells = [f"${res[k]['sep']:.4f}$" if k in res else "---"
             for k in ("truth2", "iaaft", "speed2")]
    sens_rows.append(f"{label_map[pname]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
for tau, res in exp2["tau_sensitivity"].items():
    cells = [f"${res[k]['sep']:.4f}$" for k in ("truth2", "iaaft", "speed2")]
    sens_rows.append(f"sign$(x)$, $\\tau{{=}}{tau}$ & {cells[0]} & {cells[1]} & "
                     f"{cells[2]} \\\\")
cmd("sensTableRows", "\n".join(sens_rows))

# ---- KS (exp4) --------------------------------------------------------------------
S4 = exp4["systems"]
cmd("ksTruthH", pm(exp4["h_truth"]))
ks_rows = []
for key, label in (("truth2", "independent truth (neg.\\ ctrl)"),
                   ("iaaft", "IAAFT"), ("speed2", "speed$\\times2$")):
    e = S4[key]
    ws = ("---" if e["w1_state"][0] is None
          else f"${pm(e['w1_state'], 3)}$ (${e['w1_state_floor'][0]:.3f}$)")
    ks_rows.append(f"{label} & ${pm(e['w1_marginal'])}$ & {ws} & "
                   f"${pm(e['psd_logdist_db'], 2)}$ & ${pm(e['h_block'])}$ & "
                   f"${pm(e['fano_lb'])}$ & ${pm(e['dbar_sep'])}$ \\\\")
cmd("ksTableRows", "\n".join(ks_rows))
cmd("ksIaaftDbarSep", pm(S4["iaaft"]["dbar_sep"]))
cmd("ksSpdDbarSep", pm(S4["speed2"]["dbar_sep"]))
cmd("ksSpdRatio", str(int(round(S4["speed2"]["dbar_sep"][0]
                                / S4["speed2"]["dbar_floor"][0]))))
cmd("ksNegSep", f"{S4['truth2']['dbar_sep'][0]:.4f}")

# ---- ESN (exp5 healthy sweep + exp5b degraded) --------------------------------------


def esn_row(label, ms):
    vals = [m["one_step_nrmse"] for m in ms if np.isfinite(m["one_step_nrmse"])]
    nr = sci(np.mean(vals)) if vals else None
    nr_cell = f"${nr}$" if nr else "---"
    status = "/".join(sorted({m["status"] for m in ms}))
    have = [m for m in ms if "w1_state" in m]
    if not have:
        return f"{label} & {nr_cell} & {status} & --- & --- & --- & --- \\\\"
    w1 = np.mean([m["w1_state"] for m in have])
    w1f = np.mean([m["w1_state_floor"] for m in have])
    h = np.mean([m["h_block"] for m in have])
    fano = np.mean([m["fano_lb"] for m in have])
    sep = np.mean([m["dbar_sep"] for m in have])
    return (f"{label} & {nr_cell} & {status} & ${w1:.3f}$ (${w1f:.3f}$) & "
            f"${h:.4f}$ & ${fano:.4f}$ & ${sep:.4f}$ \\\\")


esn_rows = []
for rho in sorted({m["rho_spec"] for m in exp5["models"]}):
    ms = [m for m in exp5["models"] if m["rho_spec"] == rho]
    w1s = [m["w1_state"] for m in ms if "w1_state" in m]
    # bimodal outcomes (one healthy, one failed seed): don't average — split rows
    if len(w1s) == 2 and max(w1s) > 3 * min(w1s):
        for m in ms:
            esn_rows.append(esn_row(f"$\\rho_s={rho:.2f}$ (seed {m['seed']})", [m]))
    else:
        esn_rows.append(esn_row(f"$\\rho_s={rho:.2f}$", ms))
esn_rows.append("\\midrule")
DEG_LABELS = {"linear-readout": "linear readout (no $r^2$)",
              "undertrained": "undertrained ($2{\\times}10^3$ samples)",
              "tiny-reservoir": "tiny reservoir ($n_r=50$)",
              "over-regularized": "over-regularized (ridge $10^{-1}$)"}
for cfg in ["linear-readout", "undertrained", "tiny-reservoir", "over-regularized"]:
    ms = [m for m in exp5b["models"] if m["config"] == cfg]
    esn_rows.append(esn_row(DEG_LABELS[cfg], ms))
cmd("esnTableRows", "\n".join(esn_rows))

healthy = [m for m in exp5["models"] if "w1_state" in m
           and m["w1_state"] < 3 * m["w1_state_floor"]]
cmd("esnHealthyMaxAbsDbar", f"{max(abs(m['dbar_sep']) for m in healthy):.3f}")
bad = max((m for m in exp5["models"] if "w1_state" in m),
          key=lambda m: m["w1_state"])
cmd("esnBadW", f"{bad['w1_state']:.2f}")
cmd("esnBadDbar", f"{bad['dbar_sep']:.2f}")
cmd("esnBadH", f"{bad['h_block']:.3f}")
lin = sorted([m for m in exp5b["models"] if m["config"] == "linear-readout"],
             key=lambda m: m["w1_state"])
cmd("esnLinWa", f"{lin[0]['w1_state']:.3f}")
cmd("esnLinWaFloor", f"{lin[0]['w1_state_floor']:.3f}")
cmd("esnLinDbarA", f"{lin[0]['dbar_sep']:.3f}")
cmd("esnLinDbarB", f"{lin[1]['dbar_sep']:.3f}")
cmd("esnLinHa", f"{lin[0]['h_block']:.3f}")
cmd("esnLinHb", f"{lin[1]['h_block']:.3f}")
cmd("esnLinDbarFloor", f"{lin[0]['dbar_floor']:.3f}")
_lin_ratios = [m["dbar_sep"] / m["dbar_floor"] for m in lin]
cmd("esnLinRatioLo", str(int(round(min(_lin_ratios)))))
cmd("esnLinRatioHi", str(int(round(max(_lin_ratios)))))

OUT.write_text("% AUTO-GENERATED by gen_paper_numbers.py -- do not edit by hand\n"
               + "\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {OUT} with {len(L)} macros")
