#!/usr/bin/env python3
"""Breath metabolomics pipeline for SESI-HRMS fasting study data."""

from __future__ import annotations

import argparse
import ast
import csv
import heapq
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import AutoDateLocator, DateFormatter
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.stats import linregress

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 24,
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "axes.linewidth": 1.5,
    "axes.unicode_minus": False,
})

# --- tunable parameters ---
TRIM_FRONT_PERCENT = 20
TRIM_BACK_PERCENT = 20
PPM_TOL = 5.0
MZ_MIN = 50.0
MZ_MAX = 350.0
MIN_PEAK_INTENSITY = 500
MZ_DECIMALS = 4
CLUSTER_PPM = 5
MIN_SAMPLE_SCAN_DETECTION_FRACTION = 0.80
MIN_SAMPLE_PRESENCE_FRACTION = 0.80
MIN_SAMPLES_PER_SUBJECT = 25
FDR_ALPHA = 0.05
MIN_CONSISTENT_SUBJECTS = 2
TOP_N_PLOTS = 20

BREATH_STYLES = {
    "S1": {"label": "Breath S1", "color": "#1f77b4", "marker": "o"},
    "S2": {"label": "Breath S2", "color": "#ff7f0e", "marker": "s"},
    "S3": {"label": "Breath S3", "color": "#2ca02c", "marker": "^"},
}
SUBTRACT_STYLES = {
    "S1": {"label": "Breath-blank S1", "color": "#084594", "marker": "D"},
    "S2": {"label": "Breath-blank S2", "color": "#a63603", "marker": "v"},
    "S3": {"label": "Breath-blank S3", "color": "#006d2c", "marker": "*"},
}
BLANK_STYLES = {
    "25_03_2026": {"label": "Blank 25 Mar", "color": "#111111", "marker": "X",
                   "linestyle": (0, (1, 2))},
    "04_05_2026": {"label": "Blank 04 May", "color": "#7f7f7f", "marker": "P",
                   "linestyle": (0, (1, 2))},
}
_BREATH_RE = re.compile(
    r"^(?P<date>\d{2}_\d{2}_\d{4})_FASTING_"
    r"(?P<subject>S1|S2|S3)_(?P<sample_id>\d+)_(?P<ion>pos|neg)\.mzML$")
_BLANK_RE = re.compile(
    r"^(?P<date>\d{2}_\d{2}_\d{4})_FASTING_"
    r"B(?P<sample_id>\d+(?:-continued)?)_(?P<ion>pos|neg)\.mzML$")


def safe_name(value: str) -> str:
    value = str(value).strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._,+-]+", "_", value)
    return value.strip("_") or "unknown"


def parse_breath_filename(filename: str) -> dict | None:
    m = _BREATH_RE.match(filename)
    if not m:
        return None
    d = m.groupdict()
    d["sample_id"] = int(d["sample_id"])
    return d


def parse_blank_filename(filename: str) -> dict | None:
    m = _BLANK_RE.match(filename)
    if not m:
        return None
    d = m.groupdict()
    d["subject"] = "Blank"
    d["sample_id"] = int(d["sample_id"])
    return d


# --- mzML i/o ---

def read_mzml_tic(mzml_path: Path, time_shift: float = 0.0) -> pd.DataFrame:
    rt, tic, scans = [], [], []
    try:
        import pymzml
        run = pymzml.run.Reader(str(mzml_path))
        for idx, spec in enumerate(run):
            scans.append(idx + 1)
            rt.append(spec.scan_time_in_minutes() + time_shift)
            tic.append(spec.TIC)
    except Exception:
        pass
    df = pd.DataFrame({"time_min": rt, "TIC": tic, "scan_num": scans})
    t0 = pd.to_datetime(0).value // 10 ** 9
    df["abs_time"] = pd.to_datetime(df.time_min * 60 + t0, unit="s")
    return df


def get_mzml_start_timestamp(mzml_path: Path) -> str:
    try:
        for _event, elem in ET.iterparse(str(mzml_path), events=("start",)):
            tag = elem.tag.split("}", 1)[-1]
            if tag == "run":
                return elem.attrib.get("startTimeStamp", "")
    except Exception as exc:
        return f"ERROR: {exc}"
    return ""


def iter_spectra(mzml_path: Path):
    import pymzml
    run = pymzml.run.Reader(str(mzml_path))
    for idx, spectrum in enumerate(run, start=1):
        try:
            rt_min = float(spectrum.scan_time_in_minutes())
            tic = float(spectrum.TIC or 0.0)
            mz = np.asarray(spectrum.mz, dtype=float)
            intensity = np.asarray(spectrum.i, dtype=float)
        except Exception:
            continue
        if mz.size == 0 or intensity.size == 0:
            continue
        yield idx, rt_min, tic, mz, intensity


def read_exhalion_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, header=None,
                         names=["time", "exhaled_co2", "pressure",
                                "flow_rate", "total_exhaled_volume"])
        df["time_min"] = df.time / 60
        df["abs_time"] = pd.to_datetime(df.time, unit="s")
        df["CO2"] = df["exhaled_co2"]
        df["Flow"] = df["flow_rate"]
        df["Volume"] = df["total_exhaled_volume"]
        return df
    except Exception as exc:
        print(f"  [WARN] Exhalion read failed: {exc}")
        return None


# --- step 2: breath interval extraction ---

def _exhalation_plateaus(exhalion, pressure_thresh=4.0):
    try:
        exh = exhalion[exhalion["pressure"] > pressure_thresh].copy()
        exh["dt"] = exh.time_min.diff()
        starts = [exh.time_min.iloc[0]] + exh[exh["dt"] > 0.1].time_min.tolist()
        inh = exhalion[exhalion["pressure"] < 1].copy()
        inh["dt"] = inh.time_min.diff()
        ends = inh[inh["dt"] > 0.1].time_min.tolist()
        if len(ends) < len(starts):
            ends.append(exhalion.time_min.max())
        return starts[:len(ends)], ends[:len(starts)]
    except Exception:
        return [], []


def _align_tic_to_pressure(tic_df, exhalion):
    try:
        t_tic, i_tic = tic_df.time_min.values, tic_df.TIC.values
        t_ex, p_ex = exhalion.time_min.values, exhalion["pressure"].values
        f = interp1d(t_tic, i_tic, kind="linear", fill_value=0,
                     bounds_error=False)
        corr = correlate(p_ex, f(t_ex), mode="full")
        lags = np.arange(-len(p_ex) + 1, len(p_ex))
        best_lag = lags[np.argmax(corr)]
        return best_lag * np.mean(np.diff(t_ex))
    except Exception:
        return 0.0


def _parse_crop_rule(rule_str):
    if rule_str is None:
        return (None, -2)
    if rule_str in ("None", "nan", ""):
        return (None, -1)
    if rule_str == "Delete":
        return (None, None)
    nums = re.findall(r"\d+\.?\d*", str(rule_str))
    if not nums:
        return (None, -2)
    nums = [float(n) for n in nums]
    if str(rule_str).startswith("[,"):
        return (-np.inf, nums[0])
    if str(rule_str).endswith(",]"):
        return (nums[0], np.inf)
    if len(nums) == 2:
        return (nums[0], nums[1])
    return (None, -2)


def _crop_tic(tic_df, start_sec, end_sec):
    if start_sec is None and end_sec == -2:
        tic_df = tic_df.copy()
        t0 = tic_df.time_min.min()
        tic_df["elapsed_sec"] = (tic_df.time_min - t0) * 60
        return tic_df
    if start_sec is None and end_sec in (-1, None):
        return None
    tic_df = tic_df.copy()
    t0 = tic_df.time_min.min()
    tic_df["elapsed_sec"] = (tic_df.time_min - t0) * 60
    mask = (tic_df.elapsed_sec >= start_sec) & (tic_df.elapsed_sec <= end_sec)
    cropped = tic_df[mask].copy()
    return cropped if len(cropped) > 0 else None


def run_step2_breath_intervals(mzml_dir, exhalion_dir, match_csv, crop_csv,
                                output_dir, ion_mode="pos"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"2_Breath_intervals_trimmed_{ion_mode}.csv"
    out_pdf = output_dir / f"2_Breath_plots_{ion_mode}.pdf"

    crop_rules = {}
    if crop_csv.exists():
        rules_df = pd.read_csv(crop_csv)
        col_map = {
            "S1_pos": ("S1", "pos"), "S1_neg": ("S1", "neg"),
            "S2_pos": ("S2", "pos"), "S2_neg": ("S2", "neg"),
            "S3_pos": ("S3", "pos"), "S3_neg": ("S3", "neg"),
        }
        for _, row in rules_df.iterrows():
            num = int(row["number"])
            for col, (subj, mode) in col_map.items():
                rule = row[col]
                if pd.isna(rule) or str(rule).strip() == "":
                    crop_rules[(subj, mode, num)] = None
                else:
                    crop_rules[(subj, mode, num)] = str(rule).strip()

    df_match = pd.read_csv(match_csv)
    results, interval_rows = [], []

    with PdfPages(str(out_pdf)) as pdf:
        for _, row in df_match.iterrows():
            mzml_file = str(row.mzml_filename).strip()
            txt_file = str(row.txt_filename).strip()
            if ion_mode not in mzml_file.lower():
                continue

            mzml_path = mzml_dir / mzml_file
            txt_path = exhalion_dir / txt_file
            if not mzml_path.exists() or not txt_path.exists():
                print(f"  [SKIP] missing: {mzml_file} / {txt_file}")
                continue

            info = parse_breath_filename(mzml_file)
            if info is None:
                continue
            rule_key = (info["subject"], info["ion"], info["sample_id"])
            crop_rule = crop_rules.get(rule_key)

            print(f"  [{info['subject']}-{info['sample_id']:03d}] {mzml_file}")
            if crop_rule == "Delete":
                continue

            exhalion_full = read_exhalion_file(txt_path)
            tic_raw = read_mzml_tic(mzml_path)
            if exhalion_full is None or len(tic_raw) == 0:
                continue

            start_sec, end_sec = _parse_crop_rule(crop_rule)
            tic_cropped = _crop_tic(tic_raw, start_sec, end_sec)
            if tic_cropped is None:
                continue

            shift = _align_tic_to_pressure(tic_cropped, exhalion_full)
            tic_aligned = read_mzml_tic(mzml_path, time_shift=shift)
            tic_final = _crop_tic(tic_aligned, start_sec, end_sec)
            if tic_final is None:
                continue

            starts, ends = _exhalation_plateaus(exhalion_full)
            plateaus = []
            for s, e in zip(starts, ends):
                dur = e - s
                plateaus.append({
                    "trim_start": s + dur * TRIM_FRONT_PERCENT / 100,
                    "trim_end": e - dur * TRIM_BACK_PERCENT / 100,
                })

            t_min, t_max = tic_final.time_min.min(), tic_final.time_min.max()
            valid = [p for p in plateaus
                     if max(p["trim_start"], t_min) < min(p["trim_end"], t_max)]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(tic_final.time_min, tic_final.TIC, "k-", lw=2.5,
                    label="Aligned TIC")
            ax.set_xlim(t_min, t_max)
            for p in valid:
                ax.axvspan(max(p["trim_start"], t_min),
                           min(p["trim_end"], t_max),
                           color="green", alpha=0.3)
            ax.set_xlabel("Time (min)", fontsize=24, fontweight="bold")
            ax.set_ylabel("TIC Intensity", fontsize=24, fontweight="bold")
            ax.set_title(mzml_file, fontsize=24, fontweight="bold", pad=20)
            ax.legend(fontsize=16)
            ax.tick_params(axis="both", labelsize=22)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            t0_aligned = tic_aligned.time_min.min()
            intervals = []
            for p in valid[:5]:
                intervals.append(
                    f"[{round((p['trim_start'] - t0_aligned) * 60, 1)},"
                    f"{round((p['trim_end'] - t0_aligned) * 60, 1)}]")
            while len(intervals) < 5:
                intervals.append("")
            interval_rows.append([mzml_file] + intervals)

            results.append({
                "filename": mzml_file,
                "detection_time_sec": round(
                    tic_final.elapsed_sec.max() - tic_final.elapsed_sec.min(), 2),
            })

    pd.DataFrame(results).to_csv(
        output_dir / f"2_Breath_detection_summary_{ion_mode}.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(interval_rows,
                 columns=["filename", "interval1", "interval2", "interval3",
                          "interval4", "interval5"]).to_csv(
        out_csv, index=False, encoding="utf-8-sig")
    print(f"  -> {out_csv.name}")
    print(f"  -> {out_pdf.name}")


# --- step 3: blank intervals ---

def run_step3_blank_intervals(mzml_dir, time_choose_csv, output_dir,
                               ion_mode="pos"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"3_Blank_{ion_mode}_retained_intervals.csv"
    out_pdf = output_dir / f"3_platform_plots_{ion_mode}_Background.pdf"

    time_rules = pd.read_csv(time_choose_csv, encoding="utf-8-sig")
    time_rules["sample_id"] = time_rules["sample_id"].astype(int)
    time_rules = time_rules.set_index("sample_id")

    def _get_crop_rule(sample_id, date, ion):
        col = f"{date}_{ion}"
        if sample_id not in time_rules.index or col not in time_rules.columns:
            return ""
        value = time_rules.loc[sample_id, col]
        return "" if pd.isna(value) else str(value).strip()

    files = sorted(p for p in mzml_dir.rglob("*")
                   if p.suffix.lower() == ".mzml"
                   and p.name.lower().endswith(f"_{ion_mode}.mzml"))
    print(f"  Found {len(files)} {ion_mode} blank mzML files")

    import pymzml
    final_rows = []
    with PdfPages(str(out_pdf)) as pdf:
        for idx, file in enumerate(files, 1):
            name = file.stem
            print(f"  [{idx}/{len(files)}] {name}")
            try:
                info = parse_blank_filename(file.name)
                if info is None:
                    continue
                crop_rule = _get_crop_rule(info["sample_id"], info["date"],
                                           info["ion"])
                start_time_str = get_mzml_start_timestamp(file)

                rt_all, tic_raw = [], []
                run = pymzml.run.Reader(str(file))
                for spec in run:
                    if spec.ms_level == 1:
                        rt_all.append(spec.scan_time_in_minutes())
                        tic_raw.append(sum(spec.i))
                rt_all = np.array(rt_all)
                tic_raw = np.array(tic_raw)
                if len(rt_all) == 0:
                    continue

                start_r, end_r = _parse_crop_rule(crop_rule if crop_rule else None)
                raw_start = float(np.min(rt_all))
                raw_end = float(np.max(rt_all))
                keep_start, keep_end = raw_start, raw_end
                keep_mask = np.ones(len(rt_all), dtype=bool)
                if start_r is not None or (end_r not in (None, -1, -2)):
                    keep_start = raw_start if start_r is None else max(
                        raw_start, start_r)
                    keep_end = raw_end if end_r is None else min(raw_end, end_r)
                    if keep_end >= keep_start:
                        keep_mask = (rt_all >= keep_start) & (rt_all <= keep_end)
                    else:
                        keep_mask = np.zeros(len(rt_all), dtype=bool)

                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(rt_all, tic_raw, color="lightgray", lw=1.5,
                        label="Raw TIC")
                ax.plot(rt_all[keep_mask], tic_raw[keep_mask], "blue", lw=2,
                        label="Retained TIC")
                k_start = (keep_start if not np.isnan(keep_start)
                           else float(np.min(rt_all)))
                k_end = (keep_end if not np.isnan(keep_end)
                         else float(np.max(rt_all)))
                ax.axvspan(k_start, k_end, alpha=0.18, color="green",
                           label="Retained region")
                ax.set_title(f"TIC Chromatogram\n{name}",
                             fontsize=24, fontweight="bold", pad=16)
                ax.set_xlabel("Retention Time (min)", fontsize=24,
                              fontweight="bold")
                ax.set_ylabel("Intensity", fontsize=24, fontweight="bold")
                ax.legend(fontsize=16)
                ax.tick_params(axis="both", labelsize=22)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                final_rows.append({
                    "filename": file.name, "date": info["date"],
                    "sample_id": info["sample_id"], "ion": info["ion"],
                    "status": "ok" if int(np.sum(keep_mask)) > 0
                    else "empty_after_crop",
                    "crop_rule": crop_rule,
                    "raw_rt_start_min": float(np.min(rt_all)),
                    "raw_rt_end_min": float(np.max(rt_all)),
                    "retained_rt_start_min": keep_start,
                    "retained_rt_end_min": keep_end,
                    "n_scans_raw": int(len(rt_all)),
                    "n_scans_retained": int(np.sum(keep_mask)),
                    "mzml_start_time": start_time_str,
                })
            except Exception as exc:
                print(f"    FAILED: {exc}")
                final_rows.append({"filename": file.name,
                                   "status": f"failed: {exc}"})

    pd.DataFrame(final_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  -> {out_csv.name}")
    print(f"  -> {out_pdf.name}")


# --- step 4: detection times ---

def run_step4_detection_times(breath_intervals_csv, blank_intervals_csv,
                               breath_mzml_dir, blank_mzml_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    out_breath = output_dir / "4_Breath_international_detection_time.csv"
    if breath_intervals_csv.exists():
        rows = []
        with open(breath_intervals_csv, "r", newline="",
                  encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                fname = row.get("filename", "").strip()
                if not fname:
                    continue
                mzml = breath_mzml_dir / fname
                ts = get_mzml_start_timestamp(mzml) if mzml.exists() else ""
                iso_time = date_only = time_only = ""
                if ts and not ts.startswith("ERROR:"):
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        iso_time = dt.isoformat()
                        date_only = dt.strftime("%Y-%m-%d")
                        time_only = dt.strftime("%H:%M:%S")
                    except ValueError:
                        iso_time = ts
                info = parse_breath_filename(fname) or {}
                status = "ok" if ts and not ts.startswith("ERROR:") else (
                    "missing_mzml" if not ts else ts)
                rows.append({
                    "filename": fname,
                    "source_interval_csv": breath_intervals_csv.name,
                    "filename_date": info.get("date", ""),
                    "subject": info.get("subject", ""),
                    "sample_id": info.get("sample_id", ""),
                    "ion": info.get("ion", ""),
                    "international_detection_time": iso_time,
                    "international_detection_date": date_only,
                    "international_detection_time_only": time_only,
                    "raw_startTimeStamp": ts,
                    "mzml_path": str(mzml) if mzml.exists() else "",
                    "status": status,
                })
        pd.DataFrame(rows).to_csv(out_breath, index=False, encoding="utf-8-sig")
        print(f"  -> {out_breath.name}  ({len(rows)} rows)")

    out_blank = output_dir / "4_Blank_international_detection_time.csv"
    blank_files = sorted(p for p in blank_mzml_dir.rglob("*")
                         if p.suffix.lower() == ".mzml")
    rows = []
    for mzml in blank_files:
        ts = get_mzml_start_timestamp(mzml)
        iso_time = date_only = time_only = ""
        if ts and not ts.startswith("ERROR:"):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                iso_time = dt.isoformat()
                date_only = dt.strftime("%Y-%m-%d")
                time_only = dt.strftime("%H:%M:%S")
            except ValueError:
                iso_time = ts
        info = parse_blank_filename(mzml.name) or {}
        status = "ok" if ts and not ts.startswith("ERROR:") else (
            "missing_mzml" if not ts else ts)
        rows.append({
            "filename": mzml.name,
            "filename_date": info.get("date", ""),
            "sample_id": info.get("sample_id", ""),
            "ion": info.get("ion", ""),
            "international_detection_time": iso_time,
            "international_detection_date": date_only,
            "international_detection_time_only": time_only,
            "raw_startTimeStamp": ts,
            "mzml_path": str(mzml),
            "status": status,
        })
    pd.DataFrame(rows).to_csv(out_blank, index=False, encoding="utf-8-sig")
    print(f"  -> {out_blank.name}  ({len(rows)} rows)")


# --- step 5: known-target trends ---

def _load_targets(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    mz_col = "RMM" if "RMM" in df.columns else df.columns[4]
    out = df.rename(columns={
        "Polarity": "ion", "VOC": "compound", mz_col: "target_mz"}).copy()
    out["ion"] = out["ion"].astype(str).str.strip()
    out["target_mz"] = pd.to_numeric(out["target_mz"], errors="coerce")
    out = out[out["ion"].isin(["pos", "neg"])].dropna(subset=["target_mz"])
    compound = (out["compound"].copy() if "compound" in out.columns
                else pd.Series(np.nan, index=out.index))
    compound = compound.map(lambda v: v.strip() if isinstance(v, str) else v)
    compound = compound.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    fallback = out["ion"] + "_mz_" + out["target_mz"].round(5).astype(str)
    out["compound"] = compound.where(compound.notna(), fallback)
    out["target_id"] = out["ion"] + "_" + out["compound"].map(safe_name)
    return out[["target_id", "ion", "compound", "target_mz"]].reset_index(drop=True)


def _load_breath_intervals_for_step5(processed_dir, ion_mode="pos"):
    path = processed_dir / "step2_intervals" / f"2_Breath_intervals_trimmed_{ion_mode}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    parts = []
    for _, row in df.iterrows():
        info = parse_breath_filename(str(row["filename"]))
        if info is None:
            continue
        for col in [c for c in df.columns if c.startswith("interval")]:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                interval = ast.literal_eval(str(val).strip())
            except (ValueError, SyntaxError):
                continue
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            start, end = float(interval[0]), float(interval[1])
            if end <= start:
                continue
            parts.append({
                "sample_type": "breath", "filename": row["filename"],
                "ion": ion_mode, "subject": info["subject"],
                "sample_id": info["sample_id"], "date": info["date"],
                "interval_index": int(col.replace("interval", "")),
                "rt_start_min": start / 60.0, "rt_end_min": end / 60.0,
                "interval_unit_source": "seconds",
            })
    return pd.DataFrame(parts)


def _load_blank_intervals_for_step5(processed_dir):
    parts = []
    for ion in ["pos", "neg"]:
        path = processed_dir / "step3_blanks" / f"3_Blank_{ion}_retained_intervals.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[df["status"].eq("ok")].copy()
        for _, row in df.iterrows():
            info = parse_blank_filename(str(row["filename"]))
            parts.append({
                "sample_type": "blank", "filename": row["filename"],
                "ion": ion, "subject": "Blank",
                "sample_id": info["sample_id"] if info else "",
                "date": row.get("date", ""), "interval_index": 1,
                "rt_start_min": float(row["retained_rt_start_min"]),
                "rt_end_min": float(row["retained_rt_end_min"]),
                "interval_unit_source": "minutes",
            })
    return pd.DataFrame(parts)


def _load_detection_times_for_step5(processed_dir):
    mapping = {
        "breath": processed_dir / "step4_detection" / "4_Breath_international_detection_time.csv",
        "blank": processed_dir / "step4_detection" / "4_Blank_international_detection_time.csv",
    }
    frames = []
    for sample_type, path in mapping.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[df["status"].eq("ok")].copy()
        df["sample_type"] = sample_type
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["detection_time"] = pd.to_datetime(
        out["international_detection_time"], errors="coerce", utc=True)
    out["detection_time_local"] = out["detection_time"].dt.tz_convert(None)
    out["time_only"] = pd.to_datetime(
        "1970-01-01 " + out["detection_time"].dt.strftime("%H:%M:%S"),
        errors="coerce")

    def _extract_date(fname):
        info = parse_breath_filename(fname) or parse_blank_filename(fname)
        return (info or {}).get("date", "")

    out["date"] = out["filename"].map(_extract_date)
    return out.drop_duplicates(["sample_type", "filename"])


def _extract_file_targets(sample, sample_intervals, targets, ppm_tol):
    mzml_path = Path(str(sample["mzml_path"]))
    if not mzml_path.exists():
        return [], "missing_mzml"
    intervals = sample_intervals[
        ["interval_index", "rt_start_min", "rt_end_min"]].to_dict("records")
    target_rows = targets[targets["ion"].eq(sample["ion"])].to_dict("records")
    scan_rows = []
    try:
        for scan_num, rt_min, tic, mz, intensity in iter_spectra(mzml_path):
            matched = [it for it in intervals
                       if it["rt_start_min"] <= rt_min <= it["rt_end_min"]]
            if not matched:
                continue
            for target in target_rows:
                tmz = float(target["target_mz"])
                tol = tmz * ppm_tol * 1e-6
                mask = (mz >= tmz - tol) & (mz <= tmz + tol)
                if np.any(mask):
                    best_idx = int(np.argmax(intensity[mask]))
                    absolute = float(intensity[mask][best_idx])
                    matched_mz = float(mz[mask][best_idx])
                else:
                    absolute = 0.0
                    matched_mz = np.nan
                relative = absolute / tic if tic > 0 else np.nan
                for it in matched:
                    scan_rows.append({
                        "sample_type": sample["sample_type"],
                        "filename": sample["filename"],
                        "ion": sample["ion"],
                        "subject": sample["subject"],
                        "sample_id": sample["sample_id"],
                        "date": sample["date"],
                        "interval_index": it["interval_index"],
                        "scan_num": scan_num, "rt_min": rt_min, "tic": tic,
                        "compound": target["compound"],
                        "target_id": target["target_id"],
                        "target_mz": tmz, "ppm_tol": ppm_tol,
                        "matched_mz": matched_mz,
                        "ppm_error": ((matched_mz - tmz) / tmz * 1e6
                                      if not np.isnan(matched_mz) else np.nan),
                        "absolute_intensity": absolute,
                        "relative_intensity": relative,
                        "detection_time": sample["detection_time"],
                        "detection_time_local": sample["detection_time_local"],
                        "time_only": sample["time_only"],
                        "mzml_path": str(mzml_path),
                    })
        return scan_rows, "ok"
    except Exception as exc:
        return scan_rows, f"error: {exc}"


def _percent_change_from_first(values):
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    valid = valid[valid.ne(0)]
    if valid.empty:
        return out
    first = float(valid.iloc[0])
    out.loc[values.index] = (values - first) / abs(first) * 100.0
    return out


def _make_blank_reference(summary):
    blank = summary[summary["sample_type"].eq("blank")].copy()
    if blank.empty:
        return pd.DataFrame()
    blank["blank_rank"] = blank.groupby(
        ["ion", "compound"])["detection_time_local"].rank(method="first")
    return blank


def _attach_blank_subtraction(summary):
    breath = summary[summary["sample_type"].eq("breath")].copy()
    blank = _make_blank_reference(summary)
    if breath.empty or blank.empty:
        return pd.DataFrame()

    blank_cols = ["ion", "compound", "target_id", "target_mz",
                  "detection_time_local", "time_only",
                  "mean_relative_intensity", "filename", "sample_id"]
    blank = blank[blank_cols].rename(columns={
        "detection_time_local": "blank_detection_time_local",
        "time_only": "blank_time_only",
        "mean_relative_intensity": "blank_mean_relative_intensity",
        "filename": "blank_filename",
        "sample_id": "blank_sample_id",
    })

    rows = []
    for _, row in breath.iterrows():
        candidates = blank[
            blank["ion"].eq(row["ion"])
            & blank["compound"].eq(row["compound"])
            & blank["target_id"].eq(row["target_id"])].copy()
        if candidates.empty:
            continue
        deltas = (candidates["blank_detection_time_local"]
                  - row["detection_time_local"]).abs()
        nearest = candidates.loc[deltas.idxmin()]
        out = row.to_dict()
        out.update(nearest.to_dict())
        out["relative_minus_blank"] = (
            row["mean_relative_intensity"]
            - nearest["blank_mean_relative_intensity"])
        rows.append(out)
    return pd.DataFrame(rows)


def run_step5_known_targets(processed_dir, target_csv, output_dir,
                              ion_mode="pos"):
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(target_csv)
    intervals = pd.concat([
        _load_breath_intervals_for_step5(processed_dir, ion_mode),
        _load_blank_intervals_for_step5(processed_dir),
    ], ignore_index=True)
    if intervals.empty:
        print("  [WARN] No interval data -- run step 2/3 first.")
        return

    detection = _load_detection_times_for_step5(processed_dir)
    if detection.empty:
        print("  [WARN] No detection times -- run step 4 first.")
        return

    intervals = intervals.merge(
        detection, on=["sample_type", "filename", "ion"], how="left",
        suffixes=("", "_time"))
    for col in ["subject", "date"]:
        time_col = f"{col}_time"
        if time_col in intervals.columns:
            intervals[col] = intervals[col].fillna(intervals[time_col])

    samples = (
        intervals.dropna(subset=["mzml_path"])
        .drop_duplicates(["sample_type", "filename", "ion", "mzml_path"])
        .sort_values(["sample_type", "ion", "filename"])
        .reset_index(drop=True))

    scan_rows, status_rows = [], []
    for idx, sample in samples.iterrows():
        sample_intervals = intervals[
            intervals["sample_type"].eq(sample["sample_type"])
            & intervals["filename"].eq(sample["filename"])
            & intervals["ion"].eq(sample["ion"])]
        rows, status = _extract_file_targets(
            sample, sample_intervals, targets, PPM_TOL)
        scan_rows.extend(rows)
        status_rows.append({
            "sample_type": sample["sample_type"],
            "filename": sample["filename"], "ion": sample["ion"],
            "n_scan_target_rows": len(rows), "status": status,
        })
        print(f"  [{idx + 1}/{len(samples)}] {sample['filename']} -> {status}")

    scan_df = pd.DataFrame(scan_rows)
    status_df = pd.DataFrame(status_rows)
    if scan_df.empty:
        print("  [WARN] No scan-target rows produced.")
        return

    detected = scan_df["absolute_intensity"].gt(0)
    scan_df["detected_scan_num"] = scan_df["scan_num"].where(detected)
    scan_df["rel_int_detected"] = scan_df["relative_intensity"].where(detected)

    keys = ["sample_type", "filename", "ion", "subject", "sample_id", "date",
            "compound", "target_id", "target_mz",
            "detection_time", "detection_time_local", "time_only", "mzml_path"]
    summary = (
        scan_df.groupby(keys, dropna=False)
        .agg(n_scans=("scan_num", "nunique"),
             n_detected_scans=("detected_scan_num", "nunique"),
             n_intervals=("interval_index", "nunique"),
             mean_relative_intensity=("rel_int_detected", "mean"),
             median_relative_intensity=("rel_int_detected", "median"),
             detection_fraction=("absolute_intensity",
                                 lambda v: float(np.mean(np.asarray(v) > 0))),
             mean_matched_mz=("matched_mz", "mean"),
             mean_ppm_error=("ppm_error", "mean"))
        .reset_index())

    subtract = (_attach_blank_subtraction(summary)
                if not summary.empty else pd.DataFrame())

    targets.to_csv(output_dir / "targets_used.csv", index=False,
                   encoding="utf-8-sig")
    intervals.to_csv(output_dir / "intervals_with_detection_time.csv",
                     index=False, encoding="utf-8-sig")
    status_df.to_csv(output_dir / "processing_status.csv", index=False,
                     encoding="utf-8-sig")
    scan_df.to_csv(output_dir / "scan_peak_values_long.csv", index=False,
                   encoding="utf-8-sig")
    summary.to_csv(output_dir / "file_target_intensity_summary.csv",
                   index=False, encoding="utf-8-sig")
    if not subtract.empty:
        subtract.to_csv(output_dir / "breath_nearest_blank_subtracted.csv",
                        index=False, encoding="utf-8-sig")

    targets_uniq = (
        summary[["ion", "compound", "target_mz"]]
        .drop_duplicates().sort_values(["ion", "target_mz", "compound"]))

    val_col = "mean_relative_intensity"
    subtract_col = "relative_minus_blank"
    pdf_path = output_dir / "known_mz_relative_trends.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        for _, tgt in targets_uniq.iterrows():
            ion = str(tgt["ion"])
            compound = str(tgt["compound"])

            fig, (ax, ax_pct) = plt.subplots(
                2, 1, figsize=(14, 16),
                gridspec_kw={"height_ratios": [2.2, 1]}, sharex=False)

            blank_df = summary[
                summary["sample_type"].eq("blank")
                & summary["ion"].eq(ion)
                & summary["compound"].eq(compound)].copy()
            breath_df = summary[
                summary["sample_type"].eq("breath")
                & summary["ion"].eq(ion)
                & summary["compound"].eq(compound)].copy()

            for date_grp, style in BLANK_STYLES.items():
                dg = blank_df[blank_df["date"].eq(date_grp)].sort_values(
                    "time_only")
                if dg.empty:
                    continue
                ax.plot(dg["time_only"], dg[val_col],
                        label=style["label"], color=style["color"],
                        marker=style["marker"], linewidth=2.2, markersize=7,
                        linestyle=style["linestyle"],
                        markerfacecolor="white", markeredgewidth=1.4, alpha=0.95)

            for subj, style in BREATH_STYLES.items():
                dg = breath_df[breath_df["subject"].eq(subj)].sort_values(
                    "time_only")
                if dg.empty:
                    continue
                ax.plot(dg["time_only"], dg[val_col],
                        label=style["label"], color=style["color"],
                        marker=style["marker"], linewidth=3.2, markersize=6.5)

            if not subtract.empty:
                sub_df = subtract[
                    subtract["ion"].eq(ion)
                    & subtract["compound"].eq(compound)].copy()
                for subj, style in SUBTRACT_STYLES.items():
                    dg = sub_df[sub_df["subject"].eq(subj)].sort_values(
                        "time_only")
                    if dg.empty:
                        continue
                    ax.plot(dg["time_only"], dg[subtract_col],
                            label=style["label"], color=style["color"],
                            marker=style["marker"], linewidth=3.0,
                            markersize=6.5, linestyle=(0, (6, 3)),
                            markerfacecolor="white", markeredgewidth=1.3,
                            alpha=0.95)

            pct_plotted = False
            for subj, style in BREATH_STYLES.items():
                dg = breath_df[breath_df["subject"].eq(subj)].sort_values(
                    "time_only").copy()
                if dg.empty:
                    continue
                dg["pct_change"] = _percent_change_from_first(dg[val_col])
                ax_pct.plot(dg["time_only"], dg["pct_change"],
                            label=style["label"], color=style["color"],
                            marker=style["marker"], linewidth=3.0,
                            markersize=6.5)
                pct_plotted = True

            tgt_mz = breath_df["target_mz"].dropna()
            if tgt_mz.empty:
                tgt_mz = blank_df["target_mz"].dropna()
            title_mz = f"{tgt_mz.iloc[0]:.5f}" if not tgt_mz.empty else "NA"

            ax.axhline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.4)
            ax.xaxis.set_major_locator(AutoDateLocator(minticks=6, maxticks=12))
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax.set_xlabel("International detection time", fontsize=24,
                          fontweight="bold")
            ax.set_ylabel("Relative intensity / TIC", fontsize=24,
                          fontweight="bold")
            ax.set_title(f"{compound} ({ion}, m/z={title_mz})",
                         fontsize=24, fontweight="bold", pad=18)
            ax.legend(loc="best", fontsize=16, ncol=2)
            ax.tick_params(axis="both", labelsize=22)
            ax.tick_params(axis="x", labelrotation=45)

            ax_pct.axhline(0, color="black", linewidth=1.2, linestyle="-",
                           alpha=0.4)
            ax_pct.xaxis.set_major_locator(
                AutoDateLocator(minticks=6, maxticks=12))
            ax_pct.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax_pct.set_xlabel("International detection time", fontsize=24,
                              fontweight="bold")
            ax_pct.set_ylabel("Breath change (%)", fontsize=24,
                              fontweight="bold")
            ax_pct.set_title("Breath percent change from first point",
                             fontsize=24, fontweight="bold", pad=10)
            if pct_plotted:
                ax_pct.legend(loc="best", fontsize=16, ncol=3)
            ax_pct.tick_params(axis="both", labelsize=22)
            ax_pct.tick_params(axis="x", labelrotation=45)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"  -> {pdf_path.name}")

    summary_path = output_dir / "run_summary.csv"
    pd.DataFrame([
        {"metric": "targets", "value": len(targets)},
        {"metric": "samples", "value": len(samples)},
        {"metric": "scan_target_rows", "value": len(scan_df)},
        {"metric": "summary_rows", "value": len(summary)},
        {"metric": "breath_blank_subtracted_rows", "value": len(subtract)},
    ]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  -> {summary_path.name}")


# --- step 6: untargeted screening ---

def _hierarchical_cluster_1d(values, distance_threshold):
    n = len(values)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    start = list(range(n))
    end = list(range(n))
    active = [True] * n
    left_nbr = [-1] + list(range(n - 1))
    right_nbr = list(range(1, n)) + [-1]
    version = [0] * n
    queue: list[tuple[float, int, int, int, int]] = []

    def push_pair(li, ri):
        if li == -1 or ri == -1:
            return
        heapq.heappush(queue, (
            float(values[end[ri]] - values[start[li]]),
            li, ri, version[li], version[ri]))

    for i in range(n - 1):
        push_pair(i, i + 1)

    while queue:
        dist, li, ri, lv, rv = heapq.heappop(queue)
        if dist > distance_threshold:
            break
        if (not active[li] or not active[ri] or right_nbr[li] != ri
                or version[li] != lv or version[ri] != rv):
            continue
        active[ri] = False
        end[li] = end[ri]
        nxt = right_nbr[ri]
        right_nbr[li] = nxt
        if nxt != -1:
            left_nbr[nxt] = li
        version[li] += 1
        push_pair(left_nbr[li], li)
        push_pair(li, nxt)

    labels = np.zeros(n, dtype=int)
    lbl, idx = 0, 0
    while idx != -1:
        if active[idx]:
            lbl += 1
            labels[start[idx]:end[idx] + 1] = lbl
        idx = right_nbr[idx]
    return labels
 
def _build_mz_clusters(scan_peak_df, ppm):
    rows = []
    for ion, ion_df in scan_peak_df.groupby("ion"):
        mz_vals = np.sort(
            ion_df["mz_4dp"].dropna().unique().astype(float))
        if mz_vals.size == 0:
            continue
        log_mz = np.log(mz_vals)
        labels = _hierarchical_cluster_1d(
            log_mz,
            math.log1p(ppm * 1e-6)
        )
        for idx, lbl in enumerate(np.unique(labels), start=1):
            members = mz_vals[labels == lbl]
            member_df = ion_df[ion_df["mz_4dp"].isin(members)]
            mz_intensity = (
                member_df.groupby("mz_4dp")["intensity"]
                .sum())
            cmz = float(mz_intensity.idxmax())
            fid = f"{ion}_mz_{cmz:.4f}_{idx:05d}"
            for member in members:
                rows.append({
                    "ion": ion,
                    "mz_4dp": float(member),
                    "feature_id": fid,
                    "cluster_mz": cmz})
    return pd.DataFrame(rows)

def _build_sample_feature_matrix(scan_peak_df, clusters, sample_meta, qc):
    peaks = scan_peak_df.merge(clusters, on=["ion", "mz_4dp"], how="left")
    scan_feat = peaks.groupby(
        ["filename", "feature_id", "cluster_mz", "ion", "scan_num"],
        as_index=False).agg(
            scan_relative_intensity=("relative_intensity", "max"),
            scan_raw_intensity=("intensity", "max"),
            mz_mean=("mz_value", "mean"))
    sample_sums = scan_feat.groupby(
        ["filename", "feature_id", "cluster_mz", "ion"], as_index=False).agg(
            sum_relative_intensity=("scan_relative_intensity", "sum"),
            sum_raw_intensity=("scan_raw_intensity", "sum"),
            n_detected_scans=("scan_num", "nunique"),
            mz_mean=("mz_mean", "mean"))
    sample_sums = sample_sums.merge(
        qc[["filename", "n_platform_scans"]], on="filename", how="left")
    sample_sums["detection_fraction"] = (
        sample_sums["n_detected_scans"]
        / sample_sums["n_platform_scans"].clip(lower=1))
    sample_sums["intensity"] = (
        sample_sums["sum_relative_intensity"]
        / sample_sums["n_detected_scans"].clip(lower=1))
    keys = sample_meta[["filename", "ion", "subject", "sample_id", "date",
                         "detection_time_local", "time_only",
                         "time_hours"]].drop_duplicates("filename")
    matrix = sample_sums.merge(keys.drop(columns=["ion"]), on="filename",
                               how="left")
    return matrix.sort_values(
        ["ion", "feature_id", "subject", "time_hours", "sample_id"]
    ).reset_index(drop=True)


def _filter_by_presence(sample_feature, sample_meta):
    if sample_feature.empty:
        return sample_feature, pd.DataFrame()
    total = (sample_meta[["filename", "ion"]].drop_duplicates()
             .groupby("ion").size().rename("n_total_samples").reset_index())
    eligible = sample_feature[
        sample_feature["detection_fraction"].ge(
            MIN_SAMPLE_SCAN_DETECTION_FRACTION)].copy()
    if eligible.empty:
        return eligible, pd.DataFrame()
    presence = (
        eligible.groupby(["ion", "feature_id", "cluster_mz"], as_index=False)
        .agg(n_present_samples=("filename", "nunique"),
             min_detection_fraction=("detection_fraction", "min"),
             median_detection_fraction=("detection_fraction", "median"))
        .merge(total, on="ion", how="left"))
    presence["presence_fraction"] = (
        presence["n_present_samples"]
        / presence["n_total_samples"].clip(lower=1))
    presence["passes"] = presence["presence_fraction"].ge(
        MIN_SAMPLE_PRESENCE_FRACTION)
    kept = set(presence.loc[presence["passes"], "feature_id"])
    filtered = eligible[eligible["feature_id"].isin(kept)].copy()
    return filtered, presence


def _bh_adjust(pvalues):
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out.loc[order] = adj
    return out


def _regression_per_subject(sample_feature):
    rows = []
    for (subject, fid), grp in sample_feature.groupby(
            ["subject", "feature_id"]):
        grp = grp.sort_values("time_hours")
        if len(grp) < MIN_SAMPLES_PER_SUBJECT or grp["time_hours"].nunique() < 2:
            continue
        x = grp["time_hours"].to_numpy(dtype=float)
        y = grp["intensity"].to_numpy(dtype=float)
        if np.nanstd(y) == 0:
            rows.append({
                "subject": subject, "feature_id": fid,
                "ion": grp["ion"].iloc[0],
                "cluster_mz": grp["cluster_mz"].iloc[0],
                "n_points": len(grp),
                "slope": 0.0, "intercept": float(y[0]), "rvalue": 0.0,
                "pvalue": 1.0, "stderr": 0.0,
                "mean_intensity": float(np.mean(y)),
                "first_intensity": float(y[0]),
                "last_intensity": float(y[-1]),
                "percent_change": 0.0,
            })
            continue
        result = linregress(x, y)
        first, last = y[0], y[-1]
        pct = np.nan if first == 0 else (last - first) / abs(first) * 100
        rows.append({
            "subject": subject, "feature_id": fid,
            "ion": grp["ion"].iloc[0],
            "cluster_mz": grp["cluster_mz"].iloc[0],
            "n_points": len(grp),
            "slope": float(result.slope),
            "intercept": float(result.intercept),
            "rvalue": float(result.rvalue),
            "pvalue": float(result.pvalue),
            "stderr": float(result.stderr),
            "mean_intensity": float(np.mean(y)),
            "first_intensity": float(first),
            "last_intensity": float(last),
            "percent_change": float(pct) if not np.isnan(pct) else np.nan,
        })
    reg = pd.DataFrame(rows)
    if reg.empty:
        return reg
    reg["pvalue_bh"] = _bh_adjust(reg["pvalue"])
    reg["significant_bh"] = reg["pvalue_bh"] < FDR_ALPHA
    reg["trend_dir"] = np.sign(reg["slope"]).astype(int)
    reg["trend_direction"] = np.where(
        reg["trend_dir"] > 0, "increase",
        np.where(reg["trend_dir"] < 0, "decrease", "flat"))
    return reg.sort_values(["pvalue_bh", "pvalue"]).reset_index(drop=True)


def _summarize_consistent_trends(regression):
    if regression.empty:
        return pd.DataFrame(), pd.DataFrame()
    sig = regression[
        regression["significant_bh"] & regression["trend_dir"].ne(0)].copy()
    kept, summary_rows = [], []
    for fid, grp in sig.groupby("feature_id", sort=False):
        counts = grp["trend_dir"].value_counts()
        if len(counts) != 1:
            continue
        direction = int(counts.index[0])
        if int(counts.iloc[0]) < MIN_CONSISTENT_SUBJECTS:
            continue
        kept.append(grp.copy())
        summary_rows.append({
            "feature_id": fid,
            "ion": grp["ion"].iloc[0],
            "cluster_mz": grp["cluster_mz"].iloc[0],
            "n_subjects": len(grp),
            "trend_direction": "increase" if direction > 0 else "decrease",
            "min_pvalue_bh": grp["pvalue_bh"].min(),
            "max_pvalue_bh": grp["pvalue_bh"].max(),
            "mean_slope": grp["slope"].mean(),
            "median_slope": grp["slope"].median(),
            "mean_abs_rvalue": grp["rvalue"].abs().mean(),
            "mean_intensity": grp["mean_intensity"].mean(),
            "max_abs_percent_change": grp["percent_change"].abs().max(),
        })
    kept_df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["rank_score"] = (
            -np.log10(summary["max_pvalue_bh"].clip(lower=1e-300))
            + summary["mean_abs_rvalue"].fillna(0) * 5
            + np.log10(summary["max_abs_percent_change"].fillna(0).abs() + 1))
        summary = summary.sort_values(
            ["rank_score", "mean_abs_rvalue"], ascending=False)
    return kept_df, summary

def _volcano_plot(regression, results_dir):
    if regression.empty:
        return
    feat_stats = regression.groupby(
        ["feature_id", "ion", "cluster_mz"], as_index=False
    ).agg(
        n_subject_regressions=("subject", "nunique"),
        n_significant_subjects=("significant_bh", "sum"),
        max_slope=("slope", "max"),
        max_pvalue_bh=("pvalue_bh", "max"),
        max_abs_rvalue=("rvalue", lambda x: x.abs().max()))
    feat_stats["neg_log10_max_pvalue_bh"] = -np.log10(
        feat_stats["max_pvalue_bh"].clip(lower=1e-300))
    feat_stats["trend_direction"] = np.where(
        feat_stats["max_slope"] > 0, "increase",
        np.where(feat_stats["max_slope"] < 0, "decrease", "flat"))
    feat_stats["significant_any"] = feat_stats["n_significant_subjects"].gt(0)
    feat_stats["group"] = np.where(
        feat_stats["significant_any"],
        feat_stats["trend_direction"],
        "not_significant")
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = {"increase": "#d62728", "decrease": "#1f77b4", "flat": "#7f7f7f", "not_significant": "#bdbdbd"}
    for grp_name, grp_df in feat_stats.groupby("group"):
        ax.scatter(grp_df["max_slope"],grp_df["neg_log10_max_pvalue_bh"], s=45, c=colors.get(grp_name, "#7f7f7f"), 
                   alpha=0.78 if grp_name != "not_significant" else 0.45, edgecolors="none", label=grp_name.replace("_", " "))
    ax.axhline(-np.log10(FDR_ALPHA), color="black", linewidth=1.2, linestyle=(0, (6, 4)),  alpha=0.7)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("Maximum slope across subjects", fontsize=22, fontweight="bold")
    ax.set_ylabel("-log10(max BH-adjusted p-value)", fontsize=22, fontweight="bold")
    ax.set_title("Feature significance volcano plot", fontsize=22, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="best", fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "07_volcano_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_step6_unknown_screening(processed_dir, output_dir, ion_mode="pos"):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots_linear_regression"
    plots_dir.mkdir(exist_ok=True)

    intervals_path = (processed_dir / "step2_intervals"
                      / f"2_Breath_intervals_trimmed_{ion_mode}.csv")
    if not intervals_path.exists():
        print(f"  [WARN] {intervals_path} not found -- run step 2 first.")
        return
    df_int = pd.read_csv(intervals_path, encoding="utf-8-sig")
    interval_cols = [c for c in df_int.columns if c.startswith("interval")]
    interval_rows = []
    for _, row in df_int.iterrows():
        info = parse_breath_filename(str(row["filename"]))
        if info is None:
            continue
        for col in interval_cols:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                interval = ast.literal_eval(str(val).strip())
            except (ValueError, SyntaxError):
                continue
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            s, e = float(interval[0]) / 60.0, float(interval[1]) / 60.0
            if e <= s:
                continue
            interval_rows.append({
                "filename": row["filename"], "ion": ion_mode,
                "subject": info["subject"],
                "sample_id": info["sample_id"], "date": info["date"],
                "interval_index": int(col.replace("interval", "")),
                "rt_start_min": s, "rt_end_min": e,
            })
    intervals = pd.DataFrame(interval_rows)
    if intervals.empty:
        print("  [WARN] No intervals parsed -- check step 2 output.")
        return

    det_path = (processed_dir / "step4_detection"
                / "4_Breath_international_detection_time.csv")
    if not det_path.exists():
        print("  [WARN] Detection-time CSV not found -- run step 4 first.")
        return
    det = pd.read_csv(det_path, encoding="utf-8-sig")
    det = det[det["status"].eq("ok")].copy()
    det["subject"] = det["filename"].map(
        lambda v: parse_breath_filename(str(v))).map(
        lambda d: d["subject"] if d else None)
    det["detection_time"] = pd.to_datetime(
        det["international_detection_time"], errors="coerce", utc=True)
    det["detection_time_local"] = det["detection_time"].dt.tz_convert(None)
    det["time_only"] = pd.to_datetime(
        "1970-01-01 " + det["detection_time"].dt.strftime("%H:%M:%S"),
        errors="coerce")
    det["time_hours"] = (det["time_only"].dt.hour
                         + det["time_only"].dt.minute / 60
                         + det["time_only"].dt.second / 3600)

    samples = (
        intervals.drop_duplicates(
            ["filename", "ion", "subject", "sample_id", "date"])
        .merge(det[["filename", "mzml_path", "detection_time",
                     "detection_time_local", "time_only", "time_hours"]],
               on="filename", how="left")
        .dropna(subset=["mzml_path"])
        .sort_values(["ion", "date", "subject", "sample_id"])
        .reset_index(drop=True))

    scan_parts, qc_rows = [], []
    for idx, sample in samples.iterrows():
        print(f"  [{idx + 1}/{len(samples)}] {sample['filename']}")
        mzml_path = Path(str(sample["mzml_path"]))
        sample_intervals = intervals[
            intervals["filename"].eq(sample["filename"])]
        qc = {"filename": sample["filename"], "ion": sample["ion"],
              "subject": sample["subject"],
              "sample_id": sample["sample_id"], "date": sample["date"],
              "status": "ok", "n_platform_scans": 0, "n_peak_rows": 0}
        if not mzml_path.exists():
            qc["status"] = "missing_mzml"
            qc_rows.append(qc)
            continue
        interval_recs = sample_intervals[
            ["rt_start_min", "rt_end_min"]].to_dict("records")
        rows = []
        try:
            for scan_num, rt_min, tic, mz_arr, int_arr in iter_spectra(
                    mzml_path):
                if not any(it["rt_start_min"] <= rt_min <= it["rt_end_min"]
                           for it in interval_recs):
                    continue
                qc["n_platform_scans"] += 1
                if tic <= 0:
                    continue
                mask = ((mz_arr >= MZ_MIN) & (mz_arr <= MZ_MAX)
                        & (int_arr > MIN_PEAK_INTENSITY))
                if not np.any(mask):
                    continue
                mz_sel = mz_arr[mask]
                int_sel = int_arr[mask]
                mz_round = np.round(mz_sel, MZ_DECIMALS)
                for mz4, mzv, intensity in zip(mz_round, mz_sel, int_sel):
                    rows.append({
                        "filename": sample["filename"],
                        "ion": sample["ion"],
                        "subject": sample["subject"],
                        "sample_id": sample["sample_id"],
                        "date": sample["date"],
                        "scan_num": scan_num, "rt_min": rt_min,
                        "tic": tic, "mz_4dp": float(mz4),
                        "mz_value": float(mzv),
                        "intensity": float(intensity),
                        "relative_intensity": float(intensity / tic),
                    })
        except Exception as exc:
            qc["status"] = f"error: {exc}"
            qc_rows.append(qc)
            continue
        qc["n_peak_rows"] = len(rows)
        if qc["n_platform_scans"] == 0:
            qc["status"] = "no_platform_scans"
        elif not rows:
            qc["status"] = "no_peaks_after_filters"
        if rows:
            scan_parts.append(pd.DataFrame(rows))
        qc_rows.append(qc)

    scan_peak_df = (pd.concat(scan_parts, ignore_index=True)
                    if scan_parts else pd.DataFrame())
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(output_dir / "01_sample_qc.csv", index=False,
              encoding="utf-8-sig")
    if scan_peak_df.empty:
        print("  [WARN] No scan peaks extracted.")
        return

    clusters = _build_mz_clusters(scan_peak_df, CLUSTER_PPM)
    sample_feat = _build_sample_feature_matrix(
        scan_peak_df, clusters, samples, qc)
    sample_feat, presence = _filter_by_presence(sample_feat, samples)

    if sample_feat.empty:
        print("  [WARN] No features passed presence filter.")
        scan_peak_df.to_csv(output_dir / "01_scan_peak_relative_long.csv",
                            index=False, encoding="utf-8-sig")
        clusters.to_csv(output_dir / "02_mz_5ppm_clusters.csv", index=False,
                        encoding="utf-8-sig")
        return

    kept_fids = set(sample_feat["feature_id"].dropna().unique())
    clusters_kept = clusters[clusters["feature_id"].isin(kept_fids)].copy()

    scan_peak_df.to_csv(output_dir / "01_scan_peak_relative_long.csv",
                        index=False, encoding="utf-8-sig")
    clusters_kept.to_csv(output_dir / "02_mz_5ppm_clusters.csv", index=False,
                         encoding="utf-8-sig")
    sample_feat.to_csv(output_dir / "03_cluster_feature_sample_matrix.csv",
                       index=False, encoding="utf-8-sig")
    presence.to_csv(output_dir / "03_feature_presence_filter_summary.csv",
                    index=False, encoding="utf-8-sig")

    regression = _regression_per_subject(sample_feat)
    regression.to_csv(output_dir / "04_subject_feature_linear_regression.csv",
                      index=False, encoding="utf-8-sig")

    consistent_rows, consistent_summary = _summarize_consistent_trends(
        regression)
    consistent_rows.to_csv(
        output_dir / "05_consistent_trend_regression_rows.csv",
        index=False, encoding="utf-8-sig")
    consistent_summary.to_csv(output_dir / "05_consistent_trend_summary.csv",
                              index=False, encoding="utf-8-sig")

    if not consistent_summary.empty:
        vd = consistent_summary.copy()
        vd["neg_log10_max_pvalue_bh"] = -np.log10(
            vd["max_pvalue_bh"].clip(lower=1e-300))
        vd["abs_mean_slope"] = vd["mean_slope"].abs()
        vd["label"] = (vd["ion"].astype(str) + " "
                       + vd["cluster_mz"].round(4).astype(str))
        vd.to_csv(output_dir / "08_volcano_data.csv", index=False,
                  encoding="utf-8-sig")

    _volcano_plot(regression, output_dir)

    plotted = []
    for _, row in consistent_summary.head(TOP_N_PLOTS).iterrows():
        data = sample_feat[
            sample_feat["feature_id"].eq(row["feature_id"])].copy()
        fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        for subj, sty in BREATH_STYLES.items():
            sd = data[data["subject"].eq(subj)].sort_values("time_hours")
            if sd.empty:
                continue
            ax_r.plot(sd["time_hours"], sd["intensity"],
                      color=sty["color"], marker=sty["marker"],
                      linewidth=2.8, markersize=6, label=sty["label"])
            first = sd["intensity"].iloc[0]
            if first != 0:
                pct = (sd["intensity"] - first) / abs(first) * 100
                ax_p.plot(sd["time_hours"], pct,
                          color=sty["color"], marker=sty["marker"],
                          linewidth=2.5, markersize=6,
                          linestyle=(0, (6, 3)),
                          markerfacecolor="white", markeredgewidth=1.2,
                          label=sty["label"])
        title = (f"{row['ion']} m/z {row['cluster_mz']:.4f} "
                 f"({row['trend_direction']})")
        ax_r.set_title(title, fontsize=22, fontweight="bold")
        ax_r.set_ylabel("Mean relative intensity", fontsize=20,
                        fontweight="bold")
        ax_r.axhline(0, color="black", linewidth=1.0, alpha=0.4)
        ax_r.tick_params(axis="both", labelsize=16)
        ax_r.legend(loc="best", fontsize=14)
        ax_p.set_ylabel("Change (%)", fontsize=20, fontweight="bold")
        ax_p.set_xlabel("Detection time (hour)", fontsize=20,
                        fontweight="bold")
        ax_p.axhline(0, color="black", linewidth=1.0, alpha=0.4)
        ax_p.tick_params(axis="both", labelsize=16)
        ax_p.legend(loc="best", fontsize=14)
        fig.tight_layout()
        png = plots_dir / f"{row['feature_id']}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plotted.append({
            "feature_id": row["feature_id"],
            "ion": row["ion"],
            "cluster_mz": row["cluster_mz"],
            "trend_direction": row["trend_direction"],
            "plot": str(png.relative_to(output_dir)),
        })
    pd.DataFrame(plotted).to_csv(output_dir / "06_plot_index.csv",
                                 index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"metric": "samples", "value": len(samples)},
        {"metric": "scan_peak_rows", "value": len(scan_peak_df)},
        {"metric": "clusters", "value": len(clusters_kept)},
        {"metric": "sample_feature_rows", "value": len(sample_feat)},
        {"metric": "regression_rows", "value": len(regression)},
        {"metric": "consistent_features", "value": len(consistent_summary)},
        {"metric": "plots", "value": len(plotted)},
    ]).to_csv(output_dir / "00_run_summary.csv", index=False,
              encoding="utf-8-sig")

    print(f"  -> step 6 output: {output_dir}/")


# --- cli ---

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Breath Metabolomics Pipeline -- Fasting Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python breath_analysis.py --all\n"
               "  python breath_analysis.py --step2 --ion pos\n"
               "  python breath_analysis.py --step5 --known-ion pos\n"
               "  python breath_analysis.py --step6 --unknown-ion pos\n")
    parser.add_argument("--all", action="store_true",
                        help="Run steps 2-6 sequentially.")
    parser.add_argument("--step2", action="store_true")
    parser.add_argument("--step3", action="store_true")
    parser.add_argument("--step4", action="store_true")
    parser.add_argument("--step5", action="store_true")
    parser.add_argument("--step6", action="store_true")

    parser.add_argument("--base-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mzml-breath", type=Path, default=None)
    parser.add_argument("--mzml-blank", type=Path, default=None)
    parser.add_argument("--exhalion-dir", type=Path, default=None)
    parser.add_argument("--match-csv", type=Path, default=None)
    parser.add_argument("--crop-csv", type=Path, default=None)
    parser.add_argument("--blank-crop-csv", type=Path, default=None)
    parser.add_argument("--target-csv", type=Path, default=None)

    parser.add_argument("--ion", type=str, default="pos",
                        choices=["pos", "neg"])
    parser.add_argument("--known-ion", type=str, default="pos",
                        choices=["pos", "neg"])
    parser.add_argument("--unknown-ion", type=str, default="pos",
                        choices=["pos", "neg"])
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    base = args.base_dir
    out_root = args.output_dir or (base / "output")

    mzml_breath = args.mzml_breath or (base / "centroid_MS" / "Breath")
    mzml_blank = args.mzml_blank or (base / "centroid_MS" / "Blank")
    exhalion_dir = args.exhalion_dir or (base / "Exhalion")
    match_csv = args.match_csv or (base / "1.1_Exhalion_SESE_Filename.csv")
    crop_csv = args.crop_csv or (base / "1.2_TIC_crop_converted.csv")
    blank_crop = args.blank_crop_csv or (base / "1.2_blank_TimeChoose.csv")
    target_csv = args.target_csv or (base / "1.3_target_mz.csv")

    run_all = args.all
    any_step = run_all or any(
        [args.step2, args.step3, args.step4, args.step5, args.step6])
    if not any_step:
        parser.print_help()
        sys.exit(1)

    print(f"Breath Metabolomics Pipeline")
    print(f"  base   = {base}")
    print(f"  output = {out_root}")

    if run_all or args.step2:
        print("\n-- Step 2: Breath intervals --")
        run_step2_breath_intervals(
            mzml_dir=mzml_breath, exhalion_dir=exhalion_dir,
            match_csv=match_csv, crop_csv=crop_csv,
            output_dir=out_root / "step2_intervals", ion_mode=args.ion)

    if run_all or args.step3:
        print("\n-- Step 3: Blank intervals --")
        run_step3_blank_intervals(
            mzml_dir=mzml_blank, time_choose_csv=blank_crop,
            output_dir=out_root / "step3_blanks", ion_mode=args.ion)

    if run_all or args.step4:
        print("\n-- Step 4: Detection times --")
        breath_int_csv = (out_root / "step2_intervals"
                          / f"2_Breath_intervals_trimmed_{args.ion}.csv")
        blank_int_csv = (out_root / "step3_blanks"
                         / f"3_Blank_{args.ion}_retained_intervals.csv")
        run_step4_detection_times(
            breath_intervals_csv=breath_int_csv,
            blank_intervals_csv=blank_int_csv,
            breath_mzml_dir=mzml_breath, blank_mzml_dir=mzml_blank,
            output_dir=out_root / "step4_detection")

    if run_all or args.step5:
        print("\n-- Step 5: Known-target trends --")
        run_step5_known_targets(
            processed_dir=out_root, target_csv=target_csv,
            output_dir=out_root / "step5_known", ion_mode=args.known_ion)

    if run_all or args.step6:
        print("\n-- Step 6: Untargeted screening --")
        run_step6_unknown_screening(
            processed_dir=out_root,
            output_dir=out_root / "step6_unknown", ion_mode=args.unknown_ion)

    print(f"\nDone.  Output: {out_root}")


if __name__ == "__main__":
    main()
