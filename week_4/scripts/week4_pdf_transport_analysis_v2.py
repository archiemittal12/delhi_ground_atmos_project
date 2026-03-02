# week_4/scripts/week4_pdf_transport_analysis_v2.py
"""
WEEK 4 — PDF TRANSPORT & DIRECTIONAL ENRICHMENT (v2)
- 12 PDF plots: for each year (2021-2024) and month (Dec, Jan, Feb),
  overlaying 4 regions (North, East, South_Central, West_SW).
- One directional enrichment plot per region (aggregated over DJF and all years).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = PROJECT_ROOT / "final_station_data_clean"
OUTPUT_DIR = PROJECT_ROOT / "week_4" / "outputs" / "pdf_transport"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- REGIONS ----------------
regions = {
    "North": [
        "station_1430_rohini_delhi_dpcc_1hr",
        "station_1423_jahangirpuri_delhi_dpcc_1hr",
        "station_1420_ashok_vihar_delhi_dpcc_1hr",
        "station_1434_wazirpur_delhi_dpcc_1hr",
        "station_1426_narela_delhi_dpcc_1hr",
        "station_1560_bawana_delhi_dpcc_1hr",
    ],
    "East": [
        "station_114_ihbas_dilshad_garden_delhi_cpcb_1hr",
        "station_301_anand_vihar_delhi_dpcc_1hr",
        "station_1431_patparganj_delhi_dpcc_1hr",
        "station_1435_vivek_vihar_delhi_dpcc_1hr",
        "station_1432_sonia_vihar_delhi_dpcc_1hr",
        "station_1563_pusa_delhi_dpcc_1hr",
    ],
    "South_Central": [
        "station_122_mandir_marg_delhi_dpcc_1hr",
        "station_124_r_k_puram_delhi_dpcc_1hr",
        "station_1428_okhla_phase_2_delhi_dpcc_1hr",
        "station_1429_nehru_nagar_delhi_dpcc_1hr",
        "station_1562_sri_aurobindo_marg_delhi_dpcc_1hr",
    ],
    "West_SW": [
        "station_113_shadipur_delhi_cpcb_1hr",
        "station_115_nsit_dwarka_delhi_cpcb_1hr",
        "station_125_punjabi_bagh_delhi_dpcc_1hr",
        "station_1422_dwarka_sector_8_delhi_dpcc__1hr",
        "station_1427_najafgarh_delhi_dpcc_1hr",
        "station_1561_mundka_delhi_dpcc_1hr",
    ],
}

YEARS = [2021, 2022, 2023, 2024]
DJF_MONTHS = [12, 1, 2]
MONTH_NAMES = {12: "Dec", 1: "Jan", 2: "Feb"}

# ---------------- helpers to detect columns ----------------
def detect_column(cols, keywords):
    """
    Return first column name from cols that contains any keyword (case-insensitive).
    """
    cols_map = {c: c.lower() for c in cols}
    for k in keywords:
        k = k.lower()
        for orig, lower in cols_map.items():
            if k in lower:
                return orig
    return None

def standardize_columns(df):
    """
    Make sure df has 'datetime', 'pm25', 'wind_dir', 'wind_speed' columns when possible.
    Returns df (may raise ValueError if no time/pm found).
    """
    cols = df.columns.tolist()

    # timestamp / datetime
    if "timestamp" in cols:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in cols:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # try to detect 'date' + 'time' or other variants
        date_col = detect_column(cols, ["date", "day"])
        time_col = detect_column(cols, ["time", "hour"])
        if date_col and time_col:
            df["datetime"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        else:
            # try any timestamp-like column
            ts_candidate = detect_column(cols, ["ts", "timestamp", "time", "date", "datetime"])
            if ts_candidate:
                df["datetime"] = pd.to_datetime(df[ts_candidate], errors="coerce")
            else:
                raise ValueError("No timestamp/datetime column found in data file. Columns: " + ", ".join(cols))

    # PM column
    pm_col = detect_column(cols, ["pm2", "pm25", "pm 2.5", "pm2.5"])
    if pm_col is None:
        raise ValueError("No PM2.5 column found. Columns: " + ", ".join(cols))
    df["pm25"] = pd.to_numeric(df[pm_col], errors="coerce")

    # wind direction
    wd_col = detect_column(cols, ["wind_dir", "wd", "wind direction", "wind_dir_deg", "winddirection"])
    if wd_col:
        df["wind_dir"] = pd.to_numeric(df[wd_col], errors="coerce")

    # wind speed
    ws_col = detect_column(cols, ["wind_speed", "ws", "windspd", "speed"])
    if ws_col:
        df["wind_speed"] = pd.to_numeric(df[ws_col], errors="coerce")

    return df

# ---------------- load region-month data ----------------
def load_region_month(region, year, month):
    """
    Collects csv files for stations in a given region, for a given year and month.
    Returns concatenated DataFrame or None if no data.
    """
    dfs = []
    for station in regions[region]:
        station_path = BASE_PATH / station
        if not station_path.exists():
            # station folder missing — just skip
            continue
        # read csvs that contain the year in their filename
        for f in station_path.glob("*.csv"):
            if str(year) in f.name:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"Warning: failed to read {f}: {e}")
                    continue
                try:
                    df = standardize_columns(df)
                except Exception as e:
                    # If a file lacks time/pm etc, skip it
                    print(f"Skipping {f.name} — {e}")
                    continue
                # select month
                df = df[df["datetime"].dt.month == month]
                if not df.empty:
                    dfs.append(df)
    if len(dfs) == 0:
        return None
    return pd.concat(dfs, ignore_index=True, sort=False)

# ---------------- compute pdf routine ----------------
def plot_pdf_for_year_month(year, month, out_dir):
    """
    For a single year & month, plot PDFs for each region (4 curves) on the same axis.
    Saves figure to out_dir/{year}_{monthname}_pdf.png
    """
    plt.figure(figsize=(10, 6))
    plotted = 0
    for region in regions:
        df = load_region_month(region, year, month)
        if df is None:
            continue
        vals = df["pm25"].dropna().values
        vals = vals[vals > 0]  # remove non-physical zeros if any
        if len(vals) < 30:
            # skip too few points (avoid bad KDEs)
            print(f"Skipping {region} {year}-{month}: only {len(vals)} PM samples")
            continue
        try:
            kde = gaussian_kde(vals)
            x_max = np.percentile(vals, 99)
            x = np.linspace(0, max(30, x_max), 400)
            y = kde(x)
            plt.plot(x, y, label=region, linewidth=1.5)
            plotted += 1
        except Exception:
            # fallback to histogram density
            hist, edges = np.histogram(vals, bins=50, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            plt.plot(centers, hist, label=region, linewidth=1.5)
            plotted += 1

    if plotted == 0:
        print(f"No curves plotted for {year}-{month}; skipping figure.")
        plt.close()
        return

    plt.title(f"PM2.5 PDFs — {year} {MONTH_NAMES[month]} (4 regions)")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Probability density")
    plt.xlim(left=0)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_file = out_dir / f"{year}_{MONTH_NAMES[month]}_pdf.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved PDF:", out_file)

# ---------------- directional enrichment per region ----------------
def sector_from_deg(wd):
    # Map degrees to 8 sector labels
    wd = wd % 360
    labels = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int(((wd + 22.5) % 360) // 45)
    return labels[idx]

def compute_and_plot_enrichment_per_region(out_dir):
    """
    For each region, aggregate DJF data across all YEARS,
    compute enrichment (freq in extreme / freq in background) over 8 sectors,
    and save one plot per region.
    Also save a CSV with enrichment numbers.
    """
    rows = []
    for region in regions:
        # collect all DJF hours (all years) for this region
        dfs = []
        for y in YEARS:
            for m in DJF_MONTHS:
                df = load_region_month(region, y, m)
                if df is not None:
                    dfs.append(df)
        if len(dfs) == 0:
            print(f"No DJF data for region {region} — skipping enrichment")
            continue
        df_all = pd.concat(dfs, ignore_index=True, sort=False)
        df_all = df_all.dropna(subset=["pm25", "wind_dir"])
        if df_all.empty:
            print(f"No wind_dir/pm data for region {region} — skipping")
            continue
        # compute threshold (top 15%) using the region's DJF distribution
        threshold = df_all["pm25"].quantile(0.85)
        background_dirs = df_all["wind_dir"].values
        extreme_dirs = df_all[df_all["pm25"] >= threshold]["wind_dir"].values

        # bin into 8 sectors
        sectors = ["N","NE","E","SE","S","SW","W","NW"]
        bg_counts = {s:0 for s in sectors}
        ex_counts = {s:0 for s in sectors}

        for wd in background_dirs:
            s = sector_from_deg(wd)
            bg_counts[s] += 1
        for wd in extreme_dirs:
            s = sector_from_deg(wd)
            ex_counts[s] += 1

        # convert to frequencies
        bg_total = sum(bg_counts.values()) or 1
        ex_total = sum(ex_counts.values()) or 1
        bg_freq = {s: bg_counts[s]/bg_total for s in sectors}
        ex_freq = {s: ex_counts[s]/ex_total for s in sectors}

        # enrichment ratio (extreme / background) with safe divide
        enrichment = {s: (ex_freq[s] / (bg_freq[s] + 1e-9)) for s in sectors}

        # Save numeric rows for CSV
        for s in sectors:
            rows.append({
                "Region": region,
                "Sector": s,
                "BG_count": bg_counts[s],
                "EX_count": ex_counts[s],
                "BG_freq": bg_freq[s],
                "EX_freq": ex_freq[s],
                "Enrichment": enrichment[s],
                "threshold_85pct_pm25": float(threshold)
            })

        # Plot enrichment as bar + dashed line at 1
        fig, ax = plt.subplots(figsize=(7,4.5))
        x = np.arange(len(sectors))
        y = [enrichment[s] for s in sectors]
        ax.bar(x, y, tick_label=sectors)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Enrichment (extreme / background)")
        ax.set_title(f"Directional Enrichment — {region} (DJF, all years)")
        ax.set_ylim(0, max(1.2, np.nanpercentile(y, 95)*1.3))
        for i,v in enumerate(y):
            ax.text(i, v + 0.03*np.nanmax(y), f"{v:.2f}", ha='center', fontsize=8)
        plt.tight_layout()
        out_file = out_dir / f"enrichment_{region}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print("Saved enrichment plot for:", region)

    # write CSV summary
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "directional_enrichment_by_region.csv", index=False)
        print("Saved directional_enrichment_by_region.csv")

# ---------------- main execution ----------------
def main():
    # 1) Create 12 PDF plots (4 years × 3 months)
    for year in YEARS:
        for month in DJF_MONTHS:
            plot_pdf_for_year_month(year, month, OUTPUT_DIR)

    # 2) Enrichment per region (DJF aggregated)
    compute_and_plot_enrichment_per_region(OUTPUT_DIR)

    print("\nAll outputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()