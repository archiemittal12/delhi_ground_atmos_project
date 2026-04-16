"""
ASD390 Week 6 — Task 2
Monthly PM2.5 maps for Oct/Nov/Dec/Jan/Feb, pooled across 2021-2024.
5 separate PNG files, one per month.

Place at:  delhi_ground_atmos_project/week_6/task2_monthly_maps.py
Run from project root:  python week_6/task2_monthly_maps.py
"""
from pathlib import Path
import pandas as pd

# Reuse everything from task 1
from task1_annual_maps import (
    load_all_stations,
    plot_delhi_stations,
    compute_shared_scale,
    PROJECT_ROOT,
)

# ============================================================
# CONFIG
# ============================================================
OUT_DIR = PROJECT_ROOT / "week_6" / "task2_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [10, 11, 12, 1, 2]
MONTH_NAMES = {
    10: "October",
    11: "November",
    12: "December",
    1:  "January",
    2:  "February",
}


# ============================================================
# AGGREGATION
# ============================================================
def compute_month_pooled_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (station, calendar month), compute the mean PM2.5
    pooled across 2021-2024.

    Two-step aggregation (honest to missing years):
      1. Monthly mean per station per year
      2. Average those per-year monthly means across the 4 years

    This gives equal weight to each year even if one has more valid
    hours than another. Returns:
        [site_id, folder, lat, lon, month, pm25_pooled_mean]
    """
    df = df.copy()
    df["year"]  = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # Step 1: mean per (station, year, month)
    step1 = (
        df.groupby(
            ["site_id", "folder", "lat", "lon", "year", "month"],
            as_index=False,
        )["pm25"].mean()
    )

    # Step 2: average across years for each (station, month)
    step2 = (
        step1.groupby(
            ["site_id", "folder", "lat", "lon", "month"],
            as_index=False,
        )["pm25"].mean()
        .rename(columns={"pm25": "pm25_pooled_mean"})
    )

    return step2[step2.month.isin(MONTHS)].reset_index(drop=True)


# ============================================================
# DRIVER
# ============================================================
def main():
    print("=" * 60)
    print("ASD390 Week 6 — Task 2: Monthly pooled PM2.5 maps")
    print("=" * 60)

    print("\n[1/4] Loading station data...")
    df = load_all_stations()
    print(f"  total rows:    {len(df):,}")
    print(f"  unique sites:  {df.site_id.nunique()}")

    print("\n[2/4] Computing month-pooled means (2021-2024)...")
    pooled = compute_month_pooled_means(df)
    pivot = pooled.pivot(
        index="site_id", columns="month", values="pm25_pooled_mean"
    ).round(1)
    # Reorder columns to Oct-Feb
    pivot = pivot[MONTHS]
    pivot.columns = [MONTH_NAMES[m] for m in MONTHS]
    print(pivot)

    print("\n[3/4] Computing shared color scale (P5–P95) across all 5 months...")
    vmin, vmax = compute_shared_scale(pooled.pm25_pooled_mean.values)
    print(f"  vmin={vmin:.1f}  vmax={vmax:.1f}  ug/m3")
    print(f"  (note: scale is separate from Task 1 because monthly means")
    print(f"   have different dynamic range than annual means)")

    print("\n[4/4] Plotting...")
    for m in MONTHS:
        sub = pooled[pooled.month == m].copy()
        name = MONTH_NAMES[m]
        out_path = OUT_DIR / f"task2_{m:02d}_{name.lower()}.png"
        plot_delhi_stations(
            sub,
            value_col="pm25_pooled_mean",
            vmin=vmin, vmax=vmax,
            title=f"{name} mean PM$_{{2.5}}$ — Delhi (2021–2024 pooled)",
            out_path=out_path,
        )
        print(f"  saved {out_path.name}  ({len(sub)} stations)")

    print("\nDone.")


if __name__ == "__main__":
    main()
