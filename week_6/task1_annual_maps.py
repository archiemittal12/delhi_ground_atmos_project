"""
ASD390 Week 6 — Task 1
Annual mean PM2.5 maps across Delhi stations, one plot per year (2021-2024).

Place at:  delhi_ground_atmos_project/week_6/task1_annual_maps.py
Run from project root:  python week_6/task1_annual_maps.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "final_station_data_clean"
OUT_DIR      = PROJECT_ROOT / "week_6" / "task1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024]

# Delhi NCR bounding box (with small pad)
DELHI_EXTENT = [76.80, 77.40, 28.40, 28.90]  # [lon_min, lon_max, lat_min, lat_max]


# ============================================================
# STATION METADATA
# site_id -> (lat, lon)
# Pulled from your week_5 STATION_COORDS dict, keyed by CPCB site_id
# so we can match the station folder names directly.
# ============================================================
SITE_COORDS = {
    "113":  (28.6530, 77.1470),  # Shadipur
    "114":  (28.6812, 77.3025),  # IHBAS Dilshad Garden
    "115":  (28.6080, 77.0310),  # NSIT Dwarka
    "122":  (28.6360, 77.1980),  # Mandir Marg
    "124":  (28.5640, 77.1822),  # R K Puram
    "125":  (28.6760, 77.1340),  # Punjabi Bagh
    "301":  (28.6470, 77.3157),  # Anand Vihar
    "1420": (28.6900, 77.1813),  # Ashok Vihar
    "1422": (28.5810, 77.0590),  # Dwarka Sector 8
    "1423": (28.7302, 77.1693),  # Jahangirpuri
    "1426": (28.8555, 77.0940),  # Narela
    "1427": (28.6100, 76.9780),  # Najafgarh
    "1428": (28.5300, 77.2700),  # Okhla Phase 2
    "1429": (28.5660, 77.2490),  # Nehru Nagar
    "1430": (28.7157, 77.1100),  # Rohini
    "1431": (28.6300, 77.2800),  # Patparganj
    "1432": (28.7200, 77.2700),  # Sonia Vihar
    "1434": (28.7028, 77.1632),  # Wazirpur
    "1435": (28.6690, 77.3150),  # Vivek Vihar
    "1560": (28.7900, 77.0380),  # Bawana
    "1561": (28.6860, 77.0530),  # Mundka

    # TODO: confirm these 5 — they were NOT in your week_5 STATION_COORDS
    # dict but appear in the folder list / REGIONS mapping. Look them up
    # from CPCB CAAQMS station list or Google their exact coordinates.
    "1421": (28.5500, 77.1900),  # Dr Karni Singh Shooting Range  (APPROX — verify)
    "1424": (28.5830, 77.2340),  # Jawaharlal Nehru Stadium       (APPROX — verify)
    "1425": (28.6120, 77.2370),  # Major Dhyan Chand Natl Stadium (APPROX — verify)
    "1562": (28.6356, 77.1492),  # Pusa                           (APPROX — verify)
    "1563": (28.5350, 77.1870),  # Sri Aurobindo Marg             (APPROX — verify)
}


def extract_site_id(folder_name: str) -> str:
    """station_113_shadipur_delhi_cpcb_1hr -> '113'"""
    return folder_name.split("_")[1]


# ============================================================
# DATA LOADING
# ============================================================
def load_station_year(folder_name: str, year: int) -> pd.DataFrame:
    """
    Load one station's yearly CSV.
    Columns in raw file (per week_5 notebook): 'timestamp', 'PM 2.5', ...
    Returns: [timestamp, pm25]
    """
    path = DATA_DIR / folder_name / f"{year}.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = df[["timestamp", "PM 2.5"]].rename(columns={"PM 2.5": "pm25"})
    return out


def load_all_stations() -> pd.DataFrame:
    """
    Returns long-format: [site_id, folder, lat, lon, timestamp, pm25]
    """
    rows = []
    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = DATA_DIR / folder
        if not folder_path.is_dir():
            continue

        site_id = extract_site_id(folder)
        if site_id not in SITE_COORDS:
            print(f"  WARNING: no coords for site_id {site_id} ({folder}) — skipping")
            continue
        lat, lon = SITE_COORDS[site_id]

        for year in YEARS:
            try:
                df = load_station_year(folder, year)
            except FileNotFoundError:
                print(f"  missing file: {folder}/{year}.csv")
                continue
            df["site_id"] = site_id
            df["folder"]  = folder
            df["lat"]     = lat
            df["lon"]     = lon
            rows.append(df)

    if not rows:
        raise RuntimeError("No station data loaded — check DATA_DIR path")

    return pd.concat(rows, ignore_index=True)


# ============================================================
# AGGREGATION
# ============================================================
def compute_annual_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: [site_id, folder, lat, lon, year, pm25_annual_mean]
    """
    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    out = (
        df.groupby(["site_id", "folder", "lat", "lon", "year"], as_index=False)
          ["pm25"].mean()
          .rename(columns={"pm25": "pm25_annual_mean"})
    )
    return out[out.year.isin(YEARS)].reset_index(drop=True)


def compute_shared_scale(values, p_low=5, p_high=95):
    arr = np.asarray(values, dtype=float)
    return float(np.nanpercentile(arr, p_low)), float(np.nanpercentile(arr, p_high))


# ============================================================
# PLOTTING
# ============================================================
def plot_delhi_stations(
    station_df: pd.DataFrame,
    value_col: str,
    vmin: float,
    vmax: float,
    title: str,
    out_path: Path,
    cmap: str = "YlOrRd",
    extent=DELHI_EXTENT,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 8.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", alpha=0.3)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False

    valid   = station_df[station_df[value_col].notna()]
    missing = station_df[station_df[value_col].isna()]

    sc = ax.scatter(
        valid.lon, valid.lat,
        c=valid[value_col],
        s=180, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolor="black", linewidth=0.7,
        transform=ccrs.PlateCarree(), zorder=5,
    )

    if len(missing) > 0:
        ax.scatter(
            missing.lon, missing.lat,
            s=180, facecolors="none", edgecolors="grey", linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=4,
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label(r"PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=12)

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DRIVER
# ============================================================
def main():
    print("=" * 60)
    print("ASD390 Week 6 — Task 1: Annual PM2.5 station maps")
    print("=" * 60)

    print("\n[1/4] Loading station data...")
    df = load_all_stations()
    print(f"  total rows:    {len(df):,}")
    print(f"  unique sites:  {df.site_id.nunique()}")
    print(f"  date range:    {df.timestamp.min()} to {df.timestamp.max()}")
    print(f"  pm25 NaN %:    {df.pm25.isna().mean() * 100:.2f}%")

    print("\n[2/4] Computing annual means per station...")
    annual = compute_annual_means(df)
    pivot  = annual.pivot(index="site_id", columns="year",
                          values="pm25_annual_mean").round(1)
    print(pivot)

    print("\n[3/4] Computing shared color scale (P5–P95)...")
    vmin, vmax = compute_shared_scale(annual.pm25_annual_mean.values)
    print(f"  vmin={vmin:.1f}  vmax={vmax:.1f}  ug/m3")

    print("\n[4/4] Plotting...")
    for year in YEARS:
        sub = annual[annual.year == year].copy()
        out_path = OUT_DIR / f"task1_annual_pm25_{year}.png"
        plot_delhi_stations(
            sub,
            value_col="pm25_annual_mean",
            vmin=vmin, vmax=vmax,
            title=f"Annual mean PM$_{{2.5}}$ — Delhi {year}",
            out_path=out_path,
        )
        print(f"  saved {out_path.name}  ({len(sub)} stations)")

    print("\nDone.")


if __name__ == "__main__":
    main()
