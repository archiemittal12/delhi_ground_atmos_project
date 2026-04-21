"""
ASD390 Week 6 — Ventilation Index Stagnant Anomaly
Spatial plots of VI anomaly on India-ASI stagnant days vs climatology.

VI = ws10m × BLH (m²/s)
Anomaly = mean VI on stagnant days − climatological mean VI for that month
→ negative values = reduced ventilation during stagnation (expected)

Output: 20 PNGs (4 years × 5 months Oct–Feb) in week_6/vi_stagnant_outputs/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.ops import unary_union
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "week_6" / "data"
SHAPEFILE    = (DATA_DIR / "shapefiles" /
                "ne_10m_admin_1_states_provinces" /
                "ne_10m_admin_1_states_provinces.shp")
OUT_DIR      = PROJECT_ROOT / "week_6" / "vi_stagnant_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS  = [2021, 2022, 2023, 2024]
MONTHS = [10, 11, 12, 1, 2]
MONTH_NAMES = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb"}

WS_THRESHOLD = 3.2
TP_THRESHOLD = 1.0

COMMON_START = "2021-01-01"
COMMON_END   = "2024-12-31"

WIND_VAR = "ws10m"
T2M_VAR  = "2t"
T925_VAR = "t"
TP_VAR   = "tp"
BLH_VAR  = "blh"

COORD_RENAME = {"lat": "latitude", "lon": "longitude"}

STATION_COORDS = {
    "Rohini":        (28.7157, 77.1100),
    "Jahangirpuri":  (28.7302, 77.1693),
    "Ashok Vihar":   (28.6900, 77.1813),
    "Wazirpur":      (28.7028, 77.1632),
    "Narela":        (28.8555, 77.0940),
    "Bawana":        (28.7900, 77.0380),
    "IHBAS":         (28.6812, 77.3025),
    "Anand Vihar":   (28.6470, 77.3157),
    "Patparganj":    (28.6300, 77.2800),
    "Vivek Vihar":   (28.6690, 77.3150),
    "Sonia Vihar":   (28.7200, 77.2700),
    "Pusa":          (28.6356, 77.1492),
    "Mandir Marg":   (28.6360, 77.1980),
    "RK Puram":      (28.5640, 77.1822),
    "Okhla Ph2":     (28.5300, 77.2700),
    "Nehru Nagar":   (28.5660, 77.2490),
    "Sri Aurobindo": (28.5350, 77.1870),
    "Shadipur":      (28.6530, 77.1470),
    "NSIT Dwarka":   (28.6080, 77.0310),
    "Punjabi Bagh":  (28.6760, 77.1340),
    "Dwarka Sec8":   (28.5810, 77.0590),
    "Najafgarh":     (28.6100, 76.9780),
    "Mundka":        (28.6860, 77.0530),
}


# ============================================================
# HELPERS
# ============================================================
def normalize_coords(ds):
    rename = {k: v for k, v in COORD_RENAME.items()
              if k in ds.coords or k in ds.dims}
    if rename:
        ds = ds.rename(rename)
    return ds


def normalize_time(ds):
    """Strip sub-daily time components → all timestamps become midnight.
    This fixes the 00:00 vs 11:30 mismatch between wind and met files."""
    ds = ds.copy()
    ds["time"] = ds.time.dt.floor("D")
    return ds


def clip_to_delhi(ds, LAT, LON):
    return ds.sel(
        latitude=slice(LAT[1], LAT[0]),
        longitude=slice(LON[0], LON[1]),
        time=slice(COMMON_START, COMMON_END),
    )


def load_delhi_geometry():
    india = gpd.read_file(str(SHAPEFILE))
    delhi = india[india["name"].str.contains("Delhi", case=False, na=False)].copy()
    delhi = delhi.to_crs(epsg=4326)
    delhi_geom = unary_union(delhi.geometry)
    minx, miny, maxx, maxy = delhi.total_bounds
    pad = 0.05
    LAT = (miny - pad, maxy + pad)
    LON = (minx - pad, maxx + pad)
    return delhi_geom, LAT, LON


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(LAT, LON):
    """Load all ERA5 datasets, normalize times, clip to Delhi, compute daily VI."""

    # --- Wind (daily, timestamps at 00:00) ---
    print("  loading wind speed...")
    ds_wind = xr.open_mfdataset(
        [str(DATA_DIR / "ERA5_WS_daily_2021.nc"),
         str(DATA_DIR / "ERA5_WS_daily_2022_2024.nc")],
        combine="by_coords",
    )
    ds_wind = normalize_coords(ds_wind)
    ds_wind = normalize_time(ds_wind)
    wind_d = clip_to_delhi(ds_wind, LAT, LON)

    # --- T2m (daily, timestamps at 11:30) ---
    print("  loading t2m...")
    ds_t2m = xr.open_dataset(DATA_DIR / "era5_daily_t2m_2021_2024.nc")
    ds_t2m = normalize_coords(ds_t2m)
    ds_t2m = normalize_time(ds_t2m)
    t2m_d = clip_to_delhi(ds_t2m, LAT, LON)

    # --- T925 (daily, timestamps at 11:30) ---
    print("  loading t925...")
    ds_t925 = xr.open_dataset(DATA_DIR / "era5_daily_t925hpa_2021_2024.nc")
    ds_t925 = normalize_coords(ds_t925)
    ds_t925 = normalize_time(ds_t925)
    if "plev" in ds_t925.dims:
        ds_t925 = ds_t925.squeeze("plev", drop=True)
    t925_d = clip_to_delhi(ds_t925, LAT, LON)

    # --- Precip (daily, timestamps at 11:30) ---
    print("  loading precip...")
    ds_tp = xr.open_dataset(DATA_DIR / "era5_daily_tp_2021_2024.nc")
    ds_tp = normalize_coords(ds_tp)
    ds_tp = normalize_time(ds_tp)
    tp_d = clip_to_delhi(ds_tp, LAT, LON)

    # --- BLH (HOURLY → daily mean) ---
    print("  loading BLH (hourly, 3GB — this may take a minute)...")
    ds_blh = xr.open_dataset(
        DATA_DIR / "Copy of blh_era5_hr_2021_2024.nc",
        chunks={"time": 720},
    )
    ds_blh = normalize_coords(ds_blh)
    blh_delhi = ds_blh.sel(
        latitude=slice(LAT[1], LAT[0]),
        longitude=slice(LON[0], LON[1]),
        time=slice(COMMON_START, COMMON_END),
    )
    print("  resampling BLH to daily mean...")
    blh_daily = blh_delhi[BLH_VAR].resample(time="1D").mean().compute()
    # After resample("1D"), timestamps are already at midnight — no floor needed
    print(f"  BLH daily shape: {blh_daily.shape}")

    # --- Compute daily VI = ws10m × BLH ---
    print("  computing daily VI = ws10m × BLH...")

    # All timestamps are now at midnight — find common dates
    common_times = np.intersect1d(wind_d.time.values, blh_daily.time.values)
    print(f"  common time steps between wind and BLH: {len(common_times)}")

    if len(common_times) > 0:
        ws = wind_d[WIND_VAR].sel(time=common_times)
        blh = blh_daily.sel(time=common_times)
        vi_daily = (ws * blh).rename("vi")
    else:
        raise RuntimeError(
            "No common timestamps between wind and BLH even after floor('D'). "
            "Check the raw time values of both files."
        )

    print(f"  VI daily: {vi_daily.shape}, "
          f"range [{float(vi_daily.min()):.0f}, {float(vi_daily.max()):.0f}] m²/s")

    # Also ensure met files share the same time axis
    # (they should, since all are floored to midnight with 1461 days)
    met_times = common_times  # use the same time axis for everything

    wind_d = wind_d.sel(time=met_times)
    t2m_d  = t2m_d.sel(time=met_times)
    t925_d = t925_d.sel(time=met_times)
    tp_d   = tp_d.sel(time=met_times)

    return wind_d, t2m_d, t925_d, tp_d, vi_daily


# ============================================================
# STAGNATION MASK + VI ANOMALY
# ============================================================
def build_stagnant_mask(wind_d, t2m_d, t925_d, tp_d):
    """Combined India-ASI mask: all 3 conditions TRUE simultaneously."""
    cond_wind = wind_d[WIND_VAR] < WS_THRESHOLD
    cond_inv  = (t925_d[T925_VAR] - t2m_d[T2M_VAR]) > 0
    cond_dry  = tp_d[TP_VAR] < TP_THRESHOLD
    mask = cond_wind & cond_inv & cond_dry
    return mask


def compute_vi_stagnant_anomaly(vi_daily, stag_mask):
    """
    For each year-month:
      VI_stagnant_mean = mean VI on stagnant days in that month
      VI_clim           = climatological mean VI for that calendar month (all days)
      Anomaly           = VI_stagnant_mean − VI_clim
    """
    # Filter to Oct-Feb
    oct_feb_sel = vi_daily["time"].dt.month.isin(MONTHS)
    vi_of = vi_daily.sel(time=oct_feb_sel)
    stag_of = stag_mask.sel(time=oct_feb_sel)

    print(f"\n  Oct-Feb subset: {len(vi_of.time)} days")

    # Climatological mean VI per calendar month (ALL days, not just stagnant)
    vi_clim = vi_of.groupby("time.month").mean(dim="time")
    print("  VI climatology (domain-avg, m²/s):")
    for m in MONTHS:
        print(f"    month {m:02d}: {float(vi_clim.sel(month=m).mean()):,.0f}")

    # Per year-month: mean VI on stagnant days only
    results = {}
    for year in YEARS:
        for month in MONTHS:
            sel = (
                (vi_of["time"].dt.year == year) &
                (vi_of["time"].dt.month == month)
            )
            vi_ym = vi_of.sel(time=sel)
            stag_ym = stag_of.sel(time=sel)

            if len(vi_ym.time) == 0:
                print(f"    {year}-{month:02d}: no data, skipping")
                continue

            # Mask non-stagnant days with NaN
            stag_bool = stag_ym.values.astype(bool)
            vi_vals = vi_ym.values.copy().astype(float)
            vi_vals[~stag_bool] = np.nan

            n_stag = int(np.any(stag_bool, axis=(1, 2)).sum())

            with np.errstate(all="ignore"):
                vi_stag_mean = np.nanmean(vi_vals, axis=0)

            if np.all(np.isnan(vi_stag_mean)):
                print(f"    {year}-{month:02d}: 0 stagnant days, skipping")
                continue

            # Anomaly
            clim_vals = vi_clim.sel(month=month).values
            anomaly = vi_stag_mean - clim_vals

            results[(year, month)] = xr.DataArray(
                anomaly,
                dims=["latitude", "longitude"],
                coords={
                    "latitude": vi_of.latitude,
                    "longitude": vi_of.longitude,
                },
            )

            anom_mean = float(np.nanmean(anomaly))
            print(f"    {year}-{month:02d}: {n_stag:2d} stagnant days, "
                  f"domain-avg VI anomaly = {anom_mean:+,.0f} m²/s")

    return results


# ============================================================
# PLOTTING
# ============================================================
def geometry_mask(geom, x, y):
    pts = np.column_stack([x.ravel(), y.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        ext = MplPath(np.asarray(poly.exterior.coords))
        inside = ext.contains_points(pts)
        for hole in poly.interiors:
            inside &= ~MplPath(np.asarray(hole.coords)).contains_points(pts)
        mask |= inside
    return mask.reshape(x.shape)


def plot_vi_anomaly(anomaly_da, year, month, delhi_geom, LAT, LON, out_path):
    ngrid = 300
    grid_lon = np.linspace(LON[0], LON[1], ngrid)
    grid_lat = np.linspace(LAT[0], LAT[1], ngrid)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    lon2d, lat2d = np.meshgrid(
        anomaly_da.longitude.values,
        anomaly_da.latitude.values,
    )
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    values = anomaly_da.values.ravel()

    grid_z = griddata(points, values, (grid_lon2d, grid_lat2d), method="linear")
    grid_z_nn = griddata(points, values, (grid_lon2d, grid_lat2d), method="nearest")
    grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)
    grid_z = gaussian_filter(grid_z, sigma=3)

    inside = geometry_mask(delhi_geom, grid_lon2d, grid_lat2d)
    grid_z_masked = np.where(inside, grid_z, np.nan)

    vmax = max(np.nanpercentile(np.abs(grid_z_masked), 98), 100)
    levels = np.linspace(-vmax, vmax, 25)

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    contour = ax.contourf(
        grid_lon2d, grid_lat2d, grid_z_masked,
        levels=levels, cmap="RdBu", extend="both",
        transform=ccrs.PlateCarree(),
    )

    ax.add_geometries([delhi_geom], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=2.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="gray")
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", alpha=0.25)

    ax.scatter(lon2d.ravel(), lat2d.ravel(),
               s=60, color="k", marker="+", linewidths=1.8,
               transform=ccrs.PlateCarree(), zorder=5)

    for name, (slat, slon) in STATION_COORDS.items():
        ax.plot(slon, slat, "^", ms=7, color="gold",
                markeredgecolor="k", markeredgewidth=0.6,
                transform=ccrs.PlateCarree(), zorder=6)

    ax.set_extent([LON[0], LON[1], LAT[0], LAT[1]])
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(contour, ax=ax, shrink=0.78, pad=0.03)
    cbar.set_label("VI Anomaly (m² s⁻¹)", fontsize=12)

    n_cells = len(anomaly_da.latitude) * len(anomaly_da.longitude)
    ax.text(0.01, 0.01, f"ERA5 0.25° grid: {n_cells} cells in domain",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    legend_elements = [
        Line2D([0], [0], marker="+", color="k", label="ERA5 grid pt",
               linestyle="None", ms=9),
        Line2D([0], [0], marker="^", color="gold", label="CPCB station",
               linestyle="None", ms=8, markeredgecolor="k", markeredgewidth=0.6),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    mname = MONTH_NAMES[month]
    plt.title(
        f"VI Stagnant Anomaly — Delhi  ({year}-{month:02d})\n"
        f"VI = ws10m × BLH  |  −ve = reduced ventilation on stagnant days",
        fontsize=12, weight="bold",
    )

    plt.subplots_adjust(right=0.85)
    plt.savefig(out_path, dpi=96, bbox_inches="tight")
    plt.close()


# ============================================================
# DRIVER
# ============================================================
def main():
    print("=" * 65)
    print("ASD390 Week 6 — VI Stagnant Anomaly Spatial Maps")
    print("  VI = ws10m × BLH")
    print("  Anomaly = mean VI on India-ASI stagnant days − VI climatology")
    print("=" * 65)

    print("\n[1/5] Loading Delhi geometry...")
    delhi_geom, LAT, LON = load_delhi_geometry()
    print(f"  bbox: LAT {LAT}, LON {LON}")

    print("\n[2/5] Loading ERA5 datasets + computing daily VI...")
    wind_d, t2m_d, t925_d, tp_d, vi_daily = load_all_data(LAT, LON)

    print("\n[3/5] Building India-ASI stagnant mask...")
    stag_mask = build_stagnant_mask(wind_d, t2m_d, t925_d, tp_d)
    stag_pct = float(stag_mask.mean()) * 100
    print(f"  {len(vi_daily.time)} days, {stag_pct:.1f}% stagnant overall")

    print("\n[4/5] Computing VI stagnant anomaly per year-month...")
    anomalies = compute_vi_stagnant_anomaly(vi_daily, stag_mask)

    print(f"\n[5/5] Plotting {len(anomalies)} maps...")
    n = 0
    for (year, month), anom_da in sorted(anomalies.items()):
        out_path = OUT_DIR / f"vi_stagnant_anomaly_{year}_{month:02d}.png"
        plot_vi_anomaly(anom_da, year, month, delhi_geom, LAT, LON, out_path)
        n += 1
        print(f"  saved {out_path.name}")

    print(f"\nDone. {n} plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()