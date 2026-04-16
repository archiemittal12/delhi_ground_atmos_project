"""
ASD390 Week 6 — Task 3
Per-condition stagnation anomaly maps (Delhi-zoomed).

Uses NEW daily files from professor (in week_6/data/):
  - era5_daily_t2m_2021_2024.nc      (variable: '2t', Celsius, lat/lon)
  - era5_daily_t925hpa_2021_2024.nc  (variable: 't', Celsius, plev=92500, lat/lon)
  - era5_daily_tp_2021_2024.nc       (variable: 'tp', units VERIFY)

Wind files unchanged (also in week_6/data/):
  - ERA5_WS_daily_2021.nc            (variable: 'ws10m', m/s, latitude/longitude)
  - ERA5_WS_daily_2022_2024.nc

Output: 60 PNGs (3 conditions x 4 years x 5 months) in week_6/task3_outputs/
"""
from pathlib import Path
import numpy as np
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

# *** UPDATE if you put the shapefile elsewhere ***
SHAPEFILE = DATA_DIR / "shapefiles" / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp"

OUT_DIR      = PROJECT_ROOT / "week_6" / "task3_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS  = [2021, 2022, 2023, 2024]
MONTHS = [10, 11, 12, 1, 2]

# Thresholds
WS_THRESHOLD = 3.2     # m/s
TP_THRESHOLD = 1.0     # mm  (assumes precip variable is in mm; verify!)
# Inversion threshold is a *difference* so K vs C doesn't matter

COMMON_START = "2021-01-01"
COMMON_END   = "2024-12-31"

# Variable names — confirmed from print(ds) output
WIND_VAR  = "ws10m"
T2M_VAR   = "2t"          # ECMWF convention — kept the weird name
T925_VAR  = "t"
TP_VAR    = "tp"

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

CONDITIONS = {
    "wind": {
        "title_short": "Calm wind",
        "title_long":  "Calm wind (ws10m < 3.2 m/s)",
        "cbar_label":  "Calm-wind day anomaly (Δ days / month)",
    },
    "inversion": {
        "title_short": "Temperature inversion",
        "title_long":  "Temperature inversion (T$_{925}$ > T$_{2m}$)",
        "cbar_label":  "Inversion day anomaly (Δ days / month)",
    },
    "dry": {
        "title_short": "Dry",
        "title_long":  "Dry day (precip < 1 mm/day)",
        "cbar_label":  "Dry day anomaly (Δ days / month)",
    },
}


# ============================================================
# DATA LOADING
# ============================================================
def normalize_coords(ds):
    rename = {k: v for k, v in COORD_RENAME.items() if k in ds.coords or k in ds.dims}
    if rename:
        ds = ds.rename(rename)
    return ds


def load_era5():
    print("  loading wind...")
    ds_wind = xr.open_mfdataset(
        [str(DATA_DIR / "ERA5_WS_daily_2021.nc"),
         str(DATA_DIR / "ERA5_WS_daily_2022_2024.nc")],
        combine="by_coords",
    )
    print("  loading t2m...")
    ds_t2m  = xr.open_dataset(DATA_DIR / "era5_daily_t2m_2021_2024.nc")
    print("  loading t925...")
    ds_t925 = xr.open_dataset(DATA_DIR / "era5_daily_t925hpa_2021_2024.nc")
    print("  loading precip...")
    ds_tp   = xr.open_dataset(DATA_DIR / "era5_daily_tp_2021_2024.nc")

    ds_wind = normalize_coords(ds_wind)
    ds_t2m  = normalize_coords(ds_t2m)
    ds_t925 = normalize_coords(ds_t925)
    ds_tp   = normalize_coords(ds_tp)

    if "plev" in ds_t925.dims:
        ds_t925 = ds_t925.squeeze("plev", drop=True)

    # ----- SANITY CHECK: precip units -----
    # ERA5 raw is in metres; if professor's daysum kept that, threshold must scale
    sample_max = float(ds_tp[TP_VAR].max())
    print(f"  precip sample max = {sample_max:.4f}")
    if sample_max < 1.0:
        print("  WARNING: precip max < 1.0 — likely in METRES, not mm")
        print("           Auto-converting threshold from mm to metres")
        global TP_THRESHOLD
        TP_THRESHOLD = TP_THRESHOLD / 1000.0
        print(f"           New TP_THRESHOLD = {TP_THRESHOLD} m")

    return ds_wind, ds_t2m, ds_t925, ds_tp


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


def clip_to_delhi(ds, LAT, LON):
    """Latitude in these files goes from +39 down to +5, so use LAT[1]:LAT[0]."""
    return ds.sel(
        latitude=slice(LAT[1], LAT[0]),
        longitude=slice(LON[0], LON[1]),
        time=slice(COMMON_START, COMMON_END),
    )


# ============================================================
# CONDITION MASKS
# ============================================================
def build_condition_masks(wind_d, t2m_d, t925_d, tp_d):
    return {
        "wind":      wind_d[WIND_VAR] < WS_THRESHOLD,
        "inversion": (t925_d[T925_VAR] - t2m_d[T2M_VAR]) > 0,
        "dry":       tp_d[TP_VAR] < TP_THRESHOLD,
    }


def compute_anomaly(mask):
    count = mask.resample(time="ME").sum()
    count = count.sel(time=count["time"].dt.month.isin(MONTHS))
    climatology = count.groupby("time.month").mean(dim="time")
    anomaly     = count.groupby("time.month") - climatology
    return count, climatology, anomaly


# ============================================================
# PLOTTING
# ============================================================
def geometry_mask(geom, x, y):
    pts  = np.column_stack([x.ravel(), y.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        ext    = MplPath(np.asarray(poly.exterior.coords))
        inside = ext.contains_points(pts)
        for hole in poly.interiors:
            inside &= ~MplPath(np.asarray(hole.coords)).contains_points(pts)
        mask |= inside
    return mask.reshape(x.shape)


def plot_condition_map(condition_key, anomaly, year, month,
                        delhi_geom, LAT, LON, out_path):
    meta = CONDITIONS[condition_key]
    anom_plot = anomaly.sel(time=f"{year}-{month:02d}")

    ngrid = 300
    grid_lon = np.linspace(LON[0], LON[1], ngrid)
    grid_lat = np.linspace(LAT[0], LAT[1], ngrid)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    lon2d, lat2d = np.meshgrid(
        anom_plot.longitude.values,
        anom_plot.latitude.values,
    )
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    values = anom_plot.values.ravel()

    grid_z    = griddata(points, values, (grid_lon2d, grid_lat2d), method="linear")
    grid_z_nn = griddata(points, values, (grid_lon2d, grid_lat2d), method="nearest")
    grid_z    = np.where(np.isnan(grid_z), grid_z_nn, grid_z)
    grid_z    = gaussian_filter(grid_z, sigma=3)

    inside        = geometry_mask(delhi_geom, grid_lon2d, grid_lat2d)
    grid_z_masked = np.where(inside, grid_z, np.nan)

    vmax   = max(np.nanpercentile(np.abs(grid_z_masked), 98), 0.1)
    levels = np.linspace(-vmax, vmax, 25)

    fig = plt.figure(figsize=(9, 9))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    contour = ax.contourf(
        grid_lon2d, grid_lat2d, grid_z_masked,
        levels=levels, cmap="RdBu_r", extend="both",
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
    cbar.set_label(meta["cbar_label"], fontsize=12)

    n_cells = len(anom_plot.latitude) * len(anom_plot.longitude)
    ax.text(0.01, 0.01, f"ERA5 0.25° grid: {n_cells} cells in domain",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    legend_elements = [
        Line2D([0], [0], marker="+", color="k", label="ERA5 grid pt",
               linestyle="None", ms=9),
        Line2D([0], [0], marker="^", color="gold", label="CPCB station",
               linestyle="None", ms=8, markeredgecolor="k", markeredgewidth=0.6),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.title(
        f"{meta['title_short']} Anomaly — Delhi  ({year}-{month:02d})\n"
        f"{meta['title_long']}  |  +ve = more {meta['title_short'].lower()} days",
        fontsize=12, weight="bold",
    )
    plt.subplots_adjust(right=0.85)
    plt.savefig(out_path, dpi=96, bbox_inches="tight")
    plt.close()


# ============================================================
# DRIVER
# ============================================================
def main():
    print("=" * 60)
    print("ASD390 Week 6 — Task 3: Per-condition anomaly maps")
    print("=" * 60)

    print("\n[1/5] Loading ERA5 datasets...")
    ds_wind, ds_t2m, ds_t925, ds_tp = load_era5()

    print("\n[2/5] Loading Delhi geometry...")
    delhi_geom, LAT, LON = load_delhi_geometry()
    print(f"  bbox: LAT {LAT}, LON {LON}")

    print("\n[3/5] Clipping to Delhi domain...")
    wind_d  = clip_to_delhi(ds_wind, LAT, LON)
    t2m_d   = clip_to_delhi(ds_t2m,  LAT, LON)
    t925_d  = clip_to_delhi(ds_t925, LAT, LON)
    tp_d    = clip_to_delhi(ds_tp,   LAT, LON)
    print(f"  grid cells: {len(wind_d.latitude)} lat x {len(wind_d.longitude)} lon")
    print(f"  time steps: {len(wind_d.time)}")

    print("\n[4/5] Building masks and anomalies...")
    masks = build_condition_masks(wind_d, t2m_d, t925_d, tp_d)
    anomalies = {}
    for cond, mask in masks.items():
        count, clim, anom = compute_anomaly(mask)
        anomalies[cond] = anom
        print(f"  {cond:10s}: monthly climatology (domain-avg, days/month):")
        for m in MONTHS:
            print(f"    month {m:02d}: {float(clim.sel(month=m).mean()):5.1f}")

    print("\n[5/5] Plotting 3 x 4 x 5 = 60 maps...")
    n = 0
    for cond in CONDITIONS:
        anom = anomalies[cond]
        for year in YEARS:
            for month in MONTHS:
                try:
                    out_path = OUT_DIR / f"task3_{cond}_{year}_{month:02d}.png"
                    plot_condition_map(cond, anom, year, month,
                                       delhi_geom, LAT, LON, out_path)
                    n += 1
                    if n % 10 == 0:
                        print(f"    {n} plots saved")
                except KeyError:
                    print(f"    no data for {cond} {year}-{month:02d}, skipping")

    print(f"\nDone. {n} plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
