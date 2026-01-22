# scripts/organize_ws_wd_robust.py
import os, re, shutil
from pathlib import Path
import pandas as pd

SOURCE = "station data"
TARGET = "station_data"
YEARS_KEEP = {"2021","2022","2023","2024"}
OUT_SUMMARY = "station_availability_summary_robust.csv"
OUT_STATIONS = "stations_list_robust.csv"
OUT_LOG = "organize_log_robust.txt"

Path(TARGET).mkdir(parents=True, exist_ok=True)

def tidy_colname(s: str) -> str:
    if s is None:
        return ""
    col = str(s).strip()
    col = col.replace("Â","").replace("µ","u").replace("μ","u").replace("³","3").replace("°","deg")
    col = re.sub(r"[()/\\\[\]\{\}\-]", "_", col)
    col = col.replace("%","pct").replace(".","_")
    col = re.sub(r"\s+","_", col)
    col = re.sub(r"[^0-9A-Za-z_]","", col)
    col = re.sub(r"_+","_", col)
    return col.strip("_").lower()

def best_candidate_by_coverage(df, mapping_orig_to_tidy, substrings):
    """
    Return (orig_col, tidy_col) that matches any substring in 'substrings'
    and has the largest count of numeric (coerced) non-NA values.
    """
    candidates = []
    for orig, tidy in mapping_orig_to_tidy.items():
        for s in substrings:
            if s in tidy:
                candidates.append((orig, tidy))
                break
    if not candidates:
        return None, None

    best = (None, None, -1)  # orig, tidy, score
    for orig, tidy in candidates:
        # how many real numeric values exist (coerce to numeric)
        series = df.get(orig)
        # if column not in df (shouldn't happen), skip
        if series is None:
            continue
        num = pd.to_numeric(series, errors="coerce")
        score = int(num.notna().sum())  # number of numeric entries
        if score > best[2]:
            best = (orig, tidy, score)
    return best[0], best[1]

summary_rows=[]
station_rows=[]
logs=[]

if not Path(SOURCE).exists():
    raise FileNotFoundError(f"Source folder not found: {SOURCE}")

for fname in sorted(os.listdir(SOURCE)):
    if not fname.lower().endswith(".csv"):
        continue
    site_match = re.search(r"site_(\d+)", fname, flags=re.I)
    years_found = re.findall(r"(20\d{2})", fname)
    year = years_found[0] if years_found else None
    if not site_match or year is None or year not in YEARS_KEEP:
        logs.append(f"SKIP {fname} (site={bool(site_match)} year={year})")
        continue
    station_id = site_match.group(1)
    parts = re.split(r"site_\d+_", fname, flags=re.I)
    raw_label = parts[1] if len(parts)>1 else fname
    raw_label = raw_label.rsplit(".",1)[0]
    cleaned_label = re.sub(r"[^\w]+","_", raw_label).strip("_").lower()[:40]
    folder_name = f"station_{station_id}_{cleaned_label}"
    station_folder = os.path.join(TARGET, folder_name)
    os.makedirs(station_folder, exist_ok=True)
    src = os.path.join(SOURCE, fname)
    dst = os.path.join(station_folder, f"{year}.csv")
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logs.append(f"COPY_ERROR {fname} -> {e}")
        continue

    try:
        df = pd.read_csv(dst, encoding="latin1", low_memory=True)
    except Exception as e:
        logs.append(f"PANDAS_READ_ERROR {fname} -> {e}")
        summary_rows.append([folder_name,station_id,year,0.0,0.0,0.0])
        station_rows.append([folder_name,station_id,raw_label,cleaned_label])
        continue

    # build mapping orig->tidy (and ensure unique tidy names)
    mapping = {}
    tidy_seen = {}
    for orig in df.columns:
        t = tidy_colname(orig)
        if t in tidy_seen:
            # append suffix to make unique
            suffix = 1
            while f"{t}_{suffix}" in tidy_seen:
                suffix += 1
            t = f"{t}_{suffix}"
        mapping[orig] = t
        tidy_seen[t] = orig

    # choose best PM, WS, WD by coverage (numeric count)
    pm_subs = ["pm2_5","pm25","pm_2_5","pm"]
    ws_subs = ["vws","ws","wind_speed","windspeed","speed","spd"]
    wd_subs = ["wd","wind_dir","winddirection","direction","deg"]

    pm_orig, pm_tidy = best_candidate_by_coverage(df, mapping, pm_subs)
    ws_orig, ws_tidy = best_candidate_by_coverage(df, mapping, ws_subs)
    wd_orig, wd_tidy = best_candidate_by_coverage(df, mapping, wd_subs)

    # rename columns to tidy unique names
    df = df.rename(columns=mapping)

    def avail_pct(col):
        if col is None:
            return 0.0
        ser = df.get(col)
        if ser is None:
            return 0.0
        num = pd.to_numeric(ser, errors="coerce")
        return round(num.notna().mean()*100, 2)

    pm_avail = avail_pct(pm_tidy)
    ws_avail = avail_pct(ws_tidy)
    wd_avail = avail_pct(wd_tidy)

    summary_rows.append([folder_name,station_id,year,pm_avail,ws_avail,wd_avail])
    station_rows.append([folder_name,station_id,raw_label,cleaned_label])
    logs.append(f"FILE: {fname} -> pm={pm_tidy}({pm_avail}%), ws={ws_tidy}({ws_avail}%), wd={wd_tidy}({wd_avail}%)")
    del df

pd.DataFrame(summary_rows,columns=["station_folder","site_id","year","pm25_avail_pct","windspd_avail_pct","winddir_avail_pct"]).to_csv(OUT_SUMMARY,index=False)
pd.DataFrame(station_rows,columns=["station_folder","site_id","raw_label","clean_label"]).drop_duplicates().to_csv(OUT_STATIONS,index=False)
with open(OUT_LOG,"w",encoding="utf-8") as f:
    f.write("\n".join(logs))

print("Done. Written:", OUT_SUMMARY, OUT_STATIONS, OUT_LOG)
