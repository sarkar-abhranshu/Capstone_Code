import json
import pandas as pd
import numpy as np
from pathlib import Path

HISTORICAL = Path("../csv/fertility_index_data.csv")
FRONTEND_DATA_DIR = Path("../frontend_data")
OUTPUT = Path("data.json")

CLASS_ORDER = {"Low": 0, "Medium": 1, "High": 2}
CLASS_COLORS = {"Low": "#ef4444", "Medium": "#eab308", "High": "#22c55e"}

hist = pd.read_csv(HISTORICAL)
hist["date"] = pd.to_datetime(hist["date"])
hist["site_id"] = (
    hist["longitude"].round(6).astype(str) + "_" + hist["latitude"].round(6).astype(str)
)

# Load all predictions from frontend_data
pred_dfs = []
models_list = []
for file_path in FRONTEND_DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file_path)
    # Extract config from filename, e.g., 'l12_h12' from 'l12_h12_bilstm.csv'
    config_parts = file_path.stem.split("_")
    model_name = f"{config_parts[0].upper()}-{config_parts[1].upper()}"
    df["model"] = model_name
    pred_dfs.append(df)
    models_list.append(model_name)

# Sort models so they appear in a consistent order
models_list = sorted(models_list)

pred = pd.concat(pred_dfs, ignore_index=True)
pred["target_date"] = pd.to_datetime(pred["target_date"])

# Latest historical fertility per site
latest_hist = (
    hist.sort_values("date")
    .groupby("site_id")
    .last()
    .reset_index()
)

# Historical time-series per site
def build_history(gf):
    rows = gf.sort_values("date")
    return [
        {"d": r.date.strftime("%Y-%m"), "f": round(r.FertilityIndex, 4), "c": r.FertilityClass}
        for _, r in rows.iterrows()
    ]

history_map = hist.groupby("site_id").apply(build_history, include_groups=False).to_dict()

# Predictions per model per site
def build_forecast(gf):
    rows = gf.sort_values("target_date")
    return [{"d": r.target_date.strftime("%Y-%m"), "f": round(r.predicted, 4)} for _, r in rows.iterrows()]

forecast_map = {}
for model in pred["model"].unique():
    mpred = pred[pred["model"] == model]
    forecast_map[model] = mpred.groupby("site_id").apply(build_forecast, include_groups=False).to_dict()

# Build sites array
sites = []
for _, row in latest_hist.iterrows():
    sid = row["site_id"]
    lat, lng = row["latitude"], row["longitude"]
    cf = round(row["FertilityIndex"], 4)
    cc = row["FertilityClass"]

    fp = {}
    for model in models_list:
        fcasts = forecast_map.get(model, {}).get(sid, [])
        fp[model] = fcasts

    sites.append({
        "id": sid,
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "cf": cf,
        "cc": cc,
        "history": history_map.get(sid, []),
        "forecast": fp,
    })

output = {
    "sites": sites,
    "classColors": CLASS_COLORS,
    "classOrder": CLASS_ORDER,
    "models": models_list,
}

with open(OUTPUT, "w") as f:
    json.dump(output, f, separators=(",", ":"))

print(f"Wrote {len(sites)} sites to {OUTPUT}")
