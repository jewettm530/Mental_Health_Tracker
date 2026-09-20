"""Optional passive context sources: weather/daylight and screen-time imports.

Weather is fetched from Open-Meteo only when config/context.json exists and
contains a location. The project stores daily environmental *opportunity*
(daylight, sunshine, cloud cover, precipitation, temperature, radiation), not
personal exposure.

Personal daylight exposure is handled separately through Apple Health's
Time in Daylight metric when present in the Health export.

Screen Time note:
Apple's built-in iPhone Screen Time history is not directly available to this
Python project. If a future app/export produces a canonical daily CSV at
``data/raw/screen_time_daily.csv``, this module normalizes and merges it.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.file_utils import ensure_dir, save_csv, save_json
from utils.paths import (
    APPLE_CLEAN_DIR,
    CONTEXT_CLEAN_DIR,
    CONTEXT_CONFIG,
    SCREEN_TIME_RAW,
    STOIC_CLEAN_DIR,
)

WEATHER_DAILY_VARS = [
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "precipitation_sum",
    "precipitation_hours",
    "shortwave_radiation_sum",
]
WEATHER_HOURLY_VARS = ["cloud_cover"]


def _load_context_config(path: Path = CONTEXT_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Context config could not be read ({path}): {exc}")
        return {}


def _infer_tracking_range() -> tuple[str | None, str | None]:
    """Infer the useful date range from processed Stoic/Apple daily files."""
    candidates: list[pd.Series] = []
    for path in [
        STOIC_CLEAN_DIR / "stoic_mental_health_daily.csv",
        APPLE_CLEAN_DIR / "apple_daily_activity.csv",
        APPLE_CLEAN_DIR / "apple_daily_heart.csv",
    ]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["date"])
        except Exception:
            continue
        s = pd.to_datetime(df.get("date"), errors="coerce").dropna()
        if not s.empty:
            candidates.append(s)
    if not candidates:
        return None, None
    all_dates = pd.concat(candidates, ignore_index=True)
    return all_dates.min().date().isoformat(), all_dates.max().date().isoformat()


def _geocode_location(name: str) -> tuple[float, float, str | None] | None:
    params = urllib.parse.urlencode({"name": name, "count": 1, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    results = payload.get("results") or []
    if not results:
        return None
    item = results[0]
    return float(item["latitude"]), float(item["longitude"]), item.get("timezone")


def _resolve_location(config: dict[str, Any]) -> tuple[float, float, str] | None:
    lat = config.get("latitude")
    lon = config.get("longitude")
    timezone = str(config.get("timezone") or "auto")
    if lat is not None and lon is not None:
        return float(lat), float(lon), timezone
    name = str(config.get("location_name") or "").strip()
    if not name:
        return None
    try:
        result = _geocode_location(name)
    except Exception as exc:
        print(f"Could not geocode context location '{name}': {exc}")
        return None
    if result is None:
        print(f"No geocoding result found for context location '{name}'.")
        return None
    lat, lon, resolved_tz = result
    if timezone == "auto" and resolved_tz:
        timezone = resolved_tz
    return lat, lon, timezone


def fetch_weather_daily(output_dir: Path = CONTEXT_CLEAN_DIR) -> pd.DataFrame:
    """Fetch daily historical environmental context from Open-Meteo.

    This is location-level weather/daylight availability, not proof of the
    user's personal sun/daylight exposure.
    """
    ensure_dir(output_dir)
    config = _load_context_config()
    location = _resolve_location(config)
    if location is None:
        print("Weather context skipped: add config/context.json with location_name or latitude/longitude.")
        return pd.DataFrame(columns=["date"])

    start_date, end_date = _infer_tracking_range()
    start_date = str(config.get("start_date") or start_date or "2024-01-01")
    end_date = str(config.get("end_date") or end_date or date.today().isoformat())
    lat, lon, timezone = location

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "temperature_unit": "fahrenheit",
        "daily": ",".join(WEATHER_DAILY_VARS),
        "hourly": ",".join(WEATHER_HOURLY_VARS),
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"Weather context fetch failed: {exc}")
        return pd.DataFrame(columns=["date"])

    daily = payload.get("daily") or {}
    if not daily.get("time"):
        print("Weather context fetch returned no daily rows.")
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame(daily).rename(columns={
        "time": "date",
        "temperature_2m_mean": "weather_temperature_mean_f",
        "apparent_temperature_mean": "weather_apparent_temperature_mean_f",
        "sunrise": "weather_sunrise",
        "sunset": "weather_sunset",
        "daylight_duration": "weather_daylight_seconds",
        "sunshine_duration": "weather_sunshine_seconds",
        "precipitation_sum": "weather_precipitation_mm",
        "precipitation_hours": "weather_precipitation_hours",
        "shortwave_radiation_sum": "weather_shortwave_radiation_mj_m2",
    })
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["weather_daylight_hours"] = pd.to_numeric(out.get("weather_daylight_seconds"), errors="coerce") / 3600
    out["weather_sunshine_hours"] = pd.to_numeric(out.get("weather_sunshine_seconds"), errors="coerce") / 3600
    denom = out["weather_daylight_hours"].replace(0, np.nan)
    out["weather_sunshine_fraction_pct"] = out["weather_sunshine_hours"] / denom * 100

    hourly = payload.get("hourly") or {}
    if hourly.get("time") and hourly.get("cloud_cover"):
        h = pd.DataFrame({"time": hourly["time"], "cloud_cover": hourly["cloud_cover"]})
        h["time"] = pd.to_datetime(h["time"], errors="coerce")
        h["date"] = h["time"].dt.normalize()
        cloud = h.groupby("date", as_index=False)["cloud_cover"].mean().rename(columns={"cloud_cover": "weather_cloud_cover_mean_pct"})
        out = out.merge(cloud, on="date", how="left")

    # Internal seconds are not useful dashboard variables once hours are created.
    out = out.drop(columns=[c for c in ["weather_daylight_seconds", "weather_sunshine_seconds"] if c in out.columns])
    save_csv(out, output_dir / "weather_daily.csv")
    save_json(
        {
            "source": "Open-Meteo Historical Weather API",
            "latitude": lat,
            "longitude": lon,
            "timezone": payload.get("timezone", timezone),
            "start_date": start_date,
            "end_date": end_date,
            "rows": len(out),
            "note": "Environmental opportunity only; not personal daylight exposure.",
        },
        output_dir / "weather_inventory.json",
    )
    return out


SCREEN_TIME_ALIASES = {
    "date": ["date", "day"],
    "screen_time_total_minutes": ["screen_time_total_minutes", "total_minutes", "screen_time_minutes"],
    "screen_time_late_night_minutes": ["screen_time_late_night_minutes", "late_night_minutes"],
    "screen_time_social_minutes": ["screen_time_social_minutes", "social_minutes"],
    "screen_time_entertainment_minutes": ["screen_time_entertainment_minutes", "entertainment_minutes"],
    "screen_time_productivity_minutes": ["screen_time_productivity_minutes", "productivity_minutes"],
    "screen_time_pickups": ["screen_time_pickups", "pickups"],
    "screen_time_notifications": ["screen_time_notifications", "notifications"],
}


def process_screen_time_csv(raw_path: Path = SCREEN_TIME_RAW, output_dir: Path = CONTEXT_CLEAN_DIR) -> pd.DataFrame:
    """Normalize an optional future daily Screen Time export.

    This function does not collect Apple's built-in Screen Time itself; it is an
    import contract for a future exporter/tracker that can produce daily data.
    """
    ensure_dir(output_dir)
    if not raw_path.exists():
        return pd.DataFrame(columns=["date"])
    try:
        raw = pd.read_csv(raw_path, low_memory=False)
    except Exception as exc:
        print(f"Screen-time CSV could not be read: {exc}")
        return pd.DataFrame(columns=["date"])

    lower_to_original = {str(c).strip().lower(): c for c in raw.columns}
    data: dict[str, pd.Series] = {}
    for canonical, aliases in SCREEN_TIME_ALIASES.items():
        original = next((lower_to_original[a.lower()] for a in aliases if a.lower() in lower_to_original), None)
        if original is not None:
            data[canonical] = raw[original]
    if "date" not in data:
        print("Screen-time CSV skipped: no date/day column found.")
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame(data)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date"]).groupby("date", as_index=False).mean(numeric_only=True)
    save_csv(out, output_dir / "screen_time_daily.csv")
    return out


def process_context_sources(output_dir: Path = CONTEXT_CLEAN_DIR) -> dict[str, pd.DataFrame]:
    ensure_dir(output_dir)
    weather = fetch_weather_daily(output_dir)
    screen = process_screen_time_csv(SCREEN_TIME_RAW, output_dir)
    return {"weather": weather, "screen_time": screen}
