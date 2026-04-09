"""
data_generator.py
=================
Synthetic time-series data generator for a tropical city (Davao, Philippines).

Produces realistic hourly weather and impact-centric observations including
occasional extreme heat events, following seasonal and diurnal patterns
representative of the Davao Region climate.

Usage
-----
    from data_generator import generate_davao_dataset
    weather_df, impact_df = generate_davao_dataset(n_days=365)

Real Data Sources (for actual thesis deployment)
-------------------------------------------------
- PAGASA (Philippine Atmospheric, Geophysical and Astronomical Services
  Administration): https://www.pagasa.dost.gov.ph/
- ERA5 Reanalysis (Copernicus / ECMWF):
  https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
- OpenAQ (PM2.5 / air quality): https://openaq.org/
- MODIS / Landsat LST (Urban Heat Island satellite):
  https://earthexplorer.usgs.gov/
- NASA POWER (solar / meteorological): https://power.larc.nasa.gov/

Author : [Your Name]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Random-seed helper
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_davao_dataset(
    n_days       : int  = 365,
    start_date   : str  = "2022-01-01",
    freq         : str  = "h",          # hourly observations
    extreme_prob : float = 0.05,        # probability of an extreme-heat hour
    noise_scale  : float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic hourly weather and impact records for Davao City.

    Davao City (7.07°N, 125.61°E) sits outside the Philippine typhoon belt
    and has a relatively uniform annual temperature (~27–35 °C day) with
    two weak seasons (dry: Mar–May, wet: Jun–Feb).

    Parameters
    ----------
    n_days       : Number of days to simulate.
    start_date   : ISO date string for the first observation.
    freq         : Pandas frequency string ('h' = hourly).
    extreme_prob : Fraction of hours flagged as extreme heat.
    noise_scale  : Standard deviation of Gaussian noise added to signals.

    Returns
    -------
    weather_df : DataFrame with columns
                 [timestamp, temperature, humidity, wind_speed, location].
    impact_df  : DataFrame with columns
                 [timestamp, uhi_intensity, wet_bulb_temp, pm25_level,
                  heat_index, is_extreme].
    """

    dt_index = pd.date_range(
        start=start_date, periods=n_days * 24, freq=freq
    )
    n = len(dt_index)

    # ------------------------------------------------------------------
    # Time-based signals
    # ------------------------------------------------------------------
    hour_of_day  = np.array([ts.hour     for ts in dt_index], dtype=float)
    day_of_year  = np.array([ts.dayofyear for ts in dt_index], dtype=float)

    # Diurnal cycle: peak at ~14:00, trough at ~05:00
    diurnal = np.sin(2 * np.pi * (hour_of_day - 5) / 24)

    # Annual dry season: March (day ~60) – May (day ~150)
    seasonal = 0.5 * np.sin(2 * np.pi * (day_of_year - 60) / 365)

    # ------------------------------------------------------------------
    # Temperature (°C)  – Davao mean ≈ 27 °C, daytime peak ≈ 33–35 °C
    # ------------------------------------------------------------------
    temp_base = 27.0 + 4.0 * diurnal + 2.5 * seasonal
    temp_base += RNG.normal(0, noise_scale, n)
    temp_base = np.clip(temp_base, 22.0, 42.0)

    # Inject extreme heat spikes during dry season
    extreme_mask = (
        (RNG.random(n) < extreme_prob) &
        ((day_of_year >= 60) & (day_of_year <= 150))
    )
    temp_base[extreme_mask] += RNG.uniform(3, 8, extreme_mask.sum())
    temp_base = np.clip(temp_base, 22.0, 45.0)

    # ------------------------------------------------------------------
    # Relative Humidity (%)  – inversely correlated with temperature
    # ------------------------------------------------------------------
    humidity = 90.0 - 1.4 * (temp_base - 27.0) + RNG.normal(0, 3, n)
    humidity = np.clip(humidity, 30.0, 99.0)

    # ------------------------------------------------------------------
    # Wind Speed (m/s)  – 1–5 m/s typical for Davao
    # ------------------------------------------------------------------
    wind_speed = (
        2.0
        + 1.5 * np.sin(2 * np.pi * hour_of_day / 24 + np.pi)
        + RNG.exponential(0.4, n)
    )
    wind_speed = np.clip(wind_speed, 0.0, 12.0)

    # ------------------------------------------------------------------
    # Urban Heat Island intensity (°C)
    # ------------------------------------------------------------------
    # Higher during calm, clear nights; lower during daytime convection
    uhi = (
        1.5
        + 0.8 * (1 - diurnal)          # stronger at night
        + 0.3 * seasonal
        + RNG.normal(0, 0.2, n)
    )
    uhi = np.clip(uhi, 0.0, 5.0)

    # ------------------------------------------------------------------
    # PM 2.5 (μg/m³)  – background 10–25; spikes near burning season
    # ------------------------------------------------------------------
    pm25 = 15.0 + 8.0 * seasonal + RNG.exponential(5.0, n)
    # Burning season spike (March–April)
    burning = (day_of_year >= 60) & (day_of_year <= 120)
    pm25[burning] += RNG.exponential(15, burning.sum())
    pm25 = np.clip(pm25, 1.0, 250.0)

    # ------------------------------------------------------------------
    # Derived: Wet-Bulb Temperature  (Stull 2011 approximation)
    # ------------------------------------------------------------------
    import math as _math

    def _wet_bulb(T: float, rh: float) -> float:
        return (
            T * _math.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + _math.atan(T + rh)
            - _math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * _math.atan(0.023101 * rh)
            - 4.686035
        )

    wet_bulb = np.array([
        _wet_bulb(float(T), float(rh))
        for T, rh in zip(temp_base, humidity)
    ])

    # ------------------------------------------------------------------
    # Derived: Heat Index  (Rothfusz regression, converted to °C)
    # ------------------------------------------------------------------
    def _heat_index(T_c: float, rh: float) -> float:
        T = T_c * 9 / 5 + 32
        HI = (
            -42.379
            + 2.04901523   * T
            + 10.14333127  * rh
            - 0.22475541   * T  * rh
            - 0.00683783   * T  * T
            - 0.05481717   * rh * rh
            + 0.00122874   * T  * T * rh
            + 0.00085282   * T  * rh * rh
            - 0.00000199   * T  * T * rh * rh
        )
        return (HI - 32) * 5 / 9

    heat_index = np.array([
        _heat_index(float(T), float(rh))
        for T, rh in zip(temp_base, humidity)
    ])

    # ------------------------------------------------------------------
    # Extreme heat binary label
    # ------------------------------------------------------------------
    # 1 = heat index ≥ 40 °C  OR  temperature ≥ 38 °C  (PAGASA advisory)
    is_extreme = ((heat_index >= 40.0) | (temp_base >= 38.0)).astype(int)

    # ------------------------------------------------------------------
    # Assemble DataFrames
    # ------------------------------------------------------------------
    weather_df = pd.DataFrame({
        "timestamp"   : dt_index,
        "temperature" : temp_base.round(2),
        "humidity"    : humidity.round(2),
        "wind_speed"  : wind_speed.round(2),
        "location"    : "Davao City",
    })

    impact_df = pd.DataFrame({
        "timestamp"    : dt_index,
        "uhi_intensity": uhi.round(3),
        "wet_bulb_temp": wet_bulb.round(2),
        "pm25_level"   : pm25.round(2),
        "heat_index"   : heat_index.round(2),
        "is_extreme"   : is_extreme,
    })

    print(
        f"[DataGenerator] Generated {n:,} hourly records over {n_days} days.\n"
        f"  Extreme heat hours : {is_extreme.sum():,} "
        f"({100 * is_extreme.mean():.1f}%)\n"
        f"  Temp range         : {temp_base.min():.1f} – {temp_base.max():.1f} °C\n"
        f"  Heat index range   : {heat_index.min():.1f} – {heat_index.max():.1f} °C"
    )

    return weather_df, impact_df


# ---------------------------------------------------------------------------
# Convenience: save to CSV
# ---------------------------------------------------------------------------

def save_datasets(
    weather_df : pd.DataFrame,
    impact_df  : pd.DataFrame,
    output_dir : str = "data",
) -> None:
    """
    Persist both DataFrames as CSV files.

    Parameters
    ----------
    weather_df : WeatherRecord DataFrame.
    impact_df  : ImpactRecord DataFrame.
    output_dir : Directory in which to save the files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    wp = os.path.join(output_dir, "weather_records.csv")
    ip = os.path.join(output_dir, "impact_records.csv")
    weather_df.to_csv(wp, index=False)
    impact_df.to_csv(ip,  index=False)
    print(f"[DataGenerator] Saved → {wp}")
    print(f"[DataGenerator] Saved → {ip}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wdf, idf = generate_davao_dataset(n_days=730)
    save_datasets(wdf, idf, output_dir="data")
    print("\nSample weather rows:")
    print(wdf.head(3).to_string(index=False))
    print("\nSample impact rows:")
    print(idf.head(3).to_string(index=False))