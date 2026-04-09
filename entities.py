"""
entities.py
===========
Data entity classes for the Extreme Heat Prediction System.

Defines the three core data structures used throughout the pipeline:
  - WeatherRecord  : conventional meteorological observation
  - ImpactRecord   : impact-centric / derived environmental variables
  - HeatEvent      : classified heat event with risk metadata

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Heat event severity levels based on heat-index thresholds."""
    NORMAL    = "normal"       # heat index < 32 °C
    CAUTION   = "caution"      # 32–40 °C
    DANGER    = "danger"       # 40–51 °C
    EXTREME   = "extreme"      # > 51 °C


# ---------------------------------------------------------------------------
# WeatherRecord
# ---------------------------------------------------------------------------

@dataclass
class WeatherRecord:
    """
    Represents a single conventional meteorological observation.

    Attributes
    ----------
    timestamp    : Observation datetime (UTC or local).
    temperature  : Dry-bulb air temperature in degrees Celsius.
    humidity     : Relative humidity in percent (0–100).
    wind_speed   : Wind speed in metres per second.
    location     : Station name or coordinate label (e.g. 'Davao City').
    """

    timestamp   : datetime
    temperature : float
    humidity    : float
    wind_speed  : float
    location    : str = "Davao City"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Check that all meteorological values fall within physically
        plausible ranges.

        Returns
        -------
        bool
            True if the record is valid, False otherwise.
        """
        if not (0.0 <= self.humidity <= 100.0):
            return False
        if not (-10.0 <= self.temperature <= 60.0):   # tropical city range
            return False
        if not (0.0 <= self.wind_speed <= 100.0):
            return False
        return True

    # ------------------------------------------------------------------
    # Sequence helper
    # ------------------------------------------------------------------

    def to_sequence(self) -> List[float]:
        """
        Return the record as an ordered list of floats suitable for
        feeding into a NumPy array or LSTM input tensor.

        Returns
        -------
        list[float]
            [temperature, humidity, wind_speed]
        """
        return [self.temperature, self.humidity, self.wind_speed]

    # ------------------------------------------------------------------
    # Normalisation (min-max, uses domain knowledge bounds)
    # ------------------------------------------------------------------

    def normalize(
        self,
        temp_range: tuple[float, float]  = (20.0, 45.0),
        hum_range : tuple[float, float]  = (0.0,  100.0),
        wind_range: tuple[float, float]  = (0.0,  20.0),
    ) -> List[float]:
        """
        Apply min-max normalisation to the meteorological variables.

        Parameters
        ----------
        temp_range  : (min, max) for temperature normalisation.
        hum_range   : (min, max) for humidity normalisation.
        wind_range  : (min, max) for wind-speed normalisation.

        Returns
        -------
        list[float]
            Normalised [temperature, humidity, wind_speed] in [0, 1].
        """
        def _scale(val: float, lo: float, hi: float) -> float:
            return (val - lo) / (hi - lo + 1e-8)

        return [
            _scale(self.temperature, *temp_range),
            _scale(self.humidity,    *hum_range),
            _scale(self.wind_speed,  *wind_range),
        ]

    def __repr__(self) -> str:
        return (
            f"WeatherRecord({self.timestamp.isoformat()}, "
            f"T={self.temperature:.1f}°C, "
            f"RH={self.humidity:.1f}%, "
            f"WS={self.wind_speed:.1f} m/s, "
            f"loc='{self.location}')"
        )


# ---------------------------------------------------------------------------
# ImpactRecord
# ---------------------------------------------------------------------------

@dataclass
class ImpactRecord:
    """
    Stores impact-centric / derived environmental variables for one
    observation period.

    These variables capture the *human health impact* of heat beyond
    raw temperature, and are the key additions in the impact-centric
    LSTM model.

    Attributes
    ----------
    timestamp     : Matches the corresponding WeatherRecord timestamp.
    uhi_intensity : Urban Heat Island intensity in °C (urban − rural Δ T).
    wet_bulb_temp : Wet-bulb temperature in °C (heat + humidity stress).
    pm25_level    : PM 2.5 particulate concentration in μg/m³.
    heat_index    : Apparent temperature ("feels-like") in °C.
    """

    timestamp     : datetime
    uhi_intensity : float
    wet_bulb_temp : float
    pm25_level    : float
    heat_index    : float

    # ------------------------------------------------------------------
    # Derived computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_heat_index(temp_c: float, rh: float) -> float:
        """
        Estimate the heat index (Rothfusz regression, °C).

        The formula is derived from the US NWS Rothfusz equation
        and converted from Fahrenheit to Celsius.

        Parameters
        ----------
        temp_c : Dry-bulb temperature in Celsius.
        rh     : Relative humidity in percent.

        Returns
        -------
        float
            Apparent temperature ("feels-like") in Celsius.
        """
        T = temp_c * 9 / 5 + 32   # convert to Fahrenheit for formula
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
        return (HI - 32) * 5 / 9   # back to Celsius

    @staticmethod
    def compute_wet_bulb(temp_c: float, rh: float) -> float:
        """
        Approximate wet-bulb temperature using the Stull (2011) formula.

        Stull, R. (2011). Wet-bulb temperature from relative humidity and
        air temperature. *Journal of Applied Meteorology and Climatology*,
        50(11), 2267-2269.

        Parameters
        ----------
        temp_c : Dry-bulb temperature in Celsius.
        rh     : Relative humidity in percent.

        Returns
        -------
        float
            Wet-bulb temperature in Celsius.
        """
        Tw = (
            temp_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(temp_c + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035
        )
        return Tw

    def flag_dangerous(self, pm25_threshold: float = 55.4) -> bool:
        """
        Determine whether this observation represents a dangerous
        combination of heat and air quality.

        Parameters
        ----------
        pm25_threshold : PM 2.5 level (μg/m³) above which air quality
                         is considered "Unhealthy" (US EPA standard).

        Returns
        -------
        bool
            True if heat index ≥ 40 °C **and** PM 2.5 > threshold.
        """
        return self.heat_index >= 40.0 and self.pm25_level > pm25_threshold

    def __repr__(self) -> str:
        return (
            f"ImpactRecord({self.timestamp.isoformat()}, "
            f"HI={self.heat_index:.1f}°C, "
            f"WB={self.wet_bulb_temp:.1f}°C, "
            f"UHI={self.uhi_intensity:.2f}°C, "
            f"PM2.5={self.pm25_level:.1f} μg/m³)"
        )


# ---------------------------------------------------------------------------
# HeatEvent
# ---------------------------------------------------------------------------

@dataclass
class HeatEvent:
    """
    Represents a classified extreme heat event with risk metadata.

    HeatEvents are generated from model predictions and used by the
    AlertModule to dispatch public warnings.

    Attributes
    ----------
    event_id       : Unique integer identifier for the event.
    severity       : Severity enum (NORMAL / CAUTION / DANGER / EXTREME).
    start_date     : Predicted or observed event start datetime.
    location       : Affected location string.
    predicted_risk : Continuous risk score in [0, 1] from the LSTM model.
    """

    event_id       : int
    severity       : Severity
    start_date     : datetime
    location       : str
    predicted_risk : float

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_severity(heat_index: float) -> Severity:
        """
        Map a heat index value to a Severity enum using thresholds
        recommended by PAGASA / US NWS.

        Parameters
        ----------
        heat_index : Apparent temperature in Celsius.

        Returns
        -------
        Severity
            Corresponding severity level.
        """
        if heat_index < 32.0:
            return Severity.NORMAL
        elif heat_index < 40.0:
            return Severity.CAUTION
        elif heat_index < 51.0:
            return Severity.DANGER
        else:
            return Severity.EXTREME

    def is_extreme(self) -> bool:
        """
        Return True if the event is classified as EXTREME severity.

        Returns
        -------
        bool
        """
        return self.severity == Severity.EXTREME

    def generate_alert(self) -> str:
        """
        Produce a human-readable alert message for this event.

        Returns
        -------
        str
            Formatted alert string ready for dispatch.
        """
        level_map = {
            Severity.NORMAL  : "ℹ️  Normal conditions",
            Severity.CAUTION : "⚠️  CAUTION – Heat advisory",
            Severity.DANGER  : "🔴 DANGER – Extreme heat warning",
            Severity.EXTREME : "🚨 EXTREME HEAT EMERGENCY",
        }
        header = level_map[self.severity]
        return (
            f"{header}\n"
            f"  Event ID   : {self.event_id}\n"
            f"  Location   : {self.location}\n"
            f"  Start Date : {self.start_date.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Risk Score : {self.predicted_risk:.3f}\n"
            f"  Severity   : {self.severity.value.upper()}"
        )

    def __repr__(self) -> str:
        return (
            f"HeatEvent(id={self.event_id}, "
            f"severity={self.severity.value}, "
            f"risk={self.predicted_risk:.3f}, "
            f"loc='{self.location}')"
        )