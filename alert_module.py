"""
alert_module.py  /  data_fetch_module.py
========================================
AlertModule     – threshold checking, alert dispatch, logging, reminders.
DataFetchModule – live weather & air-quality data retrieval (API wrappers).

Both modules appear as leaf nodes in the 9-block architecture diagram and
are driven by upstream outputs from EvaluationModule / LSTMModel.

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from entities import HeatEvent, Severity


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("HeatAlert")


# ===========================================================================
# AlertModule
# ===========================================================================

class AlertModule:
    """
    Threshold-based alert dispatcher for extreme heat events.

    Responsibilities
    ----------------
    • Inspect model predictions and decide whether to trigger an alert.
    • Format and "send" public alerts (prints + logs; extend for SMS / API).
    • Notify health agencies for DANGER / EXTREME events.
    • Log all events to a persistent JSON log file.
    • Schedule reminders (returns next reminder datetime).

    Parameters
    ----------
    risk_threshold   : Minimum predicted probability to trigger an alert.
    log_path         : File path for the JSON alert log.
    location         : Default location string for generated HeatEvents.
    """

    # Heat-index thresholds (°C) from PAGASA / US NWS
    SEVERITY_THRESHOLDS: Dict[str, float] = {
        "caution" : 32.0,
        "danger"  : 40.0,
        "extreme" : 51.0,
    }

    def __init__(
        self,
        risk_threshold : float = 0.5,
        log_path       : str   = "outputs/alert_log.json",
        location       : str   = "Davao City",
    ) -> None:
        self.risk_threshold = risk_threshold
        self.log_path       = Path(log_path)
        self.location       = location
        self._event_counter : int = 0
        self._alert_log     : List[Dict] = []

        # Load existing log if present
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    self._alert_log = json.load(f)
                self._event_counter = len(self._alert_log)
                _logger.info(
                    f"Loaded {self._event_counter} existing alert records."
                )
            except (json.JSONDecodeError, IOError):
                self._alert_log = []

    # ------------------------------------------------------------------
    # 1 · Threshold check
    # ------------------------------------------------------------------

    def check_threshold(
        self,
        event_prob   : float,
        heat_index   : Optional[float] = None,
        timestamp    : Optional[datetime] = None,
    ) -> Optional[HeatEvent]:
        """
        Evaluate a predicted risk probability (and optionally a heat index)
        and return a HeatEvent if the alert threshold is exceeded.

        Parameters
        ----------
        event_prob : Predicted extreme-heat probability from LSTMModel.
        heat_index : Current or forecast heat index (°C), if available.
        timestamp  : Event datetime; defaults to now.

        Returns
        -------
        HeatEvent if threshold exceeded, else None.
        """
        ts = timestamp or datetime.now()

        if event_prob < self.risk_threshold:
            return None

        # Determine severity from heat index (if provided) or probability
        if heat_index is not None:
            severity = HeatEvent.classify_severity(heat_index)
        else:
            # Map probability to severity
            if event_prob >= 0.90:
                severity = Severity.EXTREME
            elif event_prob >= 0.75:
                severity = Severity.DANGER
            elif event_prob >= self.risk_threshold:
                severity = Severity.CAUTION
            else:
                severity = Severity.NORMAL

        self._event_counter += 1
        return HeatEvent(
            event_id       = self._event_counter,
            severity       = severity,
            start_date     = ts,
            location       = self.location,
            predicted_risk = event_prob,
        )

    checkThreshold = check_threshold   # diagram alias

    # ------------------------------------------------------------------
    # 2 · Public alert dispatch
    # ------------------------------------------------------------------

    def send_public_alert(self, msg: str) -> None:
        """
        Broadcast an alert message to the public channel.

        In a production deployment this method would integrate with:
          • NDRRMC / PAGASA alert APIs
          • SMS gateways (e.g. Twilio)
          • Social media / push notification services

        For the thesis prototype it prints and logs the message.

        Parameters
        ----------
        msg : Formatted alert string (typically from HeatEvent.generate_alert()).
        """
        border = "=" * 60
        print(f"\n{border}")
        print("  📢  PUBLIC HEAT ADVISORY")
        print(border)
        print(msg)
        print(f"{border}\n")
        _logger.warning(f"PUBLIC ALERT DISPATCHED: {msg[:80]}…")

    sendPublicAlert = send_public_alert   # diagram alias

    # ------------------------------------------------------------------
    # 3 · Health agency notification
    # ------------------------------------------------------------------

    def notify_health_agency(self) -> None:
        """
        Notify the relevant health authority (DOH / City Health Office)
        when a DANGER or EXTREME event is detected.

        Placeholder: would call a REST endpoint or send an email in production.
        """
        _logger.critical(
            "HEALTH AGENCY NOTIFIED — EXTREME / DANGER heat event detected. "
            "Please activate heat emergency response protocols."
        )
        print(
            "[AlertModule] 🏥  Health agency notification sent "
            "(DOH / City Health Office)."
        )

    notifyHealthAgency = notify_health_agency   # diagram alias

    # ------------------------------------------------------------------
    # 4 · Logging
    # ------------------------------------------------------------------

    def log_alert(self, event_id: int) -> None:
        """
        Persist an alert event record to the JSON log file.

        Parameters
        ----------
        event_id : Identifier of the HeatEvent to record.
        """
        record = {
            "event_id"  : event_id,
            "logged_at" : datetime.now().isoformat(),
            "location"  : self.location,
        }
        self._alert_log.append(record)

        os.makedirs(self.log_path.parent, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self._alert_log, f, indent=2)

        _logger.info(f"Alert event_id={event_id} logged → '{self.log_path}'.")

    logAlert = log_alert   # diagram alias

    # ------------------------------------------------------------------
    # 5 · Reminder scheduler
    # ------------------------------------------------------------------

    def schedule_reminder(
        self,
        hours_ahead : int = 6,
        message     : str = "Extreme heat advisory reminder — stay hydrated.",
    ) -> datetime:
        """
        Schedule a follow-up reminder message.

        Returns the datetime when the reminder should fire; in production
        this would register a job with a task scheduler (Celery, APScheduler).

        Parameters
        ----------
        hours_ahead : How many hours from now to schedule the reminder.
        message     : Reminder message text.

        Returns
        -------
        datetime : Scheduled fire time.
        """
        fire_at = datetime.now() + timedelta(hours=hours_ahead)
        _logger.info(
            f"Reminder scheduled at {fire_at.isoformat()}: '{message}'"
        )
        print(
            f"[AlertModule] ⏰  Reminder scheduled for "
            f"{fire_at.strftime('%Y-%m-%d %H:%M')} — '{message}'"
        )
        return fire_at

    scheduleReminder = schedule_reminder   # diagram alias

    # ------------------------------------------------------------------
    # Batch processor
    # ------------------------------------------------------------------

    def process_predictions(
        self,
        probabilities   : np.ndarray,
        heat_indices    : Optional[np.ndarray] = None,
        timestamps      : Optional[List[datetime]] = None,
        notify_agencies : bool = True,
    ) -> List[HeatEvent]:
        """
        Iterate over an array of model probabilities, generate HeatEvents
        for threshold-crossing observations, dispatch alerts, and log them.

        Parameters
        ----------
        probabilities   : (n,) array of predicted extreme-heat probabilities.
        heat_indices    : Optional (n,) array of heat index values.
        timestamps      : Optional list of n datetimes.
        notify_agencies : Notify health agency on DANGER/EXTREME events.

        Returns
        -------
        List of HeatEvent objects that triggered alerts.
        """
        events = []
        for i, prob in enumerate(probabilities):
            hi  = float(heat_indices[i]) if heat_indices is not None else None
            ts  = timestamps[i]          if timestamps   is not None else None
            evt = self.check_threshold(float(prob), hi, ts)

            if evt is None:
                continue

            events.append(evt)
            alert_msg = evt.generate_alert()
            self.send_public_alert(alert_msg)
            self.log_alert(evt.event_id)

            if notify_agencies and evt.severity in (
                Severity.DANGER, Severity.EXTREME
            ):
                self.notify_health_agency()
                self.schedule_reminder(hours_ahead=3)

        print(
            f"[AlertModule] Processed {len(probabilities)} predictions — "
            f"{len(events)} alert(s) triggered."
        )
        return events


# ===========================================================================
# DataFetchModule
# ===========================================================================

class DataFetchModule:
    """
    Live data retrieval module for weather, UHI, and air-quality feeds.

    In the thesis prototype all methods fall back to returning synthetic
    placeholder data when API keys or network access are unavailable.
    Replace the stub implementations with real API calls in production.

    Supported data sources (see data_generator.py for details)
    ----------------------------------------------------------
    • PAGASA Open Data / OpenWeatherMap  → fetchWeatherAPI()
    • MODIS LST satellite               → fetchUHIData()
    • OpenAQ / AirVisual API            → fetchPM25Data()

    Parameters
    ----------
    weather_api_key : API key for the weather data provider.
    aq_api_key      : API key for the air-quality data provider.
    location        : City name or lat/lon string.
    """

    def __init__(
        self,
        weather_api_key : str = "",
        aq_api_key      : str = "",
        location        : str = "Davao City, PH",
    ) -> None:
        self.weather_api_key = weather_api_key
        self.aq_api_key      = aq_api_key
        self.location        = location
        self._raw_cache      : Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1 · Weather
    # ------------------------------------------------------------------

    def fetch_weather_api(self) -> pd.DataFrame:
        """
        Fetch current/recent weather observations.

        Real implementation: call OpenWeatherMap or PAGASA REST API,
        e.g. ``GET https://api.openweathermap.org/data/2.5/weather
                    ?q=Davao City,PH&appid=<KEY>&units=metric``

        Returns
        -------
        pd.DataFrame with columns
            [timestamp, temperature, humidity, wind_speed, location].
        """
        _logger.info(f"Fetching weather data for '{self.location}' …")

        # ── STUB: return synthetic single observation ─────────────────
        now = datetime.now().replace(microsecond=0)
        data = {
            "timestamp"   : [now],
            "temperature" : [round(31.0 + np.random.normal(0, 1.5), 2)],
            "humidity"    : [round(78.0 + np.random.normal(0, 5.0), 2)],
            "wind_speed"  : [round(2.5  + np.random.exponential(0.5), 2)],
            "location"    : [self.location],
        }
        df = pd.DataFrame(data)
        self._raw_cache["weather"] = df
        _logger.info("Weather data fetched (stub mode).")
        return df

    fetchWeatherAPI = fetch_weather_api   # diagram alias

    # ------------------------------------------------------------------
    # 2 · Urban Heat Island
    # ------------------------------------------------------------------

    def fetch_uhi_data(self) -> pd.DataFrame:
        """
        Fetch Urban Heat Island intensity data.

        Real implementation: query MODIS Land Surface Temperature product
        via Google Earth Engine or LP DAAC:
        ``ee.ImageCollection('MODIS/006/MOD11A1')``

        Returns
        -------
        pd.DataFrame with columns [timestamp, uhi_intensity, location].
        """
        _logger.info("Fetching UHI data (MODIS LST) …")

        now = datetime.now().replace(microsecond=0)
        data = {
            "timestamp"    : [now],
            "uhi_intensity": [round(1.8 + np.random.normal(0, 0.3), 3)],
            "location"     : [self.location],
        }
        df = pd.DataFrame(data)
        self._raw_cache["uhi"] = df
        _logger.info("UHI data fetched (stub mode).")
        return df

    fetchUHIData = fetch_uhi_data   # diagram alias

    # ------------------------------------------------------------------
    # 3 · PM 2.5
    # ------------------------------------------------------------------

    def fetch_pm25_data(self) -> pd.DataFrame:
        """
        Fetch PM 2.5 air-quality observations.

        Real implementation: call OpenAQ API:
        ``GET https://api.openaq.org/v2/latest
               ?city=Davao%20City&parameter=pm25``

        Returns
        -------
        pd.DataFrame with columns [timestamp, pm25_level, location].
        """
        _logger.info("Fetching PM 2.5 data (OpenAQ) …")

        now = datetime.now().replace(microsecond=0)
        data = {
            "timestamp"  : [now],
            "pm25_level" : [round(18.0 + np.random.exponential(5.0), 2)],
            "location"   : [self.location],
        }
        df = pd.DataFrame(data)
        self._raw_cache["pm25"] = df
        _logger.info("PM 2.5 data fetched (stub mode).")
        return df

    fetchPM25Data = fetch_pm25_data   # diagram alias

    # ------------------------------------------------------------------
    # 4 · JSON parser
    # ------------------------------------------------------------------

    def parse_json(self, response: str | dict) -> dict:
        """
        Parse a raw JSON API response string or dict into a flat
        Python dictionary.

        Parameters
        ----------
        response : JSON string or already-parsed dict.

        Returns
        -------
        dict : Parsed response payload.
        """
        if isinstance(response, str):
            import json as _json
            parsed = _json.loads(response)
        else:
            parsed = dict(response)
        _logger.debug(f"Parsed JSON response with {len(parsed)} top-level keys.")
        return parsed

    parseJSON = parse_json   # diagram alias

    # ------------------------------------------------------------------
    # 5 · Raw data storage
    # ------------------------------------------------------------------

    def store_raw_data(
        self,
        data       : pd.DataFrame,
        label      : str,
        output_dir : str = "data/raw",
    ) -> str:
        """
        Persist a fetched DataFrame to disk in CSV format.

        Parameters
        ----------
        data       : DataFrame to save.
        label      : Short name used in the filename (e.g. 'weather').
        output_dir : Directory for raw data files.

        Returns
        -------
        str : Full path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path   = os.path.join(output_dir, f"{label}_{ts_str}.csv")
        data.to_csv(path, index=False)
        _logger.info(f"Raw data stored → '{path}'.")
        return path

    storeRawData = store_raw_data   # diagram alias

    # ------------------------------------------------------------------
    # Convenience: fetch everything and merge
    # ------------------------------------------------------------------

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch weather, UHI, and PM 2.5 data and merge on timestamp.

        Returns a single-row (live) DataFrame ready for pipeline ingestion.
        """
        w  = self.fetch_weather_api()
        u  = self.fetch_uhi_data()
        p  = self.fetch_pm25_data()

        merged = (
            w.merge(u, on=["timestamp", "location"], how="inner")
             .merge(p, on=["timestamp", "location"], how="inner")
        )
        _logger.info(
            f"Fetched and merged live data: {len(merged)} row(s)."
        )
        return merged