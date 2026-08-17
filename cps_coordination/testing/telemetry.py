"""
cps_coordination/testing/telemetry.py
---------------------------------------
Schema + collector factory for CPS coordination evaluation telemetry
(roadmap step 8).

Reuses ``path_planning/rta/collect.py``'s ``VerboseParquetCollector`` (via
``get_collector``) for the chunked-flush-to-Parquet mechanics — it already
implements exactly the needed pattern and tags every logged batch with
``is_success`` regardless of outcome — rather than writing a new collector
class. Only the row schema/assembly below is new.

Two parallel Parquet streams per evaluation run, joined by ``episode_id``:

- **aircraft** (``AIRCRAFT_COLUMNS``) — one row per landed/failed aircraft,
  including a per-decision-step trajectory point cloud (``traj_x``/``traj_y``,
  nested ``list<float>`` columns) so tortuosity/entropy/KL can be recomputed
  offline without a second raw-trajectory format.
- **separation** (``SEPARATION_COLUMNS``) — one row per consecutive landing
  pair on a runway, so ``C_sep`` can be recomputed offline at any tolerance
  without re-deriving pairs from the aircraft table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from path_planning.rta.collect import BaseDataCollector, get_collector

AIRCRAFT_COLUMNS: List[str] = [
    "episode_id",
    "acid",
    "flight_id",
    "runway_id",
    "wake_cat",
    "k_cps",
    "runway_assignment_mode",
    "assigned_tta",
    "actual_landing_time",
    "rta_error_cps",
    "rta_error_static",
    "rta_error_solo",
    "tta_updated_mid_trajectory",
    "stall_detected",
    "success",
    "death_cause",
    "traj_x",
    "traj_y",
]

SEPARATION_COLUMNS: List[str] = [
    "episode_id",
    "runway_id",
    "acid_lead",
    "acid_trail",
    "gap_actual_s",
    "required_sep_s",
]

# Diagnostic-only stream (Vector 9, phase3_cps_coordination_plan.md): one row
# per aircraft per _assign_runways_dynamic decision cycle, sourced from
# CPSManager.drain_reassignment_log() (only non-empty when the manager was
# built with log_reassignment_events=True). Not part of the standard
# cps_eval_aircraft/cps_eval_separation pair -- opt-in, separate file, much
# higher row count (one row per aircraft per decision cycle, not per
# aircraft per episode).
REASSIGNMENT_COLUMNS: List[str] = [
    "episode_id",
    "k_cps",
    "runway_assignment_mode",
    "current_time",
    "acid",
    "current_runway",
    "fcfs_rank",
    "sigma_current",
    "eligible_runways",
    "chosen_runway",
    "switched",
    "eta_gap_s",
    "stalled_excluded",
    "sigma_per_runway",
    "eta_per_runway",
    "x",
    "y",
]

AIRCRAFT_FILENAME = "cps_eval_aircraft.parquet"
SEPARATION_FILENAME = "cps_eval_separation.parquet"
REASSIGNMENT_FILENAME = "cps_eval_reassignment.parquet"


@dataclass
class AircraftTelemetryRow:
    """One landed/failed aircraft, one row. See ``AIRCRAFT_COLUMNS``."""

    episode_id: int
    acid: str
    flight_id: str
    runway_id: str
    wake_cat: str
    k_cps: int
    runway_assignment_mode: str
    assigned_tta: float
    actual_landing_time: float
    rta_error_cps: float
    rta_error_static: float
    rta_error_solo: float
    tta_updated_mid_trajectory: bool
    stall_detected: bool
    success: bool
    death_cause: Optional[str] = None
    traj_x: List[float] = field(default_factory=list)
    traj_y: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in AIRCRAFT_COLUMNS}


@dataclass
class SeparationTelemetryRow:
    """One consecutive landing pair on a runway, one row. See ``SEPARATION_COLUMNS``."""

    episode_id: int
    runway_id: str
    acid_lead: str
    acid_trail: str
    gap_actual_s: float
    required_sep_s: float

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in SEPARATION_COLUMNS}


@dataclass
class ReassignmentTelemetryRow:
    """One aircraft's _assign_runways_dynamic decision, one cycle, one row.

    See ``REASSIGNMENT_COLUMNS``. Diagnostic-only (Vector 9) -- callers must
    explicitly opt in (``CPSManager(log_reassignment_events=True)`` +
    ``build_reassignment_collector``); nothing writes this by default.
    """

    episode_id: int
    k_cps: int
    runway_assignment_mode: str
    current_time: float
    acid: str
    current_runway: str
    fcfs_rank: int
    sigma_current: int
    eligible_runways: str
    chosen_runway: str
    switched: bool
    eta_gap_s: float
    stalled_excluded: bool
    sigma_per_runway: str = ""
    eta_per_runway: str = ""
    x: float = 0.0
    y: float = 0.0

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in REASSIGNMENT_COLUMNS}


def build_collectors(
    save_path: str,
    chunk_size: int = 25,
    fresh_start: bool = True,
) -> Tuple[BaseDataCollector, BaseDataCollector]:
    """Return ``(aircraft_collector, separation_collector)``.

    Both are ``VerboseParquetCollector`` instances (via ``get_collector``),
    writing to ``<save_path>/cps_eval_aircraft.parquet`` and
    ``<save_path>/cps_eval_separation.parquet`` respectively. The two streams
    use separate files rather than one, since they have different (and
    incompatible, for a single Arrow schema) row shapes.

    Parameters
    ----------
    save_path : str
        Directory to write the two Parquet files into (created if missing).
    chunk_size : int
        Episodes buffered per flush (passed straight through to
        ``get_collector`` — counts calls to ``finalise_episode``, one of
        which this module's callers make per episode per stream).
    fresh_start : bool
        If True (default), deletes any existing file at the destination
        paths before writing.
    """
    os.makedirs(save_path, exist_ok=True)
    aircraft_collector = get_collector(
        os.path.join(save_path, AIRCRAFT_FILENAME),
        chunk_size,
        fresh_start=fresh_start,
        is_verbose=True,
    )
    separation_collector = get_collector(
        os.path.join(save_path, SEPARATION_FILENAME),
        chunk_size,
        fresh_start=fresh_start,
        is_verbose=True,
    )
    return aircraft_collector, separation_collector


def build_reassignment_collector(
    save_path: str,
    chunk_size: int = 25,
    fresh_start: bool = True,
) -> BaseDataCollector:
    """Return a collector writing to ``<save_path>/cps_eval_reassignment.parquet``.

    Separate from ``build_collectors`` since this stream is opt-in
    (diagnostic-only, Vector 9) and callers that don't enable
    ``log_reassignment_events`` shouldn't pay for an unused file/collector.
    """
    os.makedirs(save_path, exist_ok=True)
    return get_collector(
        os.path.join(save_path, REASSIGNMENT_FILENAME),
        chunk_size,
        fresh_start=fresh_start,
        is_verbose=True,
    )
