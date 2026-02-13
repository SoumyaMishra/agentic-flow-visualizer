#!/usr/bin/env python3
"""Markdown-configurable pipeline metrics simulator with global clock and events.

Supports:
- Global clock start time
- Stages with explicit start/end/duration
- Event timeline with event IDs (parallel + serial behavior)
- C-struct-like stage/event definitions in markdown
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = "pipeline_config.md"
DEFAULT_METRICS = "metrics_output.json"
DEFAULT_DASHBOARD = "pipeline_dashboard.html"

DEFAULT_SAMPLE_INTERVAL = 0.4
MIN_SAMPLE_INTERVAL = 0.05
MAX_SAMPLE_INTERVAL = 5.0
MAX_TIMELINE_DURATION_SEC = 24 * 60 * 60
DEFAULT_GPU_VRAM_TOTAL = 8192
DEFAULT_MEM_TOTAL_MB = 16384

COLOR_PALETTE = [
    "#8B5CF6",
    "#3B82F6",
    "#10B981",
    "#F59E0B",
    "#EF4444",
    "#06B6D4",
    "#6366F1",
    "#22C55E",
]

BEHAVIOR_ALIASES = {
    "off": "off",
    "none": "off",
    "false": "off",
    "0": "off",
    "on": "on",
    "true": "on",
    "1": "on",
    "steady": "on",
    "flat": "on",
    "gradual_increase": "gradual_increase",
    "increase": "gradual_increase",
    "up": "gradual_increase",
    "inc": "gradual_increase",
    "gradual_decrease": "gradual_decrease",
    "decrease": "gradual_decrease",
    "down": "gradual_decrease",
    "dec": "gradual_decrease",
}

LOAD_MODE_ALIASES = {
    "uniform": "uniform",
    "flat": "uniform",
    "equal": "uniform",
    "custom": "custom",
}

RESOURCE_TARGET_ALIASES = {
    "": "auto",
    "auto": "auto",
    "cpu": "cpu",
    "cpu_only": "cpu",
    "gpu": "gpu",
    "gpu_only": "gpu",
    "both": "cpu_gpu",
    "cpu_gpu": "cpu_gpu",
    "gpu_cpu": "cpu_gpu",
    "none": "none",
    "off": "none",
}


@dataclass
class StageConfig:
    stage_id: str
    name: str
    start: float
    end: float
    color: str
    label: str
    hw: str
    cpu: str
    gpu: str
    mem: str
    disk: str
    network: str
    cpu_level: float
    gpu_level: float
    mem_level: float
    disk_level: float
    network_level: float
    cpu_load_mode: str
    cpu_core_loads: list[float]
    gpu_load_mode: str
    gpu_device_loads: list[float]


@dataclass
class EventConfig:
    event_id: str
    name: str
    start: float
    end: float
    resource_target: str
    cpu: str
    gpu: str
    mem: str
    disk: str
    network: str
    cpu_level: float
    gpu_level: float
    mem_level: float
    disk_level: float
    network_level: float
    cpu_load_mode: str
    cpu_core_loads: list[float]
    gpu_load_mode: str
    gpu_device_loads: list[float]


def _normalize_behavior(value: str) -> str:
    key = (value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in BEHAVIOR_ALIASES:
        allowed = ", ".join(sorted(set(BEHAVIOR_ALIASES.values())))
        raise ValueError(f"Unknown behavior '{value}'. Allowed: {allowed}")
    return BEHAVIOR_ALIASES[key]


def _normalize_load_mode(value: str | None) -> str:
    key = (value or "uniform").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LOAD_MODE_ALIASES:
        allowed = ", ".join(sorted(set(LOAD_MODE_ALIASES.values())))
        raise ValueError(f"Unknown load mode '{value}'. Allowed: {allowed}")
    return LOAD_MODE_ALIASES[key]


def _normalize_resource_target(value: str | None) -> str:
    key = (value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in RESOURCE_TARGET_ALIASES:
        allowed = ", ".join(sorted(set(RESOURCE_TARGET_ALIASES.values())))
        raise ValueError(f"Unknown resource_target '{value}'. Allowed: {allowed}")
    return RESOURCE_TARGET_ALIASES[key]


def _resolve_cpu_gpu_behavior(row: dict[str, str], resource_target: str) -> tuple[str, str]:
    cpu_raw = row.get("cpu")
    gpu_raw = row.get("gpu")

    # resource_target provides defaults; explicit cpu/gpu keys override.
    if resource_target == "cpu":
        cpu_raw = cpu_raw if cpu_raw is not None else "on"
        gpu_raw = gpu_raw if gpu_raw is not None else "off"
    elif resource_target == "gpu":
        cpu_raw = cpu_raw if cpu_raw is not None else "off"
        gpu_raw = gpu_raw if gpu_raw is not None else "on"
    elif resource_target == "cpu_gpu":
        cpu_raw = cpu_raw if cpu_raw is not None else "on"
        gpu_raw = gpu_raw if gpu_raw is not None else "on"
    elif resource_target == "none":
        cpu_raw = cpu_raw if cpu_raw is not None else "off"
        gpu_raw = gpu_raw if gpu_raw is not None else "off"

    return _normalize_behavior(cpu_raw or "off"), _normalize_behavior(gpu_raw or "off")


def _parse_level(value: str | None, default: float = 1.0) -> float:
    if value is None:
        return default
    raw = value.strip()
    if raw == "":
        return default
    level = float(raw)
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"Level must be between 0.0 and 1.0, got {level}")
    return level


def _parse_int(value: str | None, default: int, min_value: int = 1) -> int:
    if value is None or value.strip() == "":
        return default
    out = int(value.strip())
    if out < min_value:
        raise ValueError(f"Integer value must be >= {min_value}, got {out}")
    return out


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return default[:]
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        return default[:]
    out = [int(x) for x in parts]
    if any(v <= 0 for v in out):
        raise ValueError("All integer list values must be > 0")
    return out


def _parse_float_list(value: str | None) -> list[float]:
    if value is None or value.strip() == "":
        return []
    parts = [x.strip() for x in value.split(",") if x.strip()]
    out = [float(x) for x in parts]
    if any(v < 0 for v in out):
        raise ValueError("Load distribution values must be >= 0")
    return out


def _uniform_distribution(count: int) -> list[float]:
    return [1.0 / count for _ in range(count)]


def _resolve_load_distribution(mode: str, values: list[float], count: int, label: str) -> list[float]:
    if mode == "uniform":
        return _uniform_distribution(count)

    if len(values) != count:
        raise ValueError(
            f"{label} uses custom mode, expected {count} values but got {len(values)}"
        )

    s = sum(values)
    if s <= 0:
        raise ValueError(f"{label} custom values must sum to > 0")

    return [v / s for v in values]


def _parse_hms_to_sec(text: str) -> int:
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", text.strip())
    if not m:
        raise ValueError("clock_start must be HH:MM or HH:MM:SS")
    h = int(m.group(1))
    mi = int(m.group(2))
    s = int(m.group(3) or "0")
    if h > 23 or mi > 59 or s > 59:
        raise ValueError("Invalid clock_start value")
    return h * 3600 + mi * 60 + s


def _sec_to_hms(total_seconds: float) -> str:
    total = int(total_seconds) % 86400
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _level(behavior: str, progress: float) -> float:
    if behavior == "off":
        return 0.0
    if behavior == "on":
        return 0.65
    if behavior == "gradual_increase":
        return 0.1 + 0.8 * progress
    if behavior == "gradual_decrease":
        return 0.9 - 0.8 * progress
    return 0.0


def _scaled_level(behavior: str, progress: float, level: float) -> float:
    return _clamp(_level(behavior, progress) * level, 0.0, 1.0)


def _extract_section(md_text: str, title: str) -> str:
    pattern = rf"^##\s+{re.escape(title)}\s*$"
    lines = md_text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(pattern, ln.strip(), flags=re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return ""

    out = []
    for ln in lines[start:]:
        if re.match(r"^##\s+", ln.strip()):
            break
        out.append(ln)
    return "\n".join(out).strip()


def _parse_global(section_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for ln in section_text.splitlines():
        m = re.match(r"^-\s*([A-Za-z0-9_\- ]+)\s*:\s*(.+?)\s*$", ln.strip())
        if not m:
            continue
        k = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        result[k] = m.group(2).strip()
    return result


def _parse_markdown_table(section_text: str) -> list[dict[str, str]]:
    table_lines = [ln.rstrip() for ln in section_text.splitlines() if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        return []

    headers = [h.strip().lower().replace(" ", "_") for h in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for ln in table_lines[2:]:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        row = {headers[i]: cols[i] for i in range(len(headers))}
        if any(v for v in row.values()):
            rows.append(row)
    return rows


def _parse_struct_items(section_text: str) -> list[dict[str, str]]:
    """Parse c-struct-like items:
    { id: router, name: Router, start: 0.0, duration: 1.2, cpu: on }
    Supports comma values, e.g. cpu_core_loads: 1,1,1,1
    """
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("\n", " "), section_text)
    items = re.findall(r"\{([^{}]+)\}", cleaned, flags=re.DOTALL)
    rows: list[dict[str, str]] = []

    # Split on commas only when they start the next key:value pair.
    kv_pattern = re.compile(
        r"([A-Za-z0-9_\- ]+)\s*:\s*(.*?)(?=,\s*[A-Za-z0-9_\- ]+\s*:|$)",
        re.DOTALL,
    )

    for raw in items:
        row: dict[str, str] = {}
        for m in kv_pattern.finditer(raw):
            key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            val = m.group(2).strip().strip('"').strip("'")
            if key:
                row[key] = val
        if row:
            rows.append(row)
    return rows


def _parse_rows(section_text: str) -> list[dict[str, str]]:
    rows = _parse_markdown_table(section_text)
    if rows:
        return rows
    return _parse_struct_items(section_text)


def _resolve_window(row: dict[str, str], cursor: float, default_duration: float = 1.0) -> tuple[float, float]:
    start_raw = row.get("start", "").strip()
    end_raw = row.get("end", "").strip()
    dur_raw = row.get("duration", row.get("duration_sec", "")).strip()

    start = float(start_raw) if start_raw else None
    end = float(end_raw) if end_raw else None
    duration = float(dur_raw) if dur_raw else None

    if start is not None and end is not None:
        pass
    elif start is not None and duration is not None:
        end = start + duration
    elif end is not None and duration is not None:
        start = end - duration
    elif start is not None:
        end = start + default_duration
    elif duration is not None:
        start = cursor
        end = start + duration
    elif end is not None:
        start = max(0.0, end - default_duration)
    else:
        raise ValueError("Each stage/event needs start/end/duration (at least one timing expression)")

    if duration is not None and duration <= 0:
        raise ValueError("duration must be > 0")
    if start is not None and start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and end < 0:
        raise ValueError("end must be >= 0")
    if not all(math.isfinite(v) for v in [start, end] if v is not None):
        raise ValueError("start/end/duration must be finite numeric values")
    if end <= start:
        raise ValueError("end must be > start")
    if end > MAX_TIMELINE_DURATION_SEC:
        raise ValueError(f"end time exceeds max supported duration ({MAX_TIMELINE_DURATION_SEC}s)")
    return float(start), float(end)


def load_config(config_path: Path) -> tuple[dict, list[StageConfig], list[EventConfig]]:
    text = config_path.read_text(encoding="utf-8")

    global_cfg = _parse_global(_extract_section(text, "Global"))
    sample_interval = float(global_cfg.get("sample_interval_sec", DEFAULT_SAMPLE_INTERVAL))
    if not math.isfinite(sample_interval):
        raise ValueError("sample_interval_sec must be a finite number")
    if sample_interval < MIN_SAMPLE_INTERVAL or sample_interval > MAX_SAMPLE_INTERVAL:
        raise ValueError(
            f"sample_interval_sec must be between {MIN_SAMPLE_INTERVAL} and {MAX_SAMPLE_INTERVAL}"
        )
    clock_start_str = global_cfg.get("clock_start", "09:00:00")
    clock_start_sec = _parse_hms_to_sec(clock_start_str)

    cpu_core_count = _parse_int(global_cfg.get("cpu_core_count"), 8)
    l1_cache_kb = _parse_int(global_cfg.get("l1_cache_kb"), 64)
    l2_cache_kb = _parse_int(global_cfg.get("l2_cache_kb"), 512)
    l3_cache_mb = _parse_int(global_cfg.get("l3_cache_mb"), 24)

    gpu_count = _parse_int(global_cfg.get("gpu_count"), 2)
    gpu_vram_totals = _parse_int_list(global_cfg.get("gpu_vram_totals_mb"), [12288, 8192])
    if len(gpu_vram_totals) < gpu_count:
        gpu_vram_totals.extend([gpu_vram_totals[-1]] * (gpu_count - len(gpu_vram_totals)))
    gpu_vram_totals = gpu_vram_totals[:gpu_count]

    stage_rows = _parse_rows(_extract_section(text, "Stages"))
    if not stage_rows:
        raise ValueError("No stages found. Define them in ## Stages as table or { ... } structs")

    stages: list[StageConfig] = []
    cursor = 0.0
    for idx, row in enumerate(stage_rows):
        start, end = _resolve_window(row, cursor)
        cursor = max(cursor, end)
        stage_id = row.get("id", f"stage_{idx+1}").strip()
        name = row.get("name", stage_id).strip()
        color = row.get("color", COLOR_PALETTE[idx % len(COLOR_PALETTE)])
        label = row.get("label", f"Stage {idx+1}: {name}").strip()
        hw = row.get("hw", "Configurable").strip()
        cpu_load_mode = _normalize_load_mode(row.get("cpu_load_mode"))
        gpu_load_mode = _normalize_load_mode(row.get("gpu_load_mode"))
        cpu_core_loads = _resolve_load_distribution(
            cpu_load_mode,
            _parse_float_list(row.get("cpu_core_loads", row.get("cpu_loads"))),
            cpu_core_count,
            f"stage '{stage_id}' cpu_core_loads",
        )
        gpu_device_loads = _resolve_load_distribution(
            gpu_load_mode,
            _parse_float_list(row.get("gpu_device_loads", row.get("gpu_loads"))),
            gpu_count,
            f"stage '{stage_id}' gpu_device_loads",
        )

        stages.append(
            StageConfig(
                stage_id=stage_id,
                name=name,
                start=start,
                end=end,
                color=color,
                label=label,
                hw=hw,
                cpu=_normalize_behavior(row.get("cpu", "off")),
                gpu=_normalize_behavior(row.get("gpu", "off")),
                mem=_normalize_behavior(row.get("mem", "off")),
                disk=_normalize_behavior(row.get("disk", "off")),
                network=_normalize_behavior(row.get("network", "off")),
                cpu_level=_parse_level(row.get("cpu_level")),
                gpu_level=_parse_level(row.get("gpu_level")),
                mem_level=_parse_level(row.get("mem_level")),
                disk_level=_parse_level(row.get("disk_level")),
                network_level=_parse_level(row.get("network_level")),
                cpu_load_mode=cpu_load_mode,
                cpu_core_loads=cpu_core_loads,
                gpu_load_mode=gpu_load_mode,
                gpu_device_loads=gpu_device_loads,
            )
        )

    event_rows = _parse_rows(_extract_section(text, "Events"))
    events: list[EventConfig] = []
    event_cursor = 0.0
    for idx, row in enumerate(event_rows):
        start, end = _resolve_window(row, event_cursor)
        event_cursor = max(event_cursor, end)
        event_id = row.get("event_id", row.get("id", f"evt_{idx+1}")).strip()
        name = row.get("name", event_id).strip()
        resource_target = _normalize_resource_target(row.get("resource_target", row.get("target")))
        cpu_behavior, gpu_behavior = _resolve_cpu_gpu_behavior(row, resource_target)
        cpu_load_mode = _normalize_load_mode(row.get("cpu_load_mode"))
        gpu_load_mode = _normalize_load_mode(row.get("gpu_load_mode"))
        cpu_core_loads = _resolve_load_distribution(
            cpu_load_mode,
            _parse_float_list(row.get("cpu_core_loads", row.get("cpu_loads"))),
            cpu_core_count,
            f"event '{event_id}' cpu_core_loads",
        )
        gpu_device_loads = _resolve_load_distribution(
            gpu_load_mode,
            _parse_float_list(row.get("gpu_device_loads", row.get("gpu_loads"))),
            gpu_count,
            f"event '{event_id}' gpu_device_loads",
        )

        events.append(
            EventConfig(
                event_id=event_id,
                name=name,
                start=start,
                end=end,
                resource_target=resource_target,
                cpu=cpu_behavior,
                gpu=gpu_behavior,
                mem=_normalize_behavior(row.get("mem", "off")),
                disk=_normalize_behavior(row.get("disk", "off")),
                network=_normalize_behavior(row.get("network", "off")),
                cpu_level=_parse_level(row.get("cpu_level")),
                gpu_level=_parse_level(row.get("gpu_level")),
                mem_level=_parse_level(row.get("mem_level")),
                disk_level=_parse_level(row.get("disk_level")),
                network_level=_parse_level(row.get("network_level")),
                cpu_load_mode=cpu_load_mode,
                cpu_core_loads=cpu_core_loads,
                gpu_load_mode=gpu_load_mode,
                gpu_device_loads=gpu_device_loads,
            )
        )

    meta = {
        "sample_interval_sec": sample_interval,
        "clock_start": _sec_to_hms(clock_start_sec),
        "clock_start_sec": clock_start_sec,
        "cpu_core_count": cpu_core_count,
        "l1_cache_kb": l1_cache_kb,
        "l2_cache_kb": l2_cache_kb,
        "l3_cache_mb": l3_cache_mb,
        "gpu_count": gpu_count,
        "gpu_vram_totals_mb": gpu_vram_totals,
    }
    return meta, stages, events


def _active_stage(stages: list[StageConfig], t: float) -> StageConfig | None:
    active = [s for s in stages if s.start <= t < s.end]
    if not active:
        return None
    active.sort(key=lambda s: (s.start, s.end))
    return active[-1]


def _merge_metric_levels(stage: StageConfig | None, events: list[EventConfig], t: float, metric: str) -> float:
    levels = [0.0]

    if stage and stage.start <= t < stage.end:
        stage_progress = (t - stage.start) / max(1e-6, (stage.end - stage.start))
        behavior = getattr(stage, metric)
        intensity = getattr(stage, f"{metric}_level")
        levels.append(_scaled_level(behavior, stage_progress, intensity))

    for ev in events:
        if ev.start <= t < ev.end:
            ev_progress = (t - ev.start) / max(1e-6, (ev.end - ev.start))
            behavior = getattr(ev, metric)
            intensity = getattr(ev, f"{metric}_level")
            levels.append(_scaled_level(behavior, ev_progress, intensity))

    return max(levels)


def _merge_load_distribution(
    stage: StageConfig | None,
    events: list[EventConfig],
    t: float,
    metric: str,
    count: int,
) -> list[float]:
    values = [0.0] * count
    total_strength = 0.0
    uniform = _uniform_distribution(count)

    def add_distribution(cfg, start: float, end: float) -> None:
        nonlocal total_strength
        progress = (t - start) / max(1e-6, (end - start))
        behavior = getattr(cfg, metric)
        intensity = getattr(cfg, f"{metric}_level")
        strength = _scaled_level(behavior, progress, intensity)
        if strength <= 0.0:
            return

        weights = cfg.cpu_core_loads if metric == "cpu" else cfg.gpu_device_loads
        for i in range(count):
            values[i] += weights[i] * strength
        total_strength += strength

    if stage and stage.start <= t < stage.end:
        add_distribution(stage, stage.start, stage.end)

    for ev in events:
        if ev.start <= t < ev.end:
            add_distribution(ev, ev.start, ev.end)

    if total_strength <= 1e-9:
        return uniform

    s = sum(values)
    if s <= 1e-9:
        return uniform

    return [v / s for v in values]


def simulate(meta: dict, stages: list[StageConfig], events: list[EventConfig]) -> dict:
    sample_interval = meta["sample_interval_sec"]

    total_duration = 0.0
    if stages:
        total_duration = max(total_duration, max(s.end for s in stages))
    if events:
        total_duration = max(total_duration, max(e.end for e in events))

    t = 0.0
    samples: list[dict] = []

    cpu = 4.0
    cpu_temp = 31.0
    mem_percent = 20.0
    gpu_util = 0.0
    gpu_vram = 0.0
    gpu_temp = 28.0
    disk_r = 0.0
    disk_w = 0.0
    net_s = 0.0
    net_r = 0.0

    alpha = 0.28

    while t <= total_duration + 1e-9:
        stage = _active_stage(stages, t)

        cpu_lvl = _merge_metric_levels(stage, events, t, "cpu")
        gpu_lvl = _merge_metric_levels(stage, events, t, "gpu")
        mem_lvl = _merge_metric_levels(stage, events, t, "mem")
        disk_lvl = _merge_metric_levels(stage, events, t, "disk")
        net_lvl = _merge_metric_levels(stage, events, t, "network")

        cpu_target = cpu_lvl * 100
        cpu_temp_target = 30 + cpu_lvl * 55
        gpu_target = gpu_lvl * 100
        mem_target = 16 + mem_lvl * 62
        vram_target = gpu_lvl * DEFAULT_GPU_VRAM_TOTAL * 0.9
        temp_target = 27 + gpu_lvl * 55

        cpu = _clamp(cpu + alpha * (cpu_target - cpu) + random.uniform(-2.5, 2.5), 0, 100)
        cpu_temp = _clamp(cpu_temp + alpha * (cpu_temp_target - cpu_temp) + random.uniform(-0.8, 0.8), 28, 96)
        gpu_util = _clamp(gpu_util + alpha * (gpu_target - gpu_util) + random.uniform(-2.0, 2.0), 0, 100)
        mem_percent = _clamp(mem_percent + alpha * (mem_target - mem_percent) + random.uniform(-1.2, 1.2), 8, 96)
        gpu_vram = _clamp(gpu_vram + alpha * (vram_target - gpu_vram) + random.uniform(-30, 30), 0, DEFAULT_GPU_VRAM_TOTAL)
        gpu_temp = _clamp(gpu_temp + alpha * (temp_target - gpu_temp) + random.uniform(-0.8, 0.8), 25, 92)

        disk_r += max(0.0, random.uniform(0.0, 0.02) + 0.07 * disk_lvl)
        disk_w += max(0.0, random.uniform(0.0, 0.03) + 0.10 * disk_lvl)
        net_s += max(0.0, random.uniform(0.0, 0.01) + 0.04 * net_lvl)
        net_r += max(0.0, random.uniform(0.0, 0.01) + 0.05 * net_lvl)

        active_event_ids = [e.event_id for e in events if e.start <= t < e.end]
        sample_clock_sec = meta["clock_start_sec"] + t

        cpu_core_count = meta["cpu_core_count"]
        l1_base = meta["l1_cache_kb"]
        l2_base = meta["l2_cache_kb"]
        l3_base = meta["l3_cache_mb"]

        cpu_dist = _merge_load_distribution(stage, events, t, "cpu", cpu_core_count)
        cpu_cores = []
        for i in range(cpu_core_count):
            core_target = _clamp(cpu * cpu_dist[i] * cpu_core_count, 0, 100)
            core_util = _clamp(core_target + random.uniform(-5, 5), 0, 100)
            core_temp = _clamp(28 + core_util * 0.62 + random.uniform(-2.2, 2.2), 25, 98)
            cpu_cores.append({"core_id": i, "util": round(core_util, 1), "temp": round(core_temp, 1)})

        cache_view = {
            "l1_kb": [{"core_id": i, "value": round(l1_base + cpu_cores[i]["util"] * 0.15, 1)} for i in range(cpu_core_count)],
            "l2_kb": [{"core_id": i, "value": round(l2_base + cpu_cores[i]["util"] * 0.45, 1)} for i in range(cpu_core_count)],
            "l3_global_mb": round(l3_base + (cpu / 100.0) * (l3_base * 0.35), 2),
        }

        gpu_count = meta["gpu_count"]
        gpu_vram_totals = meta["gpu_vram_totals_mb"]
        gpu_dist = _merge_load_distribution(stage, events, t, "gpu", gpu_count)

        gpu_devices = []
        for i in range(gpu_count):
            split = gpu_dist[i]
            dev_util_target = _clamp(gpu_util * split * gpu_count, 0, 100)
            dev_util = _clamp(dev_util_target + random.uniform(-4, 4), 0, 100)
            dev_vram_total = gpu_vram_totals[i]
            dev_vram = _clamp((gpu_vram * split) + random.uniform(-90, 90), 0, dev_vram_total)
            dev_temp = _clamp(26 + dev_util * 0.58 + random.uniform(-2.0, 2.0), 25, 95)
            gpu_devices.append(
                {
                    "gpu_id": i,
                    "name": f"GPU-{i}",
                    "util": round(dev_util, 1),
                    "vram": round(dev_vram, 0),
                    "vram_total": dev_vram_total,
                    "temp": round(dev_temp, 1),
                }
            )

        samples.append(
            {
                "t": round(t, 2),
                "clock": _sec_to_hms(sample_clock_sec),
                "step": stage.stage_id if stage else "idle",
                "active_event_ids": active_event_ids,
                "cpu": round(cpu, 1),
                "cpu_temp": round(cpu_temp, 1),
                "cpu_cores": cpu_cores,
                "cpu_cache": cache_view,
                "mem": round(mem_percent, 1),
                "mem_mb": round((mem_percent / 100.0) * DEFAULT_MEM_TOTAL_MB, 0),
                "disk_r": round(disk_r, 2),
                "disk_w": round(disk_w, 2),
                "net_s": round(net_s, 2),
                "net_r": round(net_r, 2),
                "gpu_util": round(gpu_util, 1),
                "gpu_vram": round(gpu_vram, 0),
                "gpu_vram_total": DEFAULT_GPU_VRAM_TOTAL,
                "gpu_temp": round(gpu_temp, 1),
                "gpu_devices": gpu_devices,
            }
        )

        t += sample_interval

    steps = [
        {
            "id": s.stage_id,
            "name": s.name,
            "start": round(s.start, 2),
            "end": round(s.end, 2),
        }
        for s in stages
    ]

    step_config = {
        s.stage_id: {"label": s.label, "hw": s.hw, "color": s.color}
        for s in stages
    }

    events_meta = [
        {
            "event_id": e.event_id,
            "name": e.name,
            "start": round(e.start, 2),
            "end": round(e.end, 2),
            "resource_target": e.resource_target,
            "cpu": e.cpu,
            "gpu": e.gpu,
            "mem": e.mem,
            "disk": e.disk,
            "network": e.network,
            "resources": [
                r
                for r in ["cpu", "gpu", "mem", "disk", "network"]
                if getattr(e, r) != "off"
            ],
        }
        for e in events
    ]

    return {
        "metadata": {
            "total_duration_sec": round(total_duration, 2),
            "sample_interval_sec": sample_interval,
            "clock_start": meta["clock_start"],
            "clock_start_sec": meta["clock_start_sec"],
            "hardware_config": {
                "cpu_core_count": meta["cpu_core_count"],
                "l1_cache_kb": meta["l1_cache_kb"],
                "l2_cache_kb": meta["l2_cache_kb"],
                "l3_cache_mb": meta["l3_cache_mb"],
                "gpu_count": meta["gpu_count"],
                "gpu_vram_totals_mb": meta["gpu_vram_totals_mb"],
            },
            "steps": steps,
            "step_config": step_config,
            "events": events_meta,
        },
        "samples": samples,
    }


def inject_dashboard_data(dashboard_path: Path, data: dict) -> None:
    if not dashboard_path.exists():
        return

    text = dashboard_path.read_text(encoding="utf-8")
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    text, n0 = re.subn(
        r'(<script id="embeddedData" type="application/json">).*?(</script>)',
        rf"\1{data_json}\2",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n0 == 0:
        text, n0 = re.subn(
            r"<script>\s*// ─── Embedded Metrics Data ──────────────────────────────────",
            (
                '<script id="embeddedData" type="application/json">'
                + data_json
                + "</script>\n\n<script>\n// ─── Embedded Metrics Data ──────────────────────────────────"
            ),
            text,
            count=1,
            flags=re.DOTALL,
        )

    text, _ = re.subn(
        r"const DATA = .*?;\n\n// ─── Step Configuration",
        (
            'const DATA = JSON.parse(document.getElementById("embeddedData").textContent || "{}");\n\n'
            "// ─── Step Configuration"
        ),
        text,
        flags=re.DOTALL,
    )
    text, _ = re.subn(
        r"const STEP_CONFIG = .*?;\n\nconst steps =",
        "const STEP_CONFIG = DATA.metadata.step_config || {};\n\nconst steps =",
        text,
        flags=re.DOTALL,
    )

    # Update timer formatter to support global clock display.
    text = re.sub(
        r"function updateTimer\(t\) \{[\s\S]*?\}\n\n// ─── Animation Loop",
        (
            "function updateTimer(t, sample) {\n"
            "  const mins = Math.floor(t / 60);\n"
            "  const secs = t % 60;\n"
            "  const elapsed = String(mins).padStart(2, \"0\") + \":\" + secs.toFixed(1).padStart(4, \"0\");\n"
            "  const clock = sample && sample.clock ? sample.clock : (DATA.metadata.clock_start || \"00:00:00\");\n"
            "  document.getElementById(\"timerDisplay\").textContent = `${clock} (+${elapsed})`;\n"
            "}\n\n"
            "// ─── Animation Loop"
        ),
        text,
        flags=re.DOTALL,
    )

    # Ensure animation calls pass sample to timer.
    text = text.replace("updateTimer(s.t);", "updateTimer(s.t, s);")
    text = text.replace("updateTimer(0);", "updateTimer(0, null);")

    if n0 == 0:
        raise RuntimeError("Failed to inject embedded DATA into dashboard HTML")

    dashboard_path.write_text(text, encoding="utf-8")


def _validate_output_path(path: Path, expected_suffix: str, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd not in resolved.parents and resolved != cwd:
        raise ValueError(f"Output path must be inside working directory: {cwd}")
    if expected_suffix and resolved.suffix.lower() != expected_suffix:
        raise ValueError(f"Output path must end with {expected_suffix}: {resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"Required path does not exist: {resolved}")
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"Path must be a file, not a directory: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Configurable pipeline metrics simulator")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to markdown config file")
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS, help="Output JSON path")
    parser.add_argument("--dashboard", default=DEFAULT_DASHBOARD, help="Dashboard HTML path to update")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    metrics_path = _validate_output_path(Path(args.metrics_out), ".json")
    dashboard_path = _validate_output_path(Path(args.dashboard), ".html", must_exist=True)

    meta, stages, events = load_config(config_path)
    data = simulate(meta, stages, events)

    metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    inject_dashboard_data(dashboard_path, data)

    print("=" * 60)
    print("Markdown-configurable pipeline simulation complete")
    print(f"Config: {config_path}")
    print(f"Stages: {len(stages)}")
    print(f"Events: {len(events)}")
    print(f"Samples: {len(data['samples'])}")
    print(f"Duration: {data['metadata']['total_duration_sec']}s")
    print(f"Clock start: {data['metadata']['clock_start']}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Dashboard updated: {dashboard_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
