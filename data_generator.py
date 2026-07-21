"""
Synthetic sensor data generator.

Produces a CSV with 3 columns: temp, pressure, alarm.

- ~100,000 rows, the vast majority of which are normal, quiet baseline data.
- temp drifts (rolling-avg sense) from ~10 to ~15 degrees over the series.
- pressure drifts (rolling-avg sense) from ~200000 to ~100000 pascal over the series.
- ~100 alarms (alarm == 1) total, unchanged from before -- so the ~100 short
  anomaly windows (roughly 4,000 rows) now make up only a few percent of all
  rows, with the other ~96,000 rows being ordinary baseline + noise.
- each alarm is preceded by a short window of "weird" behavior in
  temp/pressure so a model could learn to anticipate it.
- 5 distinct kinds of pre-alarm weirdness, ~20 occurrences each:
    1. temp_spike        -> sudden/sharp spike in temp only
    2. pressure_spike     -> sudden/sharp spike (up or down) in pressure only
    3. spiral_both        -> growing-amplitude oscillation in both signals
    4. accelerated_drift  -> both signals ramp away from baseline faster than normal
    5. flicker_variance   -> both signals get noisy/jittery (variance blow-up)
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
N = 100_000                     # total rows
ROLL_WINDOW = 1000               # rolling window used to describe the "drift"

TEMP_START, TEMP_END = 10.0, 15.0            # degrees, rolling-avg drift target
PRESSURE_START, PRESSURE_END = 200_000.0, 100_000.0  # pascal, rolling-avg drift target

TEMP_NOISE_STD = 0.3             # normal sensor noise (degrees)
PRESSURE_NOISE_STD = 1000.0      # normal sensor noise (pascal)

EVENT_TYPES = [
    "temp_spike",
    "pressure_spike",
    "spiral_both",
    "accelerated_drift",
    "flicker_variance",
]
EVENTS_PER_TYPE = 20              # -> 100 alarms total
START_BUFFER = 1000               # leave room at the start (rolling window warm-up)
END_BUFFER = 150                  # leave room at the end for the longest event window

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# 1. Baseline signals (smooth drift + normal sensor noise)
# ----------------------------------------------------------------------
temp_baseline = np.linspace(TEMP_START, TEMP_END, N)
pressure_baseline = np.linspace(PRESSURE_START, PRESSURE_END, N)

temp = temp_baseline + rng.normal(0, TEMP_NOISE_STD, N)
pressure = pressure_baseline + rng.normal(0, PRESSURE_NOISE_STD, N)

alarm = np.zeros(N, dtype=int)
event_type_log = np.array([""] * N, dtype=object)  # for QA only, not saved to CSV


# ----------------------------------------------------------------------
# 2. Anomaly ("weird behavior") pattern functions
#    Each mutates temp/pressure in-place over [start, start+length) and
#    the alarm fires on the *last* row of that window.
# ----------------------------------------------------------------------
def temp_spike(temp, pressure, start, length, rng):
    """Sudden, accelerating spike in temp only."""
    idx = np.arange(start, start + length)
    ramp = np.linspace(0, 1, length) ** 3          # slow build-up, sharp near the end
    magnitude = rng.uniform(8, 15) * rng.choice([-1, 1], p=[0.15, 0.85])  # mostly up-spikes
    temp[idx] += ramp * magnitude
    # a little extra jitter right before the alarm
    temp[idx[-5:]] += rng.normal(0, 0.5, min(5, length))


def pressure_spike(temp, pressure, start, length, rng):
    """Sudden, accelerating spike (up or down) in pressure only."""
    idx = np.arange(start, start + length)
    ramp = np.linspace(0, 1, length) ** 3
    magnitude = rng.uniform(30_000, 60_000)
    sign = rng.choice([-1, 1])
    pressure[idx] += sign * ramp * magnitude
    pressure[idx[-5:]] += rng.normal(0, 800, min(5, length))


def spiral_both(temp, pressure, start, length, rng):
    """Growing-amplitude, growing-frequency oscillation in both signals (instability)."""
    idx = np.arange(start, start + length)
    t = np.linspace(0, 4 * np.pi, length)
    freq = np.linspace(1, 3, length)
    amp_temp = np.linspace(0, 6, length)
    amp_pressure = np.linspace(0, 25_000, length)
    temp[idx] += amp_temp * np.sin(t * freq)
    pressure[idx] += amp_pressure * np.cos(t * freq)


def accelerated_drift(temp, pressure, start, length, rng):
    """Both signals ramp away from baseline much faster than the normal long-term drift."""
    idx = np.arange(start, start + length)
    ramp = np.linspace(0, 1, length) ** 2
    temp_dir = rng.choice([-1, 1])
    pressure_dir = rng.choice([-1, 1])
    temp[idx] += temp_dir * ramp * rng.uniform(5, 10)
    pressure[idx] += pressure_dir * ramp * rng.uniform(20_000, 40_000)


def flicker_variance(temp, pressure, start, length, rng):
    """Both signals become noisy/jittery -- variance blows up before the alarm."""
    idx = np.arange(start, start + length)
    growing_std_temp = np.linspace(0.2, 4, length)
    growing_std_pressure = np.linspace(500, 15_000, length)
    temp[idx] += rng.normal(0, growing_std_temp)
    pressure[idx] += rng.normal(0, growing_std_pressure)


PATTERN_FUNCS = {
    "temp_spike": temp_spike,
    "pressure_spike": pressure_spike,
    "spiral_both": spiral_both,
    "accelerated_drift": accelerated_drift,
    "flicker_variance": flicker_variance,
}


# ----------------------------------------------------------------------
# 3. Lay out 100 non-overlapping event slots across the timeline,
#    shuffle which event type goes in which slot, inject each one.
# ----------------------------------------------------------------------
total_events = EVENTS_PER_TYPE * len(EVENT_TYPES)
usable_range = (N - END_BUFFER) - START_BUFFER
segment = usable_range // total_events  # even spacing between event windows

event_type_list = EVENT_TYPES * EVENTS_PER_TYPE
rng.shuffle(event_type_list)

events = []  # (start, length, type) for QA/reporting
for i, etype in enumerate(event_type_list):
    length = int(rng.integers(20, 60))          # weirdness lasts 20-59 rows before the alarm
    slot_start = START_BUFFER + i * segment
    max_jitter = max(segment - length - 5, 0)
    jitter = int(rng.integers(0, max_jitter + 1))
    start = slot_start + jitter

    PATTERN_FUNCS[etype](temp, pressure, start, length, rng)

    alarm_idx = start + length - 1
    alarm[alarm_idx] = 1
    event_type_log[alarm_idx] = etype
    events.append((start, length, etype, alarm_idx))

events.sort(key=lambda e: e[0])


# ----------------------------------------------------------------------
# 4. Assemble + save
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "temp": np.round(temp, 3),
    "pressure": np.round(pressure, 1),
    "alarm": alarm,
})

out_path = "sensor_data.csv"
df.to_csv(out_path, index=False)

# ----------------------------------------------------------------------
# 5. QA / sanity report (printed, not saved)
# ----------------------------------------------------------------------
print(f"Rows: {len(df)}")
print(f"Total alarms: {df['alarm'].sum()}")
print("Alarms per event type:")
counts = {}
for _, _, etype, _ in events:
    counts[etype] = counts.get(etype, 0) + 1
for k, v in counts.items():
    print(f"  {k}: {v}")

roll_temp = df["temp"].rolling(ROLL_WINDOW).mean()
roll_pressure = df["pressure"].rolling(ROLL_WINDOW).mean()
print(f"\nRolling({ROLL_WINDOW}) temp avg: start={roll_temp.dropna().iloc[0]:.2f}, "
      f"end={roll_temp.iloc[-1]:.2f}")
print(f"Rolling({ROLL_WINDOW}) pressure avg: start={roll_pressure.dropna().iloc[0]:.1f}, "
      f"end={roll_pressure.iloc[-1]:.1f}")

# check no overlaps
for a, b in zip(events, events[1:]):
    a_end = a[0] + a[1]
    assert a_end <= b[0], f"Overlap detected: {a} -> {b}"
print("\nNo overlapping anomaly windows. OK.")