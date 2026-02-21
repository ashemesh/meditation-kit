import time
import threading
import asyncio
import numpy as np
import sounddevice as sd
from bleak import BleakScanner, BleakClient

# -----------------------------
# User-tunable parameters
# -----------------------------
SAMPLE_RATE = 44100
BLOCK = 1024

INHALE_S = 3.0
EXHALE_S = 6.0

MASTER_VOL = 0.12       # overall loudness (keep low!)
CARRIER_HZ = 180.0      # subtle tonal center (set 0 to disable)

# "Wind" shaping (easy + effective)
LP_MIN = 250.0          # low-pass cutoff during exhale (darker)
LP_MAX = 1400.0         # low-pass cutoff during inhale (brighter)

# HRV monitoring
HRV_WINDOW_S = 60.0          # rolling window for RMSSD calculation
HRV_MIN_INTERVALS = 5        # minimum RR intervals before showing a score
HRV_TREND_THRESHOLD = 2.0    # ms change to count as ↑ or ↓ (else →)

HRV_RELAXED_THRESHOLD = 40.0   # ms — RMSSD above this = CNS relaxed
HRV_SYNC_BLEND = 0.3            # blend rate per 9 s cycle (~27 s to reach 1.0)

LP_MIN_TIGHT = 120.0            # exhale cutoff when fully synced (darker)
LP_MAX_TIGHT = 2000.0           # inhale cutoff when fully synced (brighter)

# -----------------------------
# HRV shared state
# -----------------------------
rr_buffer: list[float] = []
hrv_lock = threading.Lock()
_hrv_stop_event: threading.Event = threading.Event()

_sync_strength: float = 0.0    # 0 = passive ambient, 1 = fully breath-locked

# -----------------------------
# Helpers
# -----------------------------
def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)

def breath_phase(t: float):
    """
    Returns (phase, u) where:
      phase is "inhale" or "exhale"
      u is 0..1 progress within that phase
    """
    cycle = INHALE_S + EXHALE_S
    t = t % cycle
    if t < INHALE_S:
        return "inhale", t / INHALE_S
    return "exhale", (t - INHALE_S) / EXHALE_S

class OnePoleLP:
    """
    Simple one-pole low-pass filter for 'wind' shaping.
    """
    def __init__(self, sr: int):
        self.sr = sr
        self.z = 0.0

    def process(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        cutoff_hz = max(20.0, min(cutoff_hz, self.sr / 2 - 100))
        # One-pole coefficient (approx)
        a = np.exp(-2.0 * np.pi * cutoff_hz / self.sr)
        y = np.empty_like(x, dtype=np.float32)
        z = self.z
        for i in range(len(x)):
            z = a * z + (1 - a) * x[i]
            y[i] = z
        self.z = float(z)
        return y

def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)

# -----------------------------
# Audio engine
# -----------------------------
rng = np.random.default_rng()
lp = OnePoleLP(SAMPLE_RATE)

phase_carrier = 0.0
two_pi = 2.0 * np.pi

start_time = time.time()

def audio_callback(outdata, frames, time_info, status):
    global phase_carrier
    s = _sync_strength  # safe: main thread writes, audio thread reads (GIL, single float)

    t_now = time.time() - start_time
    ph, u = breath_phase(t_now)

    # Expand filter range and inhale authority proportional to sync strength
    lp_min = LP_MIN + (LP_MIN_TIGHT - LP_MIN) * s
    lp_max = LP_MAX + (LP_MAX_TIGHT - LP_MAX) * s

    if ph == "inhale":
        authority = 0.55 + 0.45 * s        # 55% → 100%
        shape = smoothstep(u) ** 2 * authority
    else:
        shape = 1.0 - smoothstep(u)

    cutoff = lp_min + (lp_max - lp_min) * shape

    if ph == "exhale":
        gain = MASTER_VOL * (0.80 - 0.15 * s)          # 80% → 65%
    else:
        boost = 1.07 + 0.08 * s                         # 1.07 → 1.15
        gain = MASTER_VOL * (0.78 + 0.22 * shape) * boost

    # Generate "wind": white noise -> low-pass -> gentle saturation
    noise = rng.normal(0.0, 1.0, frames).astype(np.float32)
    wind = lp.process(noise, cutoff_hz=cutoff)

    # Optional soft tonal bed (very subtle, can set CARRIER_HZ=0 to disable)
    if CARRIER_HZ > 0:
        t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
        angles = two_pi * CARRIER_HZ * t + phase_carrier
        carrier = np.sin(angles).astype(np.float32) * 0.15  # subtle
        phase_carrier = float((angles[-1] + two_pi * CARRIER_HZ / SAMPLE_RATE) % two_pi)
        sig = wind * 0.85 + carrier
    else:
        sig = wind

    # Soft clip to keep it pleasant
    sig = np.tanh(sig * 0.9).astype(np.float32)

    # Stereo output (same on L/R)
    out = (sig * gain).reshape(-1, 1)
    outdata[:] = np.repeat(out, 2, axis=1)

def parse_rr_intervals(data: bytearray) -> list[float]:
    """Parse RR intervals from BLE Heart Rate Measurement characteristic."""
    flags = data[0]
    hr_format_uint16 = flags & 0x01
    rr_present = (flags >> 4) & 0x01
    if not rr_present:
        return []
    offset = 3 if hr_format_uint16 else 2
    rr = []
    while offset + 1 < len(data):
        raw = int.from_bytes(data[offset:offset+2], "little")
        rr.append(raw * 1000.0 / 1024.0)
        offset += 2
    return rr

def calc_rmssd(rr_ms: list[float]) -> float | None:
    """RMSSD in ms. Returns None if not enough data."""
    if len(rr_ms) < HRV_MIN_INTERVALS:
        return None
    diffs = np.diff(rr_ms)
    return float(np.sqrt(np.mean(diffs ** 2)))

def hrv_notification_handler(_, data: bytearray):
    """Called by bleak on each heart rate notification (on asyncio thread)."""
    new_rr = [rr for rr in parse_rr_intervals(data) if 273 <= rr <= 2000]
    if not new_rr:
        return
    with hrv_lock:
        rr_buffer.extend(new_rr)
        total_ms = sum(rr_buffer)
        while len(rr_buffer) > 1 and total_ms > HRV_WINDOW_S * 1000:
            total_ms -= rr_buffer.pop(0)

async def hrv_run():
    """Scan for a BLE Heart Rate device, connect, and stream notifications."""
    print("[HRV] Scanning for heart rate device...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: "0000180d-0000-1000-8000-00805f9b34fb" in (ad.service_uuids or [])
            or (d.name is not None and "heart" in d.name.lower()),
        timeout=10.0,
    )
    if device is None:
        print("[HRV] No heart rate device found. HRV monitoring disabled.")
        return
    print(f"[HRV] Found {device.name}")
    HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
    async with BleakClient(device) as client:
        print(f"[HRV] Connected to {device.name}")
        await client.start_notify(HR_CHAR_UUID, hrv_notification_handler)
        while not _hrv_stop_event.is_set():
            await asyncio.sleep(0.5)
        await client.stop_notify(HR_CHAR_UUID)

def hrv_thread_fn():
    """Run the BLE asyncio event loop in a daemon thread."""
    asyncio.run(hrv_run())

def main():
    global _hrv_stop_event
    _hrv_stop_event = threading.Event()

    t = threading.Thread(target=hrv_thread_fn, daemon=True)
    t.start()

    print("Breath cue running. Inhale 3s, exhale 6s. Ctrl+C to stop.")
    print("Tip: close eyes, let the sound lead your breath.")
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        blocksize=BLOCK,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            prev_hrv: float | None = None
            cycle = INHALE_S + EXHALE_S
            while True:
                time.sleep(cycle)
                with hrv_lock:
                    rr_copy = list(rr_buffer)
                score = calc_rmssd(rr_copy)
                if score is not None:
                    if prev_hrv is None:
                        arrow = "→"
                    elif score > prev_hrv + HRV_TREND_THRESHOLD:
                        arrow = "↑"
                    elif score < prev_hrv - HRV_TREND_THRESHOLD:
                        arrow = "↓"
                    else:
                        arrow = "→"

                    # Blend sync strength toward target
                    global _sync_strength
                    target_sync = 1.0 if score >= HRV_RELAXED_THRESHOLD else 0.0
                    _sync_strength += (target_sync - _sync_strength) * HRV_SYNC_BLEND

                    print(f"HRV (RMSSD): {score:.1f} ms {arrow}  sync: {_sync_strength:.2f}")
                    prev_hrv = score
        except KeyboardInterrupt:
            pass
    _hrv_stop_event.set()
    print("\nStopped.")

if __name__ == "__main__":
    main()