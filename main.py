import time
import numpy as np
import sounddevice as sd

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

    t_now = time.time() - start_time
    # Determine breath phase
    ph, u = breath_phase(t_now)

    # Shape curve 0..1 (slow, smooth)
    # inhale: rises 0->1, exhale: falls 1->0
    if ph == "inhale":
        shape = smoothstep(u) ** 2 # slower inhale ramp
        shape *= 0.55 # reduce inhale authority
    else:
        shape = 1.0 - smoothstep(u)

    # Map breathing shape -> cutoff + gain
    cutoff = LP_MIN + (LP_MAX - LP_MIN) * shape

    # Gain: exhale slightly more "rewarding" (tiny bias)
    # (This tends to help vagal settling.)
    # Keep inhale volume flat; reward only exhale slightly
    if ph == "exhale":
         gain = MASTER_VOL * 0.80
    else:
         gain = MASTER_VOL * (0.78 + 0.22 * shape)  # exhale gets the "reward"
         gain *= 1.07

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

def main():
    print("Breath cue running. Inhale 4s, exhale 6s. Ctrl+C to stop.")
    print("Tip: close eyes, let the sound lead your breath.")
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        blocksize=BLOCK,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    print("\nStopped.")

if __name__ == "__main__":
    main()