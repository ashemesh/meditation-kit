# Long Term Specification

## Goal
Create a closed-loop system where:

Breathing sound → changes breathing → changes HRV → gently adjusts sound depth

## Architecture (High Level)
1. Base Layer – Breathing Sound Engine.
2. Input Layer - HRV (RR Intervals).
3. Depth Permission Layer - HRV confidence.
4. Modulates Layer - HRV only increases permissiveness, never intensity.

```
Breath cue →
    breathing synchronizes →
        heart rate oscillates →
            HRV rises →
                depth increases →
                    sound subtly deepens →
                        nervous system feels safe →
                            loop stabilizes
```

## Interface

Headless CLI — no visual feedback, pure audio experience. All interaction is through sound.

## BLE Connection

- Device address is **pre-configured** in a config file (no runtime UI or scanning)
- On BLE disconnect mid-session: **terminate the app** immediately

## Input Layer

Utilize the Heart Rate Measurement characteristic (UUID 0x2A37) of the Heart Rate Service (service UUID 0x180D) to transmit heart rate data, including RR intervals, over standard BLE. This is standardized by the Bluetooth SIG, enabling compatibility with various BLE libraries.

- Stream RR intervals via BLE
- Compute rolling RMSSD using a **growing window**: start with minimum viable (~20s), grow to full 90s
- Smooth to reduce jitter

### RR Interval Artifact Handling

Apply **both** strategies:
1. **Threshold rejection**: discard intervals outside 300–1500ms range
2. **Median-based outlier detection**: reject intervals that deviate significantly from a rolling median

### Personal Baseline (Calibration Phase)

- **1–2 minute calibration phase** at session start to establish the user's resting HRV baseline
- During calibration, play an **ambient drone** (no breathing rhythm) so baseline reflects resting HRV without entrainment from guided breathing
- Once calibration completes, transition to breathing cues and the closed-loop system

## Depth Permission Layer

### Cold Start Behavior

- Start at a **baseline depth > 0** (e.g. 0.3) so the experience isn't sterile from the start
- HRV takes over depth control once sufficient data is available after calibration

### HRV Confidence Model

**Compound model** — both conditions must be met for high confidence:

```
hrv_confidence ∈ [0.0 – 1.0]
depth = f(hrv_confidence)
```

Requirements:
1. RMSSD must be **above the personal baseline**
2. RMSSD must be **stable** (low variance over the window)

**Stability gates**: if RMSSD variance exceeds a threshold, confidence is capped regardless of RMSSD magnitude. Stability is the primary gatekeeper.

### Depth Decay

- When HRV drops, depth decays with a **~60 second time constant** (tunable in the 45–90s range, default 60s)
- No abrupt changes — always smooth transitions

## Modulates Layer

### Depth-to-Sound Mapping

**Staged/layered** approach: each sound parameter has its own activation threshold and response curve. Sound unfolds progressively as depth increases — not all parameters move together.

Parameters controlled by depth:
- Slight widening of LP filter contrast
- Very small downward drift in CARRIER_HZ
- Micro gain warmth adjustments

Rules:
- HRV only increases permission
- If HRV drops → depth decays slowly (see decay time constant above)
- No abrupt changes

Goal: Sound deepens only when the body is ready
