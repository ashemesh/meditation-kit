# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python audio breathing meditation guide that generates real-time synthesized "wind" sound cues to lead breathing rhythm (inhale 3s, exhale 6s). Uses procedural audio synthesis with filtered white noise and a subtle carrier tone.

## Commands

```bash
uv sync                # Install dependencies
python main.py         # Run the meditation audio (Ctrl+C to stop)
```

Requires Python 3.13+. No test suite or linting is configured.

## Architecture

Single-file application (`main.py`) using a **streaming audio callback** pattern:

- **Tunable constants** at module top: sample rate, breathing durations, filter cutoffs, volumes
- **`breath_phase(t)`**: State machine cycling inhale/exhale, returns phase name and progress (0→1)
- **`OnePoleLP`**: Stateful one-pole low-pass filter for shaping white noise into wind
- **`audio_callback()`**: Called per 1024-sample block (~23ms) by sounddevice — generates noise, filters with breathing-dependent cutoff (bright on inhale, dark on exhale), mixes carrier sine, applies tanh soft clip and gain shaping, outputs stereo
- **`main()`**: Opens `sounddevice.OutputStream` and blocks until Ctrl+C

Key dependencies: `numpy`, `sounddevice` (PortAudio backend).

## Design Notes

- Exhale gets slightly higher amplitude than inhale to encourage longer exhalation (vagal activation)
- Filter cutoff sweeps between `LP_MIN` (250 Hz, exhale) and `LP_MAX` (1400 Hz, inhale) using smoothstep interpolation
- Carrier tone at 180 Hz is mixed at 15% — kept subtle to avoid being distracting
- All audio parameters update per block for smooth continuous modulation
