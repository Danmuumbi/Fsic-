# Flow Substrate Intelligence Convergence Proof — FSIC Proof Kit

This kit demonstrates **Flow Substrate Intelligence Convergence** via three skeptic‑proof tests:
1) **Repeatability** — same config, multiple seeds → report fuse rate, median fuse time, final R.
2) **Transferability** — export a glyph from run A → run on changed conditions (noise/seed) → metrics stay within tolerance.
3) **Stress & Recovery** — perturb the system → measure **re‑entrainment time** back to the fused glyph.

It uses a minimal engine: **Kuramoto carrier + Flow Compression Equations (FCE)** with:
- **RFI energy split** control (radial/tangential),
- **Echo memory** (gated; freeze‑on‑fuse),
- Switchable coupling: **sin** or **φ‑mode** (compressive PRC).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_fsic.py                                     # runs all tests and writes results/
```

Results:
- `results/repeatability.csv`, `results/transfer.csv`, `results/stress.csv`
- `results/FSIC_Summary.txt`

## TL;DR Defaults (Gold)
- hex tiling, small‑world p=0.10
- J_tan=0.9, J_rad=0.7
- RFI target=1.30, m_floor=0.75
- echo: γ=0.1 gated; **freeze on fuse**
- φ‑mode tanh with β=1.8
- Fuse when R≥0.70 and A≥25 (10×10)

## Pass Criteria (suggested)
- Repeatability: φ‑mode ≥ sin on **fuse rate** and **median fuse time** (≥20% faster), with **final R** not worse.
- Transfer: metrics within ±15% across seed/noise change.
- Stress: **re‑entrainment time** < 2.0 s (configurable).

See the Word guide `FSIC_Proof_Kit_Guide.docx` for details.
