
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, json, tempfile, math
import numpy as np


import warnings
warnings.filterwarnings("ignore")

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import librosa
import scipy

# Add top-level project to import fsic
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Try import FSIC engine (fallback stub if missing)
try:
    from fsic.engine import run_simulation
    FSIC_AVAILABLE = True
except Exception as e:
    print("❌ Could not import fsic.engine:", e)
    FSIC_AVAILABLE = False
    def run_simulation(cfg, return_trace=False):
        return {'fused': 0, 'fuse_time': 0.0, 'final_R': 0.0}


import scipy.signal
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = np.hanning  # monkey patch to avoid attribute error

# Load preset config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../presets/gold_phi_tanh_1p8.json')
try:
    with open(CONFIG_PATH) as f:
        BASE_CONFIG = json.load(f)
except Exception as e:
    print("❌ Could not load preset config:", e)
    BASE_CONFIG = {}

# app = Flask(__name__)
# CORS(app)


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["https://fsic.vercel.app"]}})



# ---------- helpers ----------
def safe_float(x, default=0.0):
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return float(default)
        return val
    except Exception:
        return float(default)


def run_real_fsic_once(mode='phi', seed=42, steps=600):
    cfg = dict(BASE_CONFIG)
    cfg.update({'mode': mode, 'seed': int(seed), 'steps': int(steps)})
    out = run_simulation(cfg, return_trace=False)
    return {
        'fused': bool(out.get('fused', 0)),
        'fuse_time': safe_float(out.get('fuse_time', 0.0)),
        'final_R': safe_float(out.get('final_R', 0.0))
    }


def analyze_music(file_path,
                  n_fft=2048,
                  hop_length=512,
                  min_segment_dur=0.4,
                  dominance_thresh=0.40,
                  max_segments=120):
    """
    Full analysis:
      - load audio
      - beat-track (fallback to fixed windows)
      - build initial segments (beat -> beat)
      - compute band energies (low/mid/high) per segment
      - label dominant band (or 'mixed')
      - merge/smooth short segments
      - select segments (limit max_segments)
      - run FSIC (limited steps) for each selected segment
      - return segments (with FSIC results) and downsampled waveform
    """

    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    frames_per_sec = sr / float(hop_length)

    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # beat tracking (frames)
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beats = np.asarray(beats)
    except Exception:
        beats = np.array([])

    # fallback: 0.5 - 1.0s grid if beats missing
    if beats is None or len(beats) < 2:
        step_frames = max(1, int(frames_per_sec * 0.5))  # 0.5s windows fallback
        beats = np.arange(0, S.shape[1], step_frames)

    # Build initial segments (frame indices)
    segments = []
    for i in range(len(beats) - 1):
        a = int(beats[i]); b = int(beats[i+1])
        if b <= a: continue
        start_t = float(librosa.frames_to_time(a, sr=sr, hop_length=hop_length, n_fft=n_fft))
        end_t = float(librosa.frames_to_time(b, sr=sr, hop_length=hop_length, n_fft=n_fft))
        seg = {'a': a, 'b': b, 'start_t': start_t, 'end_t': end_t}
        segments.append(seg)

    # If only one beat boundary or none, create windows across entire track
    if not segments:
        step_frames = max(1, int(frames_per_sec))  # 1s windows
        for a in range(0, S.shape[1], step_frames):
            b = min(S.shape[1], a + step_frames)
            start_t = float(librosa.frames_to_time(a, sr=sr, hop_length=hop_length, n_fft=n_fft))
            end_t = float(librosa.frames_to_time(b, sr=sr, hop_length=hop_length, n_fft=n_fft))
            segments.append({'a': a, 'b': b, 'start_t': start_t, 'end_t': end_t})

    # Determine band indices
    low_idx = np.where(freqs < 250)[0]
    mid_idx = np.where((freqs >= 250) & (freqs < 2000))[0]
    high_idx = np.where(freqs >= 2000)[0]
    eps = 1e-12

    # Compute energies and features per segment
    seg_info = []
    for s in segments:
        a, b = s['a'], s['b']
        if b <= a: continue
        seg_mag = S[:, a:b]
        total_power = float(seg_mag.sum()) + eps
        low_power = float(seg_mag[low_idx, :].sum()) if low_idx.size > 0 else 0.0
        mid_power = float(seg_mag[mid_idx, :].sum()) if mid_idx.size > 0 else 0.0
        high_power = float(seg_mag[high_idx, :].sum()) if high_idx.size > 0 else 0.0

        # band fractions
        low_frac = low_power / total_power
        mid_frac = mid_power / total_power
        high_frac = high_power / total_power

        # dominant band decision
        fractions = [('low', low_frac), ('mid', mid_frac), ('high', high_frac)]
        band_name, band_frac = max(fractions, key=lambda x: x[1])
        if band_frac < dominance_thresh:
            band_label = 'mixed'
        else:
            band_label = band_name

        # spectral centroid (safe)
        try:
            centroid = float(librosa.feature.spectral_centroid(S=seg_mag, sr=sr).mean())
        except Exception:
            centroid = 0.0

        seg_info.append({
            'start_time': round(float(s['start_t']), 3),
            'end_time': round(float(s['end_t']), 3),
            'duration': round(float(s['end_t'] - s['start_t']), 3),
            'low_power': low_power,
            'mid_power': mid_power,
            'high_power': high_power,
            'total_power': total_power,
            'low_frac': low_frac,
            'mid_frac': mid_frac,
            'high_frac': high_frac,
            'band': band_label,
            'band_frac': band_frac,
            'centroid': centroid
        })

    # Merge/smooth adjacent segments: combine consecutive segments with same band or very short durations
    merged = []
    for seg in seg_info:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        # merge if same band OR previous duration < min_segment_dur
        if seg['band'] == prev['band'] or prev['duration'] < min_segment_dur:
            # extend prev
            prev['end_time'] = seg['end_time']
            prev['duration'] = round(prev['end_time'] - prev['start_time'], 3)
            # accumulate energies & recompute fractions conservatively
            prev['low_power'] += seg['low_power']
            prev['mid_power'] += seg['mid_power']
            prev['high_power'] += seg['high_power']
            prev['total_power'] += seg['total_power']
            # recompute band fractions and centroid (weighted)
            tp = prev['total_power'] + eps
            prev['low_frac'] = prev['low_power'] / tp
            prev['mid_frac'] = prev['mid_power'] / tp
            prev['high_frac'] = prev['high_power'] / tp
            # recompute band label
            fractions = [('low', prev['low_frac']), ('mid', prev['mid_frac']), ('high', prev['high_frac'])]
            bname, bfrac = max(fractions, key=lambda x: x[1])
            prev['band'] = bname if bfrac >= dominance_thresh else 'mixed'
            prev['band_frac'] = bfrac
            # centroid: average (not exact but fine)
            prev['centroid'] = (prev.get('centroid', 0.0) + seg.get('centroid', 0.0)) / 2.0
        else:
            merged.append(seg.copy())

    # If still too many segments, select top segments by energy (preserve chronological order)
    if len(merged) > max_segments:
        # choose top-K by total_power
        selected = sorted(merged, key=lambda s: s['total_power'], reverse=True)[:max_segments]
        selected = sorted(selected, key=lambda s: s['start_time'])
    else:
        selected = merged

    # Map band->mode
    band_mode_map = {'low': 'phi', 'mid': 'phi', 'high': 'sin', 'mixed': 'phi'}

    # Run FSIC for each selected segment (limited steps)
    segments_out = []
    for i, seg in enumerate(selected):
        mode = band_mode_map.get(seg['band'], 'phi')
        # map features to FSIC cfg (bounded)
        cfg = dict(BASE_CONFIG)
        cfg['mode'] = mode
        # seed: deterministic-ish, based on start_time and band
        cfg['seed'] = int((seg['start_time'] * 1000) % 999999) + (0 if seg['band'] == 'low' else (1 if seg['band'] == 'mid' else 2))
        # sigma_omega: more centroid -> slightly higher
        cfg['sigma_omega'] = float(max(0.01, min(0.25, 0.02 + seg['centroid'] / (sr * 10.0 + 1e-9))))
        # J_tan scaled by normalized band energy
        band_energy_norm = float(seg['band_frac'])  # 0..1
        cfg['J_tan'] = float(max(0.6, min(1.6, 0.9 + band_energy_norm * 0.6)))
        cfg['steps'] = 400  # shorter runs for responsiveness; increase if you want more accuracy

        # Run FSIC (protect with try)
        try:
            out = run_simulation(cfg, return_trace=False)
        except Exception as e:
            out = {'fused': 0, 'fuse_time': float('nan'), 'final_R': float('nan')}
            print("❌ run_simulation failed for segment", seg, "error:", e)

        segments_out.append({
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'duration': seg['duration'],
            'band': seg['band'],
            'band_frac': round(seg['band_frac'], 4),
            'centroid': round(seg['centroid'], 2),
            'cfg': {'seed': cfg['seed'], 'sigma_omega': cfg['sigma_omega'], 'J_tan': cfg['J_tan'], 'steps': cfg['steps'], 'mode': cfg['mode']},
            'fused': bool(out.get('fused', 0)),
            'fuse_time': safe_float(out.get('fuse_time', 0.0)),
            'final_R': safe_float(out.get('final_R', 0.0)),
            # small visualization hint (color)
            'color': '#ff6b6b' if seg['band'] == 'low' else ('#feca57' if seg['band'] == 'mid' else ('#1dd1a1' if seg['band'] == 'high' else '#6c5ce7'))
        })

    # Downsample waveform for frontend (max 3000 points)
    max_points = 3000
    wf = y if len(y) <= max_points else y[::int(np.ceil(len(y) / max_points))]
    waveform = [float(x) for x in wf]

    return segments_out, waveform, duration, sr, len(y)


# ---------- endpoints ----------
@app.route('/')
def index():
    return jsonify({
        "status": "FSIC Engine backend is running",
        "fsic_available": FSIC_AVAILABLE,
        "endpoints": ["/test-fsic", "/compare", "/upload-music"]
    })


@app.route('/test-fsic', methods=['GET'])
def test_fsic():
    try:
        res = run_real_fsic_once(mode='phi', seed=11, steps=400)
        return jsonify({
            "status": "✅ FSIC Engine Working" if FSIC_AVAILABLE else "⚠️ FSIC stub",
            "test_passed": bool(res['fused']),
            "fuse_time": res['fuse_time'],
            "final_R": res['final_R']
        })
    except Exception as e:
        return jsonify({"status": "❌ FSIC test failed", "error": str(e)}), 500


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    try:
        sin = run_real_fsic_once(mode='sin', seed=42, steps=600)
        phi = run_real_fsic_once(mode='phi', seed=42, steps=600)
        time_adv = ((sin['fuse_time'] - phi['fuse_time']) / sin['fuse_time'] * 100.0
                    if sin['fuse_time'] > 0 else 0.0)
        stability_adv = ((phi['final_R'] - sin['final_R']) / sin['final_R'] * 100.0
                         if sin['final_R'] > 0 else 0.0)

        return jsonify({
            "glyphs": [
                {"name": "Traditional System", "mode": "sin",
                 "fuse_time": sin['fuse_time'], "final_R": sin['final_R'],
                 "fused": sin['fused'], "color": "#ff6b6b"},
                {"name": "Flow System", "mode": "phi",
                 "fuse_time": phi['fuse_time'], "final_R": phi['final_R'],
                 "fused": phi['fused'], "color": "#4ecdc4"}
            ],
            "performance": {
                "time_advantage": round(float(time_adv), 3),
                "stability_advantage": round(float(stability_adv), 3),
                "message": "Real FSIC simulation results"
            }
        })
    except Exception as e:
        return jsonify({"status": "❌ FSIC compare failed", "error": str(e)}), 500


@app.route('/upload-music', methods=['POST'])
def upload_music():
    if 'file' not in request.files:
        return jsonify({'error': 'No file field; use multipart form with field "file"'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False,
                                          suffix=os.path.splitext(audio_file.filename)[1] or '.mp3')
        audio_file.save(tmp.name)
        tmp.close()
        fname = tmp.name
        print(f"🔊 Received file: {audio_file.filename} -> {fname}")

        segments, waveform, duration, sr, samples = analyze_music(fname)

        return jsonify({
            "success": True,
            "analysis_type": "real_audio",
            "audio_info": {
                "filename": audio_file.filename,
                "duration": round(duration, 3),
                "sample_rate": int(sr),
                "samples": int(samples)
            },
            "segments": segments,
            "waveform": waveform,
            "performance": {"message": "Beat-segment FSIC analysis complete", "segments_analyzed": len(segments)}
        })

    except Exception as e:
        print("❌ Error in /upload-music:", e)
        return jsonify({
            "success": False,
            "analysis_type": "fallback_demo",
            "error": str(e),
            "segments": [],
            "waveform": [],
            "performance": {"message": "Fallback triggered"}
        }), 500

    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


if __name__ == '__main__':
    print("🚀 Starting FSIC backend on https://fsic.onrender.com")
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
