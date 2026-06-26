import streamlit as st
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import gdown

# --- ADD THIS BLOCK AT THE TOP OF YOUR APP ---
@st.cache_resource
def get_database():
    # Your specific Google Drive File ID
    file_id = '1aE1echWBLCHrVap4gOTh5dcyrQWg81Bi'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'song_database.pkl'
    
    # Download from Google Drive if it's not already in the cloud folder
    if not os.path.exists(output):
        with st.spinner("Downloading 500MB song database... (this only happens once!)"):
            gdown.download(url, output, quiet=False)
            
    # Load and return the database
    with open(output, 'rb') as f:
        db = pickle.load(f)
    return db

# Replace your old loading code with this:
song_database = get_database()
# ==========================================
# 1. CORE SCRATCH SPECTROGRAM ENGINE
# ==========================================

def get_spectrogram_scratch(audio_data, fs, window_len=4096, overlap=None):
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if overlap is None:
        overlap = window_len // 2

    step = window_len - overlap
    window = np.hanning(window_len)

    num_frames = 1 + (len(audio_data) - window_len) // step
    num_freqs = window_len // 2 + 1
    Sxx = np.zeros((num_freqs, num_frames))

    for i in range(num_frames):
        start = i * step
        end = start + window_len
        frame = audio_data[start:end] * window
        fft_frame = np.fft.rfft(frame)
        Sxx[:, i] = np.abs(fft_frame) ** 2

    f = np.fft.rfftfreq(window_len, d=1/fs)
    t = np.arange(num_frames) * step / fs
    Sxx_log = np.log10(Sxx + 1e-10)
    return f, t, Sxx_log

def extract_peaks(Sxx_log, min_amplitude=-5, neighborhood_size=20):
    local_max = (maximum_filter(Sxx_log, size=neighborhood_size) == Sxx_log)
    background = (Sxx_log > min_amplitude)
    eroded_features = local_max & background
    freq_idxs, time_idxs = np.where(eroded_features)
    return list(zip(time_idxs, freq_idxs))

# ==========================================
# 2. RUNTIME HASHING STRATEGIES
# ==========================================

def generate_combinatorial_hashes(peaks, fan_value=15, min_delta_t=0, max_delta_t=100):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                dt = t2 - t1
                if min_delta_t <= dt <= max_delta_t:
                    key = (f1, f2, dt)
                    if key not in hashes: hashes[key] = []
                    hashes[key].append(t1)
    return hashes

def generate_single_peak_hashes(peaks):
    hashes = {}
    for t, f in peaks:
        if f not in hashes: hashes[f] = []
        hashes[f].append(t)
    return hashes

def generate_pitch_invariant_hashes(peaks, fan_value=15):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                if f2 == 0: continue
                dt = t2 - t1
                key = (round(f1 / f2, 2), dt)
                if key not in hashes: hashes[key] = []
                hashes[key].append((t1, f1))
    return hashes

def generate_time_stretch_hashes(peaks, fan_value=15):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                key = (f1, f2)
                if key not in hashes: hashes[key] = []
                hashes[key].append(t1)
    return hashes

# ==========================================
# 3. MATCHING ENGINES
# ==========================================

DB_FILE = "song_database.pkl"

def match_query(query_hashes, database, mode="pairs"):
    song_scores = {}
    all_offsets = {}
    for song_name, song_data in database.items():
        offsets = []
        db_hashes = song_data.get(mode, {})
        for h_key, q_times in query_hashes.items():
            if h_key in db_hashes:
                for q_t in q_times:
                    for db_t in db_hashes[h_key]:
                        offsets.append(db_t - q_t)
        if offsets:
            all_offsets[song_name] = offsets
            hist, _ = np.histogram(offsets, bins=np.arange(min(offsets)-0.5, max(offsets)+1.5, 1))
            song_scores[song_name] = np.max(hist)
        else:
            song_scores[song_name] = 0
            all_offsets[song_name] = []

    best_match = max(song_scores, key=song_scores.get) if song_scores else "Unknown"
    return best_match, song_scores, all_offsets

def match_pitch_invariant(query_hashes, database):
    song_scores = {}
    for song_name, song_data in database.items():
        db_hashes = song_data.get("pitch_hashes", {})
        score = 0
        for key, q_data in query_hashes.items():
            if key in db_hashes:
                # Count the total number of matching invariant hash pairs
                score += len(q_data) * len(db_hashes[key])
        song_scores[song_name] = score

    best_match = max(song_scores, key=song_scores.get) if song_scores else "Unknown"
    if song_scores and song_scores.get(best_match, 0) == 0:
        best_match = "Unknown"
    return best_match, song_scores

def match_time_stretch_hough(query_hashes, database):
    song_scores, hough_data, time_pairs = {}, {}, {}
    for song_name, song_data in database.items():
        db_hashes = song_data.get("time_hashes", {})
        matched_pts = []

        for key, q_times in query_hashes.items():
            if key in db_hashes:
                for q_t in q_times:
                    for db_t in db_hashes[key]:
                        matched_pts.append((q_t, db_t))

        time_pairs[song_name] = matched_pts
        hough_pts = []

        matched_pts = sorted(list(set(matched_pts)), key=lambda x: x[0])
        for i in range(len(matched_pts)):
            for j in range(i + 1, min(i + 15, len(matched_pts))):
                t_q1, t_db1 = matched_pts[i]
                t_q2, t_db2 = matched_pts[j]
                if t_q2 - t_q1 > 0:
                    m = (t_db2 - t_db1) / (t_q2 - t_q1)
                    if 0.4 <= m <= 2.5:
                        c = t_db1 - m * t_q1
                        hough_pts.append((m, c))

        hough_data[song_name] = hough_pts
        if hough_pts:
            binned = {}
            for m, c in hough_pts:
                bin_key = (round(m, 2), round(c, -1))
                binned[bin_key] = binned.get(bin_key, 0) + 1
            song_scores[song_name] = max(binned.values())
        else:
            song_scores[song_name] = 0

    best_match = max(song_scores, key=song_scores.get) if song_scores else "Unknown"
    return best_match, hough_data, time_pairs

# ==========================================
# 4. STREAMLIT INTERFACE UI
# ==========================================

st.set_page_config(page_title="Sonic Signatures", layout="wide")
st.title("🎵 Sonic Signatures Identifier")

if 'db' not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as file: st.session_state.db = pickle.load(file)
    else:
        st.session_state.db = None

with st.sidebar:
    st.header("⚙️ Configuration")
    if st.session_state.db is not None:
        st.success(f"✅ Local Database loaded! ({len(st.session_state.db)} Songs Indexed)")
    else:
        st.error("❌ song_database.pkl not found! Run build_database.py first.")
    win_len = st.select_slider("Window Length (N)", options=[1024, 2048, 4096, 8192], value=4096)

if st.session_state.db is None:
    st.warning("Please generate your database locally using build_database.py to use this application.")
else:
    app_mode = st.radio("Select Mode:", ["Single-Clip Diagnostics", "Automated Batch Processing", "Report Experiments Lab"], horizontal=True)
    st.markdown("---")

    if app_mode == "Single-Clip Diagnostics":
        st.subheader("Step-by-Step Song Identification")
        uploaded_file = st.file_uploader("Upload query audio (.mp3, .wav)", type=["mp3", "wav"])

        if uploaded_file:
            with st.spinner("Analyzing..."):
                audio, fs = librosa.load(uploaded_file, sr=22050, mono=True)
                f, t, Sxx = get_spectrogram_scratch(audio, fs, window_len=win_len)
                peaks = extract_peaks(Sxx)
                q_hashes = generate_combinatorial_hashes(peaks)
                pred, scores, offsets = match_query(q_hashes, st.session_state.db, mode="pairs")

            st.success(f"### 🏆 MATCH FOUND: **{pred}**")
            st.caption(f"Cluster Score: {scores.get(pred, 0)} aligned hashes")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**STEP 1: The Spectrogram**")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                st.pyplot(fig)
            with col2:
                st.markdown("**STEP 2: Constellation of Peaks**")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter([t[p[0]] for p in peaks], [f[p[1]] for p in peaks], color='cyan', s=5, alpha=0.7)
                ax.set_facecolor('black')
                ax.set_ylabel("Frequency Bin")
                ax.set_xlabel("Time Frame")
                st.pyplot(fig)

            st.markdown("**STEP 3: The Proof (Alignment Spike)**")
            if pred in offsets and offsets[pred]:
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.hist(offsets[pred], bins=100, color='orange', edgecolor='black')
                ax.set_xlabel("Time Offset (Database Frame - Query Frame)")
                ax.set_ylabel("# of Hashes")
                st.pyplot(fig)

    elif app_mode == "Automated Batch Processing":
        st.subheader("Batch Recognition Engine Placeholder")
        st.info("Ready for automated evaluation metrics.")

    elif app_mode == "Report Experiments Lab":
        st.subheader("Data Collection for Mathematical PDF Evaluation")
        exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
            "Single vs Pairs", "Noise Injection", "Pitch Shift Lab", "Time Stretch Lab"
        ])

        with exp_tab1:
            st.write("**Goal:** Compare matching paired peaks vs isolated single peaks.")
            uf1 = st.file_uploader("Upload clip", type=["mp3", "wav"], key="e1")
            if uf1:
                audio, fs = librosa.load(uf1, sr=22050, mono=True)
                _, _, Sxx = get_spectrogram_scratch(audio, fs, window_len=win_len)
                peaks = extract_peaks(Sxx)

                hashes_pairs = generate_combinatorial_hashes(peaks)
                pred_p, _, off_p = match_query(hashes_pairs, st.session_state.db, mode="pairs")

                hashes_singles = generate_single_peak_hashes(peaks)
                pred_s, _, off_s = match_query(hashes_singles, st.session_state.db, mode="singles")

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Pairs Pred:** {pred_p}")
                    fig, ax = plt.subplots(figsize=(5,3))
                    ax.hist(off_p.get(pred_p, []), bins=50, color='cyan')
                    st.pyplot(fig)
                with c2:
                    st.write(f"**Singles Pred:** {pred_s}")
                    fig, ax = plt.subplots(figsize=(5,3))
                    ax.hist(off_s.get(pred_s, []), bins=50, color='orange')
                    st.pyplot(fig)

        with exp_tab2:
            st.write("**Goal:** Push recognition limits with added white noise.")
            noise_lvl = st.slider("Noise Ratio", 0.0, 2.0, 0.1)
            uf2 = st.file_uploader("Upload clip", type=["mp3", "wav"], key="e2")
            if uf2:
                audio, fs = librosa.load(uf2, sr=22050, mono=True)
                noise = np.random.normal(0, np.sqrt(np.mean(audio**2)) * noise_lvl, len(audio))
                _, _, Sxx = get_spectrogram_scratch(audio + noise, fs, window_len=win_len)
                pred, scores, _ = match_query(generate_combinatorial_hashes(extract_peaks(Sxx)), st.session_state.db, mode="pairs")
                st.info(f"Prediction: **{pred}** | Surviving Hash Matches: {scores.get(pred, 0)}")

        with exp_tab3:
            st.write("### 🎵 Pitch Shift Lab")
            st.write("Uses uniform Key: `(f1 / f2, delta_t)` to find matches invariant to constant scaling factor edits.")
            c_multiplier = st.slider("Pitch Scale Factor (c)", 0.5, 2.0, 1.0, 0.1)
            uf3 = st.file_uploader("Upload Audio (Pitch Shift)", type=["mp3", "wav"], key="pitch")

            if uf3:
                with st.spinner("Processing Pitch Invariant Hashes..."):
                    audio, fs = librosa.load(uf3, sr=22050, mono=True, duration=15)
                    if c_multiplier != 1.0:
                        audio = librosa.effects.pitch_shift(audio, sr=fs, n_steps=12 * np.log2(c_multiplier))

                    _, _, Sxx = get_spectrogram_scratch(audio, fs, window_len=win_len)
                    peaks = extract_peaks(Sxx, neighborhood_size=10)
                    q_hashes = generate_pitch_invariant_hashes(peaks)
                    
                    # Direct lookup without frequency domain hough tracking
                    pred, scores = match_pitch_invariant(q_hashes, st.session_state.db)

                st.success(f"🏆 Prediction: **{pred}** | Matching Invariant Hashes (f1/f2, Δt): **{scores.get(pred, 0)}**")
                
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
                ax.set_title(f"Spectrogram (Pitch Shifted c={c_multiplier})")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_xlabel("Time (s)")
                st.pyplot(fig)

        with exp_tab4:
            st.write("### ⏱️ Time Stretch & Hough Coordinate Space Voting")
            st.write("Identifies the tracking linear configuration: $t_{db} = (1/\\beta) t_q + c$")
            beta_multiplier = st.slider("Time Stretch Factor (β)", 0.5, 2.0, 1.0, 0.1)
            uf4 = st.file_uploader("Upload Audio (Time Stretch)", type=["mp3", "wav"], key="time")

            if uf4:
                with st.spinner("Processing Linear Coordinate Warping..."):
                    audio, fs = librosa.load(uf4, sr=22050, mono=True, duration=15)
                    if beta_multiplier != 1.0:
                        audio = librosa.effects.time_stretch(audio, rate=beta_multiplier)

                    _, _, Sxx = get_spectrogram_scratch(audio, fs, window_len=win_len)
                    peaks = extract_peaks(Sxx, neighborhood_size=10)
                    q_hashes = generate_time_stretch_hashes(peaks)
                    pred, hough_data, time_pairs = match_time_stretch_hough(q_hashes, st.session_state.db)

                st.success(f"🏆 Prediction: **{pred}**")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**1. 2D Hough Parameter Voting Space**")
                    fig, ax = plt.subplots(figsize=(5,3))
                    if hough_data.get(pred) and len(hough_data[pred]) > 0:
                        m_vals, c_vals = zip(*hough_data[pred])
                        ax.scatter(m_vals, c_vals, alpha=0.3, color='darkgreen', s=10)
                        ax.axvline(x=1/beta_multiplier, color='red', linestyle='--', label=f'True m = 1/β ({1/beta_multiplier:.2f})')
                        ax.set_xlabel("X-Axis: Slope Parameter ($m = 1/\\beta$)")
                        ax.set_ylabel("Y-Axis: Intercept Parameter ($c = \Delta t_{start}/\\beta$)")
                        ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.markdown("**2. Linear Time Mapping Alignment Plot**")
                    fig2, ax2 = plt.subplots(figsize=(5,3))
                    if time_pairs.get(pred) and len(time_pairs[pred]) > 0:
                        qt_vals, dbt_vals = zip(*time_pairs[pred])
                        ax2.scatter(qt_vals, dbt_vals, alpha=0.3, color='coral', s=10)
                        ax2.set_xlabel("X-Axis: Query Time Frame ($t_{q}$)")
                        ax2.set_ylabel("Y-Axis: Database Time Frame ($t_{db}$)")
                        ax2.set_title("Matching inliers display a rigid straight line graph")
                    st.pyplot(fig2)