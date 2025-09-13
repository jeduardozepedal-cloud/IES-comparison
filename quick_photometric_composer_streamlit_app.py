# Quick Photometric Composer — Streamlit MVP
# -------------------------------------------------------------
# Features
# - Drag‑and‑drop multiple IES files
# - Parse LM‑63 (TILT=NONE) photometry
# - Beam/Field angle (50% / 10% of peak) auto‑calc
# - Polar candela plots for key C‑planes
# - Iso‑lux heatmap on a horizontal target plane (E = I*cos(theta)/d^2)
# - Throw calculator + beam/field diameters
# - Side‑by‑side presets (A/B) to compare different fixtures/settings
#
# Install & Run
#   pip install streamlit numpy pillow matplotlib
#   streamlit run app.py
# -------------------------------------------------------------

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- Data Types -----------------------------
@dataclass
class IESData:
    name: str
    v_angles: np.ndarray  # degrees, shape (Nv,)
    h_angles: np.ndarray  # degrees, shape (Nh,)
    candela: np.ndarray   # cd, shape (Nh, Nv). Order: for each H, all V
    multiplier: float
    photometric_type: int  # 1=C, 2=B, 3=A (we assume C)
    units_type: int        # 1=feet, 2=meters
    dimensions: Tuple[float, float, float]  # W, L, H in given units
    lumens_per_lamp: float
    num_lamps: int
    input_watts: Optional[float]


# ----------------------------- IES Parser -----------------------------

def _parse_float_stream(lines: List[str], start_idx: int) -> Tuple[List[float], int]:
    """Collect floats from lines starting at start_idx until caller stops reading."""
    vals: List[float] = []
    i = start_idx
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "":
            i += 1
            continue
        # Split on whitespace
        parts = ln.replace(",", " ").split()
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                pass
        i += 1
        if len(vals) >= 500000:  # guard
            break
    return vals, i


def parse_ies_bytes(b: bytes, name: str) -> IESData:
    text = b.decode("latin-1", errors="ignore")
    lines = text.replace("\r", "\n").split("\n")

    # Find TILT line
    tilt_idx = None
    for idx, ln in enumerate(lines):
        if ln.upper().startswith("TILT="):
            tilt_idx = idx
            tilt_val = ln.split("=", 1)[1].strip().upper()
            if tilt_val != "NONE":
                # For MVP, only support NONE; ignore tilt data if present
                pass
            break
    if tilt_idx is None:
        raise ValueError("Invalid IES: missing TILT line")

    # After TILT line, expect header numeric block
    floats, _ = _parse_float_stream(lines, tilt_idx + 1)
    if len(floats) < 10:
        raise ValueError("Invalid IES: insufficient header numbers")

    # Header (LM‑63 variants). We'll read at least 10 numbers reliably.
    num_lamps = int(round(floats[0]))
    lumens_per_lamp = floats[1]
    multiplier = floats[2]
    n_v = int(round(floats[3]))
    n_h = int(round(floats[4]))
    photometric_type = int(round(floats[5]))
    units_type = int(round(floats[6]))
    dim_w, dim_l, dim_h = floats[7], floats[8], floats[9]

    # Some files include ballast factor, future use, input watts; read safely
    ballast_factor = floats[10] if len(floats) > 10 else 1.0
    future_use = floats[11] if len(floats) > 11 else 1.0
    input_watts = floats[12] if len(floats) > 12 else None

    # Then v angles (n_v), h angles (n_h), then n_h*n_v candela
    # We already have the stream of floats from header onward; consume accordingly.
    # Starting position after first 10/13 values depending what's present
    # We will recompute floats to ensure indices correct by re-parsing from tilt+1 and then slicing.
    vals, _ = _parse_float_stream(lines, tilt_idx + 1)
    # Compute offset where vertical angles start
    # The first 10 values are header dims; the next 0..3 may be ballast/future/watts depending on presence.
    # Heuristic: remaining count minus expected angles+candela should match 0..3 extras.
    base_header_len_options = [10, 11, 12, 13]
    header_len = None
    for hl in base_header_len_options:
        rem = len(vals) - hl
        if rem >= n_v + n_h + n_v * n_h:
            header_len = hl
            break
    if header_len is None:
        # fallback to 13
        header_len = 13
    offset = header_len

    v_angles = np.array(vals[offset: offset + n_v], dtype=float)
    offset += n_v
    h_angles = np.array(vals[offset: offset + n_h], dtype=float)
    offset += n_h
    cds = np.array(vals[offset: offset + n_v * n_h], dtype=float)
    if cds.size < n_v * n_h:
        raise ValueError("Invalid IES: not enough candela data")
    candela = cds.reshape((n_h, n_v)) * float(multiplier)

    return IESData(
        name=name,
        v_angles=v_angles,
        h_angles=h_angles,
        candela=candela,
        multiplier=float(multiplier),
        photometric_type=int(photometric_type),
        units_type=int(units_type),
        dimensions=(float(dim_w), float(dim_l), float(dim_h)),
        lumens_per_lamp=float(lumens_per_lamp),
        num_lamps=int(num_lamps),
        input_watts=None if input_watts is None else float(input_watts),
    )


# ----------------------------- Photometric Utilities -----------------------------

def find_peak(ies: IESData) -> Tuple[int, int]:
    idx = np.unravel_index(np.argmax(ies.candela), ies.candela.shape)
    return int(idx[0]), int(idx[1])  # (h_idx, v_idx)


def angle_at_threshold(v_angles: np.ndarray, cd: np.ndarray, center_idx: int, frac: float) -> Tuple[Optional[float], Optional[float]]:
    """Given vertical angles and candela along one C‑plane, find left/right angles where cd drops to frac*peak around the peak.
    Returns (left_angle, right_angle). None if not found on one side."""
    peak = cd[center_idx]
    thr = peak * frac
    # Search left
    left = None
    for i in range(center_idx, 0, -1):
        if cd[i] >= thr and cd[i - 1] < thr:
            # Linear interp between i and i-1
            t = (thr - cd[i - 1]) / (cd[i] - cd[i - 1] + 1e-9)
            left = v_angles[i - 1] + t * (v_angles[i] - v_angles[i - 1])
            break
    # Search right
    right = None
    for i in range(center_idx, len(v_angles) - 1):
        if cd[i] >= thr and cd[i + 1] < thr:
            t = (thr - cd[i + 1]) / (cd[i] - cd[i + 1] + 1e-9)
            right = v_angles[i + 1] + t * (v_angles[i] - v_angles[i + 1])
            break
    return left, right


def beam_field_angles(ies: IESData) -> Tuple[Optional[float], Optional[float]]:
    """Compute beam (50% of peak) and field (10% of peak) from the C‑plane with the global peak."""
    h_idx, v_idx = find_peak(ies)
    v = ies.v_angles
    cd = ies.candela[h_idx, :]
    left50, right50 = angle_at_threshold(v, cd, v_idx, 0.5)
    left10, right10 = angle_at_threshold(v, cd, v_idx, 0.1)

    beam = None
    field = None
    if left50 is not None and right50 is not None:
        beam = abs(right50 - left50)
    if left10 is not None and right10 is not None:
        field = abs(right10 - left10)
    return beam, field


def candela_interp(ies: IESData, theta: float, cdeg: float) -> float:
    """Bilinear interpolation in (vertical angle theta, horizontal C angle). Angles in degrees.
    Assumes Type C (theta 0=nadir). Wrap C to 0..360.
    """
    v = ies.v_angles
    h = ies.h_angles
    cd = ies.candela

    # Clamp theta within provided range
    theta = float(np.clip(theta, v.min(), v.max()))

    # Wrap C within 0..360 and handle periodicity if data spans 0..360
    c = cdeg % 360.0

    # Find vertical indices
    vi = np.searchsorted(v, theta)
    v0 = max(0, min(len(v) - 2, vi - 1))
    v1 = v0 + 1
    tv = 0.0 if v[v1] == v[v0] else (theta - v[v0]) / (v[v1] - v[v0])

    # Handle horizontal wrap; assume increasing h angles (0..360)
    if h[0] == 0 and h[-1] >= 360:
        # Ensure a virtual wrap by extending one step
        h_ext = np.concatenate([h, h[:1] + 360])
        cd_ext = np.vstack([cd, cd[:1, :]])
    else:
        h_ext = h
        cd_ext = cd

    hi = np.searchsorted(h_ext, c)
    h0 = max(0, min(len(h_ext) - 2, hi - 1))
    h1 = h0 + 1
    th = 0.0 if h_ext[h1] == h_ext[h0] else (c - h_ext[h0]) / (h_ext[h1] - h_ext[h0])

    c00 = cd_ext[h0, v0]
    c01 = cd_ext[h0, v1]
    c10 = cd_ext[h1, v0]
    c11 = cd_ext[h1, v1]

    # Bilinear interpolation
    c0 = c00 * (1 - tv) + c01 * tv
    c1 = c10 * (1 - tv) + c11 * tv
    return float(c0 * (1 - th) + c1 * th)


# ----------------------------- Illuminance Map -----------------------------

def iso_lux_plane(
    ies: IESData,
    mount_height: float,
    grid_w: float,
    grid_d: float,
    res: int = 200,
    units: str = "m",  # "m" or "ft"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute E(x,y) on a horizontal plane at z = -mount_height (luminaire at origin pointing down).
    Returns X, Y mesh and E (lux). If units=ft, converts to meters internally then converts E to lux.
    """
    # Unit conversions
    if units == "ft":
        ft_to_m = 0.3048
        grid_w *= ft_to_m
        grid_d *= ft_to_m
        mount_height *= ft_to_m

    # Grid
    x = np.linspace(-grid_w / 2, grid_w / 2, res)
    y = np.linspace(-grid_d / 2, grid_d / 2, res)
    X, Y = np.meshgrid(x, y)

    H = float(mount_height)
    r = np.sqrt(X**2 + Y**2)
    d = np.sqrt(H**2 + r**2) + 1e-9

    theta = np.degrees(np.arctan2(r, H))  # 0 at nadir
    C = (np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0

    # Vectorized interpolation (loop for clarity/simplicity)
    E = np.zeros_like(X, dtype=float)
    for i in range(res):
        for j in range(res):
            I = candela_interp(ies, float(theta[i, j]), float(C[i, j]))
            E[i, j] = I * (H / d[i, j]) / (d[i, j] ** 2)  # cos(theta)=H/d

    # E currently in lux if I is in candela (SI). If ies units are feet, I is still cd; no change.
    # Return E (lux)
    return X, Y, E


# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Quick Photometric Composer", page_icon="💡", layout="wide")
st.title("Quick Photometric Composer")
st.caption("Drop IES files, compare beams, and preview iso‑lux on a plane. (MVP — visual aid, not a full lighting calc.)")

uploaded = st.file_uploader("Upload one or more IES files", type=["ies"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload IES files to get started.")
    st.stop()

# Parse all
ies_list: List[IESData] = []
errors: List[str] = []
for f in uploaded:
    try:
        ies = parse_ies_bytes(f.read(), name=f.name)
        ies_list.append(ies)
    except Exception as e:
        errors.append(f"{f.name}: {e}")

if errors:
    st.warning("\n".join(errors))

if not ies_list:
    st.error("No valid IES files parsed.")
    st.stop()

# Sidebar controls
st.sidebar.header("Target Plane & View")
units = st.sidebar.selectbox("Units", ["m", "ft"], index=0, help="Input units for mounting height and grid size.")
height = st.sidebar.number_input("Mounting height", min_value=0.1, value=3.0, step=0.1)
width = st.sidebar.number_input("Grid width", min_value=0.5, value=6.0, step=0.1)
depth = st.sidebar.number_input("Grid depth", min_value=0.5, value=6.0, step=0.1)
res = st.sidebar.slider("Resolution (px)", min_value=80, max_value=240, value=160, step=20)

st.sidebar.header("Polar Plot C‑planes")
# Choose up to 4 C planes to plot
all_c = np.unique(np.concatenate([ies.h_angles for ies in ies_list]))
def_s = [0, 45, 90, 135]
sel_c = st.sidebar.multiselect(
    "C‑planes (deg)", options=[int(x) for x in all_c], default=[c for c in def_s if c in all_c]
)
if not sel_c:
    sel_c = [int(all_c[0])]

st.sidebar.header("Throw Calculator")
ill_units = st.sidebar.selectbox("Illuminance units", ["lux", "fc"], index=0)
E_target = st.sidebar.number_input("Target center E", min_value=0.1, value=200.0, step=10.0)

# ----------------------------- Beam/Field & Throw Table -----------------------------
cols = st.columns((1.2, 1, 1))
with cols[0]:
    st.subheader("Beam / Field angles")
    rows = []
    for ies in ies_list:
        beam, field = beam_field_angles(ies)
        rows.append((ies.name, f"{beam:.1f}°" if beam else "—", f"{field:.1f}°" if field else "—"))
    st.dataframe({"IES": [r[0] for r in rows], "Beam (50%)": [r[1] for r in rows], "Field (10%)": [r[2] for r in rows]}, use_container_width=True)

with cols[1]:
    st.subheader("Center Intensity")
    ci_rows = []
    for ies in ies_list:
        # Find vertical angle closest to 0 at the C plane where it's defined
        # Use v=0 at any C plane; take max across H at v=0 if present
        if 0 in ies.v_angles:
            v0_idx = int(np.argmin(np.abs(ies.v_angles - 0)))
            I0 = float(np.max(ies.candela[:, v0_idx]))
        else:
            # approximate by nearest angle
            v0_idx = int(np.argmin(np.abs(ies.v_angles - 0)))
            I0 = float(np.max(ies.candela[:, v0_idx]))
        ci_rows.append((ies.name, I0))
    st.dataframe({"IES": [r[0] for r in ci_rows], "I(0°) cd": [f"{r[1]:,.0f}" for r in ci_rows]}, use_container_width=True)

with cols[2]:
    st.subheader("Throw for Target E")
    throw_rows = []
    # Convert E target to lux
    E_lux = E_target if ill_units == "lux" else E_target * 10.76391
    for name, I0 in [(r[0], float(r[1])) for r in ci_rows]:
        if I0 <= 0:
            d_m = float("nan")
        else:
            d_m = math.sqrt(I0 / E_lux)
        if units == "ft":
            d_disp = d_m / 0.3048
            throw_rows.append((name, f"{d_disp:.2f} ft"))
        else:
            throw_rows.append((name, f"{d_m:.2f} m"))
    st.dataframe({"IES": [r[0] for r in throw_rows], f"Distance for {E_target} {ill_units}": [r[1] for r in throw_rows]}, use_container_width=True)

# ----------------------------- Polar Plots -----------------------------
st.subheader("Polar candela (selected C‑planes)")
pp_cols = st.columns(len(sel_c))
for idx, csel in enumerate(sel_c):
    ax = None
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, projection='polar')
    for ies in ies_list:
        # Find closest available H angle to csel
        hi = int(np.argmin(np.abs(ies.h_angles - csel)))
        theta = np.radians(ies.v_angles)
        r = ies.candela[hi, :]
        # Normalize radial scale for display: scale to each fixture's max across shown plane
        ax.plot(theta, r, label=ies.name)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title(f"C = {csel}°")
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    pp_cols[idx].pyplot(fig)
    plt.close(fig)

# ----------------------------- Iso‑lux Heatmaps (A/B) -----------------------------
st.subheader("Iso‑lux plane (side‑by‑side presets)")
left, right = st.columns(2)

with left:
    st.markdown("**Preset A**")
    ies_a_name = st.selectbox("Fixture A", [ies.name for ies in ies_list], key="a_name")
    ies_a = next(ies for ies in ies_list if ies.name == ies_a_name)
    Xa, Ya, Ea = iso_lux_plane(ies_a, mount_height=height, grid_w=width, grid_d=depth, res=res, units=units)
    fig1, ax1 = plt.subplots(figsize=(5.2, 4.6))
    im1 = ax1.imshow(Ea, extent=[Xa.min(), Xa.max(), Ya.min(), Ya.max()], origin='lower')
    ax1.set_xlabel(f"x ({units})")
    ax1.set_ylabel(f"y ({units})")
    ax1.set_title(f"{ies_a.name}")
    fig1.colorbar(im1, ax=ax1, label=f"E (lux)")
    st.pyplot(fig1)
    plt.close(fig1)

with right:
    st.markdown("**Preset B**")
    ies_b_name = st.selectbox("Fixture B", [ies.name for ies in ies_list], index=min(1, len(ies_list)-1), key="b_name")
    ies_b = next(ies for ies in ies_list if ies.name == ies_b_name)
    Xb, Yb, Eb = iso_lux_plane(ies_b, mount_height=height, grid_w=width, grid_d=depth, res=res, units=units)
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.6))
    im2 = ax2.imshow(Eb, extent=[Xb.min(), Xb.max(), Yb.min(), Yb.max()], origin='lower')
    ax2.set_xlabel(f"x ({units})")
    ax2.set_ylabel(f"y ({units})")
    ax2.set_title(f"{ies_b.name}")
    fig2.colorbar(im2, ax=ax2, label=f"E (lux)")
    st.pyplot(fig2)
    plt.close(fig2)

# ----------------------------- Beam/Field diameter helper -----------------------------
st.subheader("Beam/Field diameter at distance")
cold1, cold2, cold3 = st.columns(3)
with cold1:
    dist = st.number_input(f"Distance ({units})", min_value=0.1, value=units == 'ft' and 10.0 or 3.0, step=0.1)
with cold2:
    pick_name = st.selectbox("Fixture", [ies.name for ies in ies_list], key="diam_name")
    pick = next(ies for ies in ies_list if ies.name == pick_name)
with cold3:
    b_angle, f_angle = beam_field_angles(pick)

if b_angle is None and f_angle is None:
    st.info("Beam/field angles unavailable for this file (couldn't find crossings).")
else:
    # Convert dist to meters for calc of diameter, but formula is unit‑agnostic as long as angle in radians
    diam_rows = []
    for label, ang in [("Beam", b_angle), ("Field", f_angle)]:
        if ang is None:
            diam_rows.append((label, "—"))
        else:
            diam = 2.0 * dist * math.tan(math.radians(ang / 2.0))
            diam_rows.append((label, f"{diam:.2f} {units}"))
    st.table({"Angle type": [r[0] for r in diam_rows], "Diameter": [r[1] for r in diam_rows]})

# ----------------------------- Notes -----------------------------
st.caption(
    "Notes: Assumes Type C photometry, TILT=NONE, and a horizontal target plane below the luminaire. "
    "Heatmap uses E=I(θ,C)*cos(θ)/d² with bilinear interpolation in the IES table. Results are approximate."
)
