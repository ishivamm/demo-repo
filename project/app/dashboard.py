"""
Streamlit Dashboard – Data Center Cooling PINN.

Launch with:
    streamlit run project/app/dashboard.py
"""

import sys
import os
import numpy as np

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project.data.generate import (
    DEFAULT_CONFIG,
    generate_steady_state_temperature,
    prepare_training_data,
)
from project.optimization.cooling_optimizer import (
    optimize_vent_placement,
    score_temperature_field,
    suggest_improvements,
)


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Data Center Cooling – PINN Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ PINNs for Data Center Cooling Optimisation")
st.markdown(
    "Interactively place server racks and cooling vents, then visualise the "
    "predicted temperature distribution and optimisation suggestions."
)

# -------------------------------------------------------------------
# Sidebar – user configuration
# -------------------------------------------------------------------

st.sidebar.header("Data Center Configuration")

room_w = st.sidebar.slider("Room width (m)", 5.0, 20.0, 10.0, 0.5)
room_d = st.sidebar.slider("Room depth (m)", 5.0, 20.0, 10.0, 0.5)
ambient = st.sidebar.slider("Ambient temperature (°C)", 15.0, 30.0, 22.0, 0.5)

st.sidebar.subheader("Server Racks")
n_racks = st.sidebar.number_input("Number of racks", 1, 10, 4)

racks = []
for i in range(int(n_racks)):
    with st.sidebar.expander(f"Rack {i + 1}", expanded=(i < 2)):
        rx = st.slider(f"Rack {i+1} x", 0.0, room_w, min(3.0 + 4.0 * (i % 2), room_w), 0.1, key=f"rx{i}")
        ry = st.slider(f"Rack {i+1} y", 0.0, room_d, min(3.0 + 4.0 * (i // 2), room_d), 0.1, key=f"ry{i}")
        rq = st.slider(f"Rack {i+1} heat (W)", 10.0, 200.0, 80.0 + 5.0 * i, 5.0, key=f"rq{i}")
        racks.append({"x": rx, "y": ry, "heat_output": rq})

st.sidebar.subheader("Cooling Vents")
n_vents = st.sidebar.number_input("Number of vents", 1, 8, 4)

vents = []
default_positions = [
    (0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0),
    (0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0),
]
for i in range(int(n_vents)):
    dx, dy = default_positions[i] if i < len(default_positions) else (5.0, 5.0)
    with st.sidebar.expander(f"Vent {i + 1}", expanded=(i < 2)):
        vx = st.slider(f"Vent {i+1} x", 0.0, room_w, min(dx, room_w), 0.1, key=f"vx{i}")
        vy = st.slider(f"Vent {i+1} y", 0.0, room_d, min(dy, room_d), 0.1, key=f"vy{i}")
        vt = st.slider(f"Vent {i+1} temp (°C)", 10.0, 22.0, 16.0, 0.5, key=f"vt{i}")
        vents.append({"x": vx, "y": vy, "temp": vt})

# -------------------------------------------------------------------
# Build config & generate data
# -------------------------------------------------------------------

config = {
    "room_width": room_w,
    "room_depth": room_d,
    "nx": 64,
    "ny": 64,
    "ambient_temp": ambient,
    "thermal_diffusivity": 0.01,
    "server_racks": racks,
    "cooling_vents": vents,
}

data = generate_steady_state_temperature(config)

# -------------------------------------------------------------------
# Main visualisation
# -------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature Heat Map")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    c = ax1.pcolormesh(data["X"], data["Y"], data["T"], cmap="hot", shading="auto")
    fig1.colorbar(c, ax=ax1, label="°C")
    for r in racks:
        ax1.plot(r["x"], r["y"], "ws", markersize=10)
    for v in vents:
        ax1.plot(v["x"], v["y"], "c^", markersize=10)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Current Layout")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Hotspot Detection")
    threshold = np.percentile(data["T"], 90)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.pcolormesh(data["X"], data["Y"], data["T"], cmap="hot", shading="auto")
    ax2.contour(
        data["X"], data["Y"],
        (data["T"] > threshold).astype(float),
        levels=[0.5], colors="cyan", linewidths=2,
    )
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title(f"Hotspots (>{threshold:.1f} °C)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

metrics = score_temperature_field(data["T"])
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Max Temp", f"{metrics['max_temp']:.1f} °C")
mcol2.metric("Mean Temp", f"{metrics['mean_temp']:.1f} °C")
mcol3.metric("Std Dev", f"{metrics['std_temp']:.2f} °C")
mcol4.metric("Hotspot %", f"{metrics['hotspot_frac'] * 100:.1f}%")

# -------------------------------------------------------------------
# Optimisation
# -------------------------------------------------------------------

st.markdown("---")
st.subheader("🔧 Cooling Optimisation")

if st.button("Run Optimisation (random search)"):
    with st.spinner("Searching for better vent placements…"):
        opt = optimize_vent_placement(
            base_config=config,
            n_vents=int(n_vents),
            n_candidates=40,
            seed=np.random.randint(0, 10000),
        )

    suggestions = suggest_improvements(opt)
    for s in suggestions:
        st.success(s)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    vmin = min(data["T"].min(), opt["best_T"].min())
    vmax = max(data["T"].max(), opt["best_T"].max())
    ax3a.pcolormesh(data["X"], data["Y"], data["T"], cmap="hot", shading="auto", vmin=vmin, vmax=vmax)
    ax3a.set_title("Before")
    ax3b.pcolormesh(data["X"], data["Y"], opt["best_T"], cmap="hot", shading="auto", vmin=vmin, vmax=vmax)
    for v in opt["best_vents"]:
        ax3b.plot(v["x"], v["y"], "c^", markersize=12)
    ax3b.set_title("After")
    for ax in (ax3a, ax3b):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
