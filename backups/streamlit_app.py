"""
OceanEmbed Streamlit Dashboard
Interactive map for predicting subsurface ocean temperature profiles.

Usage:
  1. Start the FastAPI backend: python -m uvicorn backend.app:app --port 8000
  2. Run this dashboard: streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="OceanEmbed — INCOIS Dashboard", layout="wide")
st.title("🌊 OceanEmbed — Interactive INCOIS Dashboard")
st.caption("Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction")

# --- Sidebar: Location selection ---
st.sidebar.header("📍 Select Location")
lat = st.sidebar.slider("Latitude", -30.0, 30.0, 12.0, 0.1)
lon = st.sidebar.slider("Longitude", 30.0, 120.0, 72.0, 0.1)
date = st.sidebar.date_input("Date", value="2022-06-15")
date_str = date.strftime("%Y-%m-%d")

# Preset locations
st.sidebar.subheader("Quick Locations")
presets = {
    "Bay of Bengal (15N, 88E)": (15.0, 88.0),
    "Arabian Sea (14N, 68E)": (14.0, 68.0),
    "Equatorial IO (0N, 75E)": (0.0, 75.0),
    "Maldives (4N, 73E)": (4.0, 73.0),
    "Sri Lanka (7N, 80E)": (7.0, 80.0),
    "Lakshadweep (11N, 73E)": (11.0, 73.0),
}
for name, (p_lat, p_lon) in presets.items():
    if st.sidebar.button(name):
        lat, lon = p_lat, p_lon
        st.rerun()

# --- Map display ---
st.subheader("🗺️ Map")
map_data = [{"lat": lat, "lon": lon}]
st.map(map_data, zoom=4)

# --- Prediction ---
predict_btn = st.button("🔮 Predict Temperature Profile", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Querying model..."):
        try:
            resp = requests.post(
                f"{API_BASE}/api/predict",
                json={"latitude": lat, "longitude": lon, "date": date_str},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API at localhost:8000. Start the backend first:\n\n```\npython -m uvicorn backend.app:app --port 8000\n```")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # --- Results ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌡️ Predicted Temperature Profile")
        profile = np.array(data["predicted_profile"])
        uncertainty = np.array(data["uncertainty"])
        depths = np.array(data["depth_levels"])

        # Separate derived (0m, 5m) from model-predicted (10m+)
        derived_temps = profile[:2]
        derived_depths = depths[:2]
        model_temps = profile[2:]
        model_depths = depths[2:]
        model_uncert = uncertainty[2:]

        fig = go.Figure()

        # Model-predicted line (10m-1000m)
        fig.add_trace(go.Scatter(
            x=model_temps, y=model_depths,
            mode="lines+markers",
            name="Predicted (10-1000m)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
        ))

        # Derived values (0m, 5m) with different style
        fig.add_trace(go.Scatter(
            x=derived_temps, y=derived_depths,
            mode="lines+markers",
            name="SST-anchored (0-5m)",
            line=dict(color="#4CAF50", width=2, dash="dash"),
            marker=dict(size=8, symbol="circle"),
        ))

        # Uncertainty band (model only)
        fig.add_trace(go.Scatter(
            x=model_temps + model_uncert, y=model_depths,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=model_temps - model_uncert, y=model_depths,
            mode="lines", line=dict(width=0),
            fill="tonextx", fillcolor="rgba(31,119,180,0.2)",
            name="Uncertainty",
        ))

        # Nearest Argo comparison
        if data.get("nearest_argo"):
            argo = data["nearest_argo"]
            argo_depths = sorted(argo["temperature_profile"].keys())
            argo_temps = [argo["temperature_profile"][d] for d in argo_depths]
            fig.add_trace(go.Scatter(
                x=argo_temps, y=argo_depths,
                mode="lines+markers",
                name=f"Argo #{argo['float_id']} ({argo['distance_deg']}° away)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            ))

        fig.update_layout(
            yaxis=dict(autorange="reversed", title="Depth (m)"),
            xaxis=dict(title="Temperature (°C)"),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Note about derived values
        st.info("📝 **0m** = SST (measured), **5m** = interpolated between SST and 10m model prediction. "
                "Model predicts 13 levels (10-1000m); 0m/5m are derived for display completeness.")

    with col2:
        st.subheader("📊 Details")
        st.markdown(f"**Location:** {lat:.2f}°N, {lon:.2f}°E")
        st.markdown(f"**Date:** {date_str}")
        st.markdown(f"**Data Mode:** `{data['data_mode']}`")
        st.markdown(f"**Confidence:** {data['confidence_score']:.1%}")

        st.subheader("Input Features")
        feat = data["features_used"]
        st.json(feat)

        st.subheader("Prediction Table")
        # Add labels for derived vs predicted
        labels = ["0m (SST)", "5m (interp)"] + [f"{d}m" for d in data["depth_levels"][2:]]
        table_data = {
            "Depth": labels,
            "Temp (°C)": np.round(profile, 2),
            "±Uncertainty": np.round(uncertainty, 2),
            "Source": ["derived", "derived"] + ["model"] * (len(profile) - 2),
        }
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        if data.get("nearest_argo"):
            argo = data["nearest_argo"]
            st.subheader("Nearest Argo Float")
            st.markdown(f"Float ID: **{argo['float_id']}**")
            st.markdown(f"Distance: **{argo['distance_deg']}°** (~{argo['distance_deg']*111:.0f} km)")
            st.markdown(f"Days apart: **{argo['days_diff']}**")

# --- Metrics tab ---
st.divider()
with st.expander("📈 Model Metrics (from held-out test set)", expanded=False):
    try:
        metrics_resp = requests.get(f"{API_BASE}/api/metrics", timeout=10)
        metrics_resp.raise_for_status()
        m = metrics_resp.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall RMSE", f"{m.get('overall_rmse', 'N/A'):.4f} °C")
        c2.metric("Overall R²", f"{m.get('overall_r2', 'N/A'):.4f}")
        c3.metric("Test Samples", f"{m.get('n_test_samples', 'N/A')}")

        if "depth_rmse" in m and "depth_levels" in m:
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=[f"{d}m" for d in m["depth_levels"]],
                y=m["depth_rmse"],
                name="RMSE (°C)",
                marker_color="#1f77b4",
            ))
            fig_m.update_layout(
                title="RMSE by Depth",
                xaxis_title="Depth",
                yaxis_title="RMSE (°C)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_m, use_container_width=True)

        st.caption("Note: Metrics are for the 13 model-predicted levels (10-1000m). "
                   "0m/5m are SST-anchored/interpolated values, not independently validated.")
    except Exception:
        st.info("Metrics not available. Start the backend to view metrics.")
