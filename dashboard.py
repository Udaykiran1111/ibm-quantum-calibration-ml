"""
dashboard.py — QubitTelemetry three-model Streamlit dashboard.

Model A = XGBoost historical baseline
Model B = Random Forest 7-day
Model C = Random Forest 30-day

The comparison view always uses the latest date for which ALL THREE
models have predictions, avoiding accidental cross-date comparisons.
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="IBM Quantum — Qubit Viability",
    page_icon="⚛️",
    layout="wide",
)

MODEL_LABELS = {
    "model_a": "Model A — Historical / XGBoost",
    "model_b": "Model B — 7-Day / Random Forest",
    "model_c": "Model C — 30-Day / Random Forest",
}

MODEL_SHORT = {
    "model_a": "Model A",
    "model_b": "Model B",
    "model_c": "Model C",
}

@st.cache_resource
def load_db():
    from database import (
        get_summary_stats,
        get_model_comparison_latest,
        get_latest_rankings,
        get_ranking_history,
        get_all_models_history,
        get_latest_common_model_date,
        get_model_data_coverage,
    )
    return {
        "summary": get_summary_stats,
        "comparison": get_model_comparison_latest,
        "latest": get_latest_rankings,
        "history": get_ranking_history,
        "all_history": get_all_models_history,
        "common_date": get_latest_common_model_date,
        "coverage": get_model_data_coverage,
    }


db = load_db()

st.title("IBM Quantum — Qubit Viability Live Analytics Engine")
st.caption(
    "Three-model prospective experiment: historical XGBoost baseline "
    "versus 7-day and 30-day Random Forest models. "
    "All models use the same 17 leakage-safe historical features."
)

try:
    stats = db["summary"]()
except Exception as exc:
    st.error(f"Database connection failed: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calibration Days", stats["days_collected"])
c2.metric("Active Qubits", stats["qubits_latest"])
c3.metric("Latest Calibration", str(stats["latest_date"]))
c4.metric("Models Live", len(stats["models_live"]))

st.divider()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Analytics Scoping")

try:
    latest_a = db["latest"](model_name="model_a")
    backends = (
        sorted(latest_a["backend"].unique().tolist())
        if not latest_a.empty
        else []
    )
except Exception:
    backends = []

selected_backend = st.sidebar.selectbox(
    "Backend",
    ["All Available Systems"] + backends,
)

top_n = st.sidebar.slider(
    "Top N",
    min_value=5,
    max_value=50,
    value=20,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model definitions")
for key in ("model_a", "model_b", "model_c"):
    st.sidebar.caption(MODEL_LABELS[key])

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Live Rankings",
        "Model Comparison",
        "Qubit History",
        "Model Coverage",
    ]
)

# ============================================================
# TAB 1 — LIVE RANKINGS
# ============================================================
with tab1:
    model_choice = st.selectbox(
        "Prediction layer",
        options=["model_a", "model_b", "model_c"],
        format_func=lambda x: MODEL_LABELS[x],
    )

    backend_arg = (
        None
        if selected_backend == "All Available Systems"
        else selected_backend
    )

    df = db["latest"](
        backend=backend_arg,
        model_name=model_choice,
        top_n=None,
    )

    if df.empty:
        st.info("No rankings available for this model yet.")
    else:
        st.subheader(
            f"{MODEL_LABELS[model_choice]} — Latest Ranking"
        )

        render_backends = (
            sorted(df["backend"].unique())
            if selected_backend == "All Available Systems"
            else [selected_backend]
        )

        for backend in render_backends:
            sub = (
                df[df["backend"] == backend]
                .sort_values("viability_rank")
                .head(top_n)
                .copy()
            )

            if sub.empty:
                continue

            display = sub[
                [
                    "viability_rank",
                    "qubit",
                    "viability_score",
                    "T1_us",
                    "T2_us",
                    "readout_error",
                    "label",
                ]
            ].copy()

            display.columns = [
                "Rank",
                "Qubit",
                "Viability Score",
                "T1 (us)",
                "T2 (us)",
                "Readout Error",
                "Observed Label",
            ]

            display["Viability Score"] = display["Viability Score"].round(4)
            display["T1 (us)"] = display["T1 (us)"].round(2)
            display["T2 (us)"] = display["T2 (us)"].round(2)
            display["Readout Error"] = display["Readout Error"].round(5)
            display["Observed Label"] = display["Observed Label"].map(
                {1: "VIABLE", 0: "NON-VIABLE"}
            )

            st.markdown(f"**{backend}**")
            st.dataframe(
                display.set_index("Rank"),
                use_container_width=True,
                height=min(600, 60 + 38 * len(display)),
            )

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================
with tab2:
    st.subheader("A/B/C Prediction Agreement")

    common_date = db["common_date"]()

    if common_date is None:
        st.info(
            "A common prediction date is not available yet. "
            "All three models must have rankings for the same date."
        )
    else:
        st.caption(
            f"Common comparison date: {common_date}"
        )

        backend_arg = (
            None
            if selected_backend == "All Available Systems"
            else selected_backend
        )

        comp = db["comparison"](backend=backend_arg)

        if comp.empty:
            st.info("No common-date comparison data.")
        else:
            view = comp[
                [
                    "backend",
                    "qubit",
                    "model_a",
                    "model_b",
                    "model_c",
                    "score_range",
                    "agreement",
                ]
            ].copy()

            view.columns = [
                "Backend",
                "Qubit",
                "Model A",
                "Model B",
                "Model C",
                "Score Range",
                "Agreement",
            ]

            for col in ["Model A", "Model B", "Model C", "Score Range"]:
                view[col] = view[col].astype(float).round(4)

            st.dataframe(
                view.sort_values(
                    ["Backend", "Qubit"]
                ),
                use_container_width=True,
                height=550,
            )

            st.markdown(
                "**Agreement:** HIGH ≤ 0.05 score range; "
                "MODERATE ≤ 0.15; LOW > 0.15."
            )

            st.subheader("Prediction Distribution")
            numeric = comp[["model_a", "model_b", "model_c"]].dropna()

            if not numeric.empty:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.boxplot(
                    [
                        numeric["model_a"],
                        numeric["model_b"],
                        numeric["model_c"],
                    ],
                    tick_labels=["Model A", "Model B", "Model C"],
                )
                ax.set_ylabel("Viability Probability")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, clear_figure=True)

# ============================================================
# TAB 3 — QUBIT HISTORY
# ============================================================
with tab3:
    st.subheader("Historical Qubit Tracking")

    if not backends:
        st.info("No backend data available.")
    else:
        b = st.selectbox(
            "Backend",
            backends,
            key="history_backend",
        )

        latest_for_b = db["latest"](
            backend=b,
            model_name="model_a",
        )

        qubits = (
            sorted(latest_for_b["qubit"].astype(int).unique())
            if not latest_for_b.empty
            else []
        )

        if not qubits:
            st.info("No qubits available.")
        else:
            q = st.selectbox(
                "Qubit",
                qubits,
                key="history_qubit",
            )

            model = st.selectbox(
                "Model layer",
                ["model_a", "model_b", "model_c"],
                format_func=lambda x: MODEL_LABELS[x],
                key="history_model",
            )

            hist = db["history"](b, int(q), model)

            if hist.empty:
                st.info("No history for this model/qubit.")
            else:
                hist = hist.sort_values("snapshot_date")

                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.plot(
                    pd.to_datetime(hist["snapshot_date"]),
                    hist["viability_score"],
                    marker="o",
                )
                ax.set_ylim(0, 1)
                ax.set_ylabel("Viability Probability")
                ax.set_xlabel("Date")
                ax.grid(alpha=0.25)
                ax.set_title(
                    f"{MODEL_SHORT[model]} — {b} Qubit {q}"
                )
                st.pyplot(fig, clear_figure=True)

                st.dataframe(
                    hist[
                        [
                            "snapshot_date",
                            "viability_score",
                            "viability_rank",
                            "label",
                            "T1_us",
                            "T2_us",
                            "readout_error",
                        ]
                    ],
                    use_container_width=True,
                )

# ============================================================
# TAB 4 — COVERAGE
# ============================================================
with tab4:
    st.subheader("Model Deployment Coverage")

    coverage = db["coverage"]()

    if coverage.empty:
        st.info("No model ranking data yet.")
    else:
        coverage["model_name"] = coverage["model_name"].map(
            MODEL_LABELS
        ).fillna(coverage["model_name"])

        st.dataframe(
            coverage,
            use_container_width=True,
        )

    st.markdown(
        """
        **Experimental rule:** Model C is not treated as an already-validated
        future predictor merely because it has a high LOBO score. Its final
        predictive performance is measured prospectively on observations
        collected after August 11, 2026.
        """
    )

st.divider()
st.caption(
    "QubitTelemetry • Three-model prospective IBM Quantum viability experiment"
)