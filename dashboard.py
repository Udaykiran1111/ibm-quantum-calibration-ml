import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# -- Page Configuration ──
st.set_page_config(
    page_title="IBM Quantum Qubit Analytics",
    layout="wide",
)

# Professional Minimal Palette
COLOR_PRIMARY = "#E9E9E9"    # Deep Navy
COLOR_SECONDARY = "#2B6CB0"  # Slate Blue
COLOR_ACCENT = "#319795"     # Teal
COLOR_MUTED = "#4A5568"      # Charcoal
COLOR_CRITICAL = "#C53030"   # Deep Red

# -- Database Module Integration ──
@st.cache_resource
def load_db_functions():
    try:
        from database import (
            get_latest_rankings, get_ranking_history,
            get_summary_stats, get_all_dates
        )
        return get_latest_rankings, get_ranking_history, get_summary_stats, get_all_dates
    except Exception as e:
        st.error(f"Database Initialization Error: {e}")
        st.stop()

get_latest_rankings, get_ranking_history, get_summary_stats, get_all_dates = load_db_functions()

# -- Header Section ──
st.title("IBM Quantum — Qubit Viability Live Analytics Engine")
st.caption(
    "Predictive performance modeling leveraging Random Forest v2 architecture (LOBO AUC: 0.904). "
    "Features derived strictly from historical coherence variables to prevent current-session target leakage."
)

# -- System KPI Summary ──
try:
    stats = get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Depth (Days)", stats['days_collected'])
    col2.metric("Active Managed Qubits", stats['qubits_latest'])
    col3.metric("Last Data Ingestion", str(stats['latest_date']))
    col4.metric("Analytics Engine", "Random Forest v2")
except Exception as e:
    st.warning(f"Metadata summary unavailable: {e}")

st.divider()

# -- Dynamic Data Hydration ──
try:
    # Pull global latest snapshot to dynamically extract discovered backends
    global_latest = get_latest_rankings()
    
    if global_latest.empty:
        st.warning(
            "Data Layer Empty. Please execute collector.py to populate historical "
            "calibration structures and initialize prediction sequences."
        )
        st.stop()
        
    # Dynamically track available operational backends from DB
    DYNAMIC_BACKENDS = sorted(global_latest['backend'].unique())

except Exception as e:
    st.error(f"Failed to pull operational data frames: {e}")
    st.stop()

# -- Enterprise Control Filters (Sidebar) ──
st.sidebar.header("Analytics Scoping")
selected_backend = st.sidebar.selectbox(
    "Target Quantum Architecture", ["All Available Systems"] + DYNAMIC_BACKENDS, index=0
)
top_n = st.sidebar.slider("Rank View Limit (Top N)", min_value=5, max_value=50, value=20)
show_only_viable = st.sidebar.checkbox("Filter: Viable Cohort Only (Label = 1)", value=False)

# Apply runtime dataframe filtering
df = global_latest.copy()
if selected_backend != "All Available Systems":
    df = df[df['backend'] == selected_backend]

if show_only_viable:
    df = df[df['label'] == 1]

# -- Interface Tab Matrix ──
tab1, tab2, tab3 = st.tabs(["Performance Rankings", "Coherence Trends Canvas", "Architecture Comparison"])

# ==============================================================================
# TAB 1 — Performance Rankings
# ==============================================================================
with tab1:
    st.subheader(f"Top {top_n} Characterized Qubits by Viability Index")
    
    backends_to_render = DYNAMIC_BACKENDS if selected_backend == "All Available Systems" else [selected_backend]
    
    for b_name in backends_to_render:
        sub_df = df[df['backend'] == b_name].nsmallest(top_n, 'viability_rank').copy()
        if sub_df.empty:
            continue
            
        st.markdown(f"**System Matrix: {b_name}**")
        
        # Clean production column presentation
        display_df = sub_df[['viability_rank', 'qubit', 'viability_score',
                             'T1_us', 'T2_us', 'readout_error', 'label']].copy()
        display_df.columns = ['Rank', 'Qubit ID', 'Viability Score', 'T1 (us)', 'T2 (us)', 'Readout Error', 'Status']
        
        # Map dynamic status identifiers cleanly without raw emojis
        display_df['Status'] = display_df['Status'].map({1: 'OPERATIONAL', 0: 'DEGRADED'})
        display_df['Viability Score'] = display_df['Viability Score'].round(4)
        display_df['T1 (us)'] = display_df['T1 (us)'].round(2)
        display_df['T2 (us)'] = display_df['T2 (us)'].round(2)
        display_df['Readout Error'] = display_df['Readout Error'].round(5)
        
        st.dataframe(
            display_df.set_index('Rank'),
            use_container_width=True,
            height=min(450, 45 + 36 * len(display_df))
        )

# ==============================================================================
# TAB 2 — Coherence Trends Canvas (Expanded Architecture View)
# ==============================================================================
with tab2:
    st.subheader("Deep Coherence & Viability Historical Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        trend_backend = st.selectbox("System Platform Selection", DYNAMIC_BACKENDS, key="sb_trend_backend")
    with col_b:
        available_qubits = sorted(
    int(q)
    for q in global_latest[
        global_latest['backend'] == trend_backend
    ]['qubit'].unique()
)

trend_qubit = st.selectbox(
    "Target Node/Qubit ID",
    available_qubits,
    key="sb_trend_qubit"
)
        
try:
        trend_history = get_ranking_history(
    trend_backend,
    int(trend_qubit)
)
        
        if trend_history.empty:
            st.info("No tracking matrix recorded for the designated physical target.")
        elif len(trend_history) < 2:
            st.info("Insufficient historical span. Cross-sectional metrics require at least 2 tracking snapshots.")
        else:
            # Huge, Clear Diagnostic Plot
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Analytical Health Profile — {trend_backend} (Qubit {trend_qubit})", 
                         fontsize=14, color=COLOR_PRIMARY, fontweight='bold')
            
            x_dates = pd.to_datetime(trend_history['snapshot_date']).dt.strftime('%b %d')
            
            # Subplot 1: Viability Evaluation
            axes[0].plot(x_dates, trend_history['viability_score'], marker='s', color=COLOR_PRIMARY, linewidth=2, label='Model Score')
            axes[0].set_ylabel('Viability Score', fontsize=10, fontweight='semibold')
            axes[0].set_ylim(-0.05, 1.05)
            
            # Subplot 2: T1 Target
            axes[1].plot(x_dates, trend_history['T1_us'], marker='o', color=COLOR_SECONDARY, linewidth=2, label='Measured T1')
            axes[1].axhline(100.0, color=COLOR_CRITICAL, linestyle='--', linewidth=1.2, label='Min Threshold (100 us)')
            axes[1].set_ylabel('T1 Relaxation (us)', fontsize=10, fontweight='semibold')
            
            # Subplot 3: T2 Target
            axes[2].plot(x_dates, trend_history['T2_us'], marker='^', color=COLOR_ACCENT, linewidth=2, label='Measured T2')
            axes[2].axhline(50.0, color=COLOR_CRITICAL, linestyle='--', linewidth=1.2, label='Min Threshold (50 us)')
            axes[2].set_ylabel('T2 Dephasing (us)', fontsize=10, fontweight='semibold')
            
            # Subplot 4: Readout Error Target
            axes[3].plot(x_dates, trend_history['readout_error'], marker='v', color=COLOR_MUTED, linewidth=2, label='Measured RE')
            axes[3].axhline(0.05, color=COLOR_CRITICAL, linestyle='--', linewidth=1.2, label='Max Threshold (5% RE)')
            axes[3].set_ylabel('Readout Error Rate', fontsize=10, fontweight='semibold')
            
            # Uniform grid and cleanups
            for ax in axes:
                ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E0')
                ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
                ax.tick_params(labelsize=9)
            
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            
except Exception as e:
        st.error(f"Error drawing historical tracking maps: {e}")

# ==============================================================================
# TAB 3 — Architecture Comparison
# ==============================================================================
with tab3:
    st.subheader("Cross-Platform Calibration & Performance Distribution")
    
    summary_metrics = []
    for b_system in DYNAMIC_BACKENDS:
        sys_subset = global_latest[global_latest['backend'] == b_system]
        if sys_subset.empty:
            continue
            
        best_idx = sys_subset['viability_rank'].idxmin()
        
        summary_metrics.append({
            'System Architecture': b_system,
            'Total Monitored Nodes': len(sys_subset),
            'Yield Capacity (%)': f"{(sys_subset['label'].mean() * 100):.1f}%",
            'System Mean Score': f"{sys_subset['viability_score'].mean():.3f}",
            'Optimal Core ID': int(sys_subset.loc[best_idx, 'qubit']),
            'Peak Core Score': f"{sys_subset['viability_score'].max():.3f}",
            'Mean T1 (us)': f"{sys_subset['T1_us'].mean():.1f}",
            'Mean Readout Error': f"{sys_subset['readout_error'].mean():.4f}"
        })
        
    if summary_metrics:
        st.dataframe(
            pd.DataFrame(summary_metrics).set_index('System Architecture'),
            use_container_width=True
        )
        
        st.write("")
        st.markdown("**Probability Density Analysis (Latest Snapshot)**")
        
        # Professional Score Spread Graph
        fig_dist, ax_dist = plt.subplots(figsize=(12, 4.5))
        
        for i, b_system in enumerate(DYNAMIC_BACKENDS):
            sys_subset = global_latest[global_latest['backend'] == b_system]
            # Use dynamic spacing for colors safely
            color_cycle = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_MUTED]
            c = color_cycle[i % len(color_cycle)]
            
            ax_dist.hist(
                sys_subset['viability_score'], 
                bins=25, 
                alpha=0.7, 
                label=b_system, 
                color=c,
                edgecolor='white',
                linewidth=0.5
            )
            
        ax_dist.set_xlabel('Viability Calibration Score', fontsize=10)
        ax_dist.set_ylabel('Quantum Processor Node Count', fontsize=10)
        ax_dist.legend(loc='upper left', frameon=True)
        ax_dist.grid(True, linestyle='--', alpha=0.4, color='#CBD5E0')
        
        plt.tight_layout()
        st.pyplot(fig_dist)
    else:
        st.info("System distribution maps could not be compiled.")

# -- Production Footer ──
st.divider()
st.caption(

    "**Vattikuti Uday Kiran"
)