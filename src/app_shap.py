import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import urllib.parse


def infer_lipid_columns(df: pd.DataFrame) -> List[str]:
    meta_cols = {
        'k',
        'model_type',
        'normalize',
        'imputer',
        'fold',
        'sample_id',
        'age',
        'true_adrenal_insufficiency',
        'pred_adrenal_insufficiency',
    }
    return [c for c in df.columns if c not in meta_cols]


def compute_per_fold_means(df: pd.DataFrame) -> pd.DataFrame:
    lipid_cols = infer_lipid_columns(df)
    results = []
    for fold, fold_df in df.groupby("fold"):
        for lipid in lipid_cols:
            vals = fold_df[lipid].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            mean_signed = float(np.mean(vals))
            mean_abs = float(np.mean(np.abs(vals)))
            results.append(
                {
                    "fold": int(fold),
                    "lipid": lipid,
                    "mean_shap": mean_signed,
                    "mean_abs_shap": mean_abs,
                }
            )
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def get_refmet_info(lipid_name: str):
    """
    Retrieve RefMet information for a lipid (e.g. 'Cer(d41:3)').
    Returns dict or None.
    """
    def _normalize_records(obj):
        # Convert various API JSON shapes into a list of record dicts
        if obj is None:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # If keys are numeric strings: {"0": {...}, "1": {...}}
            if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
                return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x)) if isinstance(obj[k], dict)]
            # If it looks like a single record
            return [obj]
        return []

    def _get_case_insensitive(d, key):
        # Return value for key in dict d, ignoring case
        if not isinstance(d, dict):
            return None
        for k, v in d.items():
            if isinstance(k, str) and k.lower() == key.lower():
                return v
        return None

    base_url = "https://www.metabolomicsworkbench.org/rest/refmet"
    encoded_name = urllib.parse.quote(lipid_name, safe="()")
    match_url = f"{base_url}/match/{encoded_name}/name/"
    try:
        match_resp = requests.get(match_url, timeout=10)
        match_resp.raise_for_status()
        match_data = match_resp.json()
    except Exception:
        return None
    candidates = _normalize_records(match_data)
    if not candidates:
        return None
    # Find first candidate with a refmet_id
    refmet_id = None
    for cand in candidates:
        refmet_id = _get_case_insensitive(cand, "refmet_id")
        if refmet_id:
            break
    if not refmet_id:
        return None
    detail_url = f"{base_url}/refmet_id/{refmet_id}/all"
    try:
        detail_resp = requests.get(detail_url, timeout=10)
        detail_resp.raise_for_status()
        detail_data = detail_resp.json()
        records = _normalize_records(detail_data)
        if records:
            return records[0]
    except Exception:
        return None
    return None


st.set_page_config(layout="wide", page_title="SHAP Lipid Importance")
left_col, right_col = st.columns(2)

with left_col:
    st.header("Top 10 Lipids by Mean |SHAP|")
    # Hardcoded experiment directory (temporary)
    exp_dir = "experiments/v4/2025-08-31-235806-1e9e1f"
    st.caption(f"Using experiment folder: {exp_dir}")
    csv_path = os.path.join(exp_dir, "instance_shap_table.csv")

    if exp_dir:
        if not os.path.exists(csv_path):
            st.warning(f"No CSV found at: {csv_path}")
        else:
            try:
                df = pd.read_csv(csv_path)
                if "fold" not in df.columns:
                    st.error("'fold' column not found in CSV.")
                else:
                    fold_means = compute_per_fold_means(df)
                    st.session_state["fold_means"] = fold_means
                    st.session_state["raw_df"] = df
                    if fold_means.empty:
                        st.warning("No SHAP values available to aggregate.")
                    else:
                        # Build "<lipid>-<fold>" entries, sort by |mean SHAP|, and plot signed mean SHAP
                        fold_means = fold_means.copy()
                        fold_means["lipid_fold"] = fold_means["lipid"].astype(str) + "-" + fold_means["fold"].astype(str)
                        top10 = fold_means.sort_values("mean_abs_shap", ascending=False).head(10)
                        top10_sorted = top10.sort_values("mean_abs_shap")
                        # Save list of lipid-fold entries currently plotted for use in the right pane
                        y_order = top10_sorted["lipid_fold"].tolist()
                        st.session_state["top_lipid_folds"] = y_order
                        fig = px.bar(top10_sorted,
                                     x="mean_abs_shap",
                                     y="lipid_fold",
                                     orientation="h",
                                     labels={"mean_abs_shap": "Mean |SHAP|", "lipid_fold": "Lipid-Fold"},
                                     title=None)
                        # Ensure y-axis category order matches the left-plot order
                        fig.update_yaxes(categoryorder="array", categoryarray=y_order)
                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.exception(e)

    st.header("Details")
    fm = st.session_state.get("fold_means")
    if fm is None or fm.empty:
        st.info("Load a valid experiment to search lipids and folds.")
    else:
        fm = fm.copy()
        fm["lipid_fold"] = fm["lipid"].astype(str) + "-" + fm["fold"].astype(str)
        # Prefer the lipid-folds shown on the left plot; fall back to all candidates
        plotted = st.session_state.get("top_lipid_folds")
        if plotted is not None and len(plotted) > 0:
            # Reverse to match the visual top-to-bottom order in the left plot
            options = list(reversed(plotted))
        else:
            options = sorted(fm["lipid_fold"].unique())
        if not options:
            st.info("No lipid-fold options available.")
            selection = None
        else:
            selection = st.selectbox("Select lipid-fold", options)
        if selection:
            try:
                lipid_part, fold_part = selection.rsplit("-", 1)
                fold_value = int(fold_part)
            except Exception:
                lipid_part = selection
                fold_value = None
            if fold_value is not None:
                row = fm[(fm["lipid"] == lipid_part) & (fm["fold"] == fold_value)]
                if row.empty:
                    st.warning("Selection not found in aggregated table.")
                else:
                    raw_df = st.session_state.get("raw_df")
                    if raw_df is None:
                        st.warning("Raw data not available.")
                    elif "sample_id" not in raw_df.columns:
                        st.error("'sample_id' column not found in CSV.")
                    elif lipid_part not in raw_df.columns:
                        st.error(f"Lipid '{lipid_part}' not found in CSV columns.")
                    else:
                        # Save selected lipid (API format uses ':' instead of '_')
                        st.session_state["selected_lipid_api"] = lipid_part.replace("_", ":")
                        fold_df = raw_df[raw_df["fold"] == fold_value]
                        vals_df = fold_df[["sample_id", lipid_part]].dropna()
                        if vals_df.empty:
                            st.warning("No SHAP values to display for the selected lipid/fold.")
                        else:
                            vals_df = vals_df.rename(columns={lipid_part: "shap_value"})
                            vals_df["sample_id"] = vals_df["sample_id"].astype(str)
                            # Create small vertical jitter so points don't overlap on a single y level
                            seed = abs(hash(f"{lipid_part}-{fold_value}")) % (2**32)
                            rng = np.random.default_rng(seed)
                            vals_df = vals_df.copy()
                            vals_df["jitter"] = rng.uniform(-0.3, 0.3, size=len(vals_df))
                            fig2 = px.scatter(vals_df,
                                              x="shap_value",
                                              y="jitter",
                                              hover_data=["sample_id"],
                                              labels={"shap_value": "SHAP value"},
                                              title=None)
                            fig2.update_traces(marker=dict(size=8, opacity=0.8))
                            fig2.update_yaxes(showticklabels=False, title=None, showgrid=False, zeroline=False)
                            fig2.update_xaxes(range=[-1.1, 1.1])
                            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220)
                            st.plotly_chart(fig2, use_container_width=True)
                            # Correlation of per-instance SHAP values with other lipids (same fold)
                            lipid_cols_all = infer_lipid_columns(raw_df)
                            fold_raw = raw_df[raw_df["fold"] == fold_value]
                            shap_matrix = fold_raw[lipid_cols_all]
                            if lipid_part in shap_matrix.columns:
                                target = shap_matrix[lipid_part]
                                corrs = shap_matrix.corrwith(target, method="spearman")
                                corrs = corrs.drop(labels=[lipid_part], errors="ignore").dropna()
                                if not corrs.empty:
                                    corr_df = corrs.to_frame(name="corr").reset_index().rename(columns={"index": "lipid"})
                                    corr_df["abs_corr"] = corr_df["corr"].abs()
                                    top5 = corr_df.sort_values("abs_corr", ascending=False).head(10)
                                    top5_disp = top5[["lipid", "corr"]].copy()
                                    top5_disp["corr"] = top5_disp["corr"].round(3)
                                    # Store for display in the right pane
                                    st.session_state["corr_table"] = top5_disp
                                    st.session_state["corr_title"] = (
                                        f"Top 10 correlated lipids (Spearman) for {lipid_part} (fold {fold_value})"
                                    )
            else:
                st.warning("Could not parse fold from selection.")

with right_col:
    st.header("Correlated lipids")
    corr_table = st.session_state.get("corr_table")
    if corr_table is None or corr_table.empty:
        st.info("Select a lipid-fold on the left to see correlated lipids.")
    else:
        title = st.session_state.get("corr_title", "Top correlated lipids (Spearman)")
        st.subheader(title)
        st.table(corr_table)

# --- RefMet annotation section (below both panes) ---
st.divider()
st.header("RefMet annotation")
selected_api_name = st.session_state.get("selected_lipid_api")
if selected_api_name:
    with st.spinner(f"Querying RefMet for '{selected_api_name}'..."):
        info = get_refmet_info(selected_api_name)
    if info:
        df_info = pd.DataFrame([info])
        st.table(df_info)
    else:
        st.info("No RefMet match found or API unavailable.")
else:
    st.caption("Select a lipid-fold above to see RefMet details here.")


