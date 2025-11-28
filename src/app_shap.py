import os
import time
import pickle
from typing import List

import numpy as np
import openai
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import urllib.parse
import re


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
            display_lipid = lipid.replace("_", ":")
            vals = fold_df[lipid].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            mean_signed = float(np.mean(vals))
            mean_abs = float(np.mean(np.abs(vals)))
            results.append(
                {
                    "fold": int(fold),
                    # Keep internal (CSV) name with underscores for lookups
                    "lipid_internal": lipid,
                    # Use display-friendly name with ':' everywhere in the app
                    "lipid": display_lipid,
                    "mean_shap": mean_signed,
                    "mean_abs_shap": mean_abs,
                }
            )
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def load_lipid_data() -> pd.DataFrame:
    """Load original lipidomics data (with ':' in lipid names)."""
    file_path = "data/SupplementaryData1-with-age.xlsx"
    sheet_name = "lipidomics_data_males"
    return pd.read_excel(file_path, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def get_refmet_info(lipid_name: str):
    """
    Retrieve RefMet information for a lipid (e.g. 'Cer(d41:3)').
    For standard names, returns a single dict or None.
    For vendor-style combined ether/plasmalogen notations of the form
    'CLASS(O+P-xx:yy)' (e.g. 'PE(O+P-44:9)'), it is converted into two
    RefMet-style candidates 'CLASS O-xx:yy' and 'CLASS P-xx:yy', each
    queried separately. If both exist, a list of dicts is returned; if
    only one exists, that single dict is returned.
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

    def _ether_plasmalogen_candidates(name: str) -> List[str] | None:
        """
        Detect combined ether/plasmalogen notation of the form
        'CLASS(O+P-xx:yy)' and convert it into two RefMet-compatible
        candidates: 'CLASS O-xx:yy' and 'CLASS P-xx:yy'.
        """
        if not isinstance(name, str):
            return None
        m = re.match(r"^(?P<class>[A-Za-z0-9]+)\(O\+P-(?P<chain>[^)]+)\)$", name.strip())
        if not m:
            return None
        cls = m.group("class")
        chain = m.group("chain")
        return [f"{cls} O-{chain}", f"{cls} P-{chain}"]

    base_url = "https://www.metabolomicsworkbench.org/rest/refmet"

    def _query_single(name: str):
        encoded_name = urllib.parse.quote(name, safe="()")
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

        # Fallback: query detailed RefMet record via API
        detail_url = f"{base_url}/refmet_id/{refmet_id}/all"
        try:
            detail_resp = requests.get(detail_url, timeout=10)
            detail_resp.raise_for_status()
            detail_data = detail_resp.json()
            records = _normalize_records(detail_data)
            if records:
                rec = records[0]
                # Augment with refmet_id for downstream mapping
                if "refmet_id" not in rec:
                    rec["refmet_id"] = refmet_id
                return rec
        except Exception:
            return None
        return None

    # Build list of candidate lipid names to try. For combined ether/plasmalogen
    # notation we try both O- and P- forms; otherwise we just use the input.
    candidates_to_try = _ether_plasmalogen_candidates(lipid_name) or [lipid_name]

    results: List[dict] = []
    seen_refmet_ids: set[str] = set()
    for name in candidates_to_try:
        rec = _query_single(name)
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("refmet_id") or "").strip()
        if rid:
            if rid in seen_refmet_ids:
                continue
            seen_refmet_ids.add(rid)
        results.append(rec)

    if not results:
        # Fall back to the existing "no match" behaviour upstream.
        return None
    if len(results) == 1:
        return results[0]
    return results


@st.cache_data(show_spinner=False)
def get_refmet_studies(refmet_name: str):
    """
    Retrieve studies mentioning a given RefMet name using the Metabolomics Workbench API.
    Returns a DataFrame or None if nothing is found.
    """
    base_url = "https://metabolomicsworkbench.org/rest/study/refmet_name"
    refmet_name = (refmet_name or "").strip()
    # Keep ':', ';', and parentheses unescaped; spaces will become %20
    encoded_name = urllib.parse.quote(refmet_name, safe=":;()")
    # JSON-like dict-of-dicts endpoint (easier to parse)
    url = f"{base_url}/{encoded_name}/data/"

    def _records_from_json(obj):
        """
        Normalize various JSON shapes into list-of-dict records.
        Expected shape here is dict-of-dicts with numeric keys:
        {"1": {...}, "2": {...}, ...}
        """
        if obj is None:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # dict-of-dicts with numeric keys
            if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
                return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x)) if isinstance(obj[k], dict)]
            return [obj]
        return []

    try:
        resp = requests.get(url, timeout=15)
        status = resp.status_code
        text = resp.text
        try:
            resp.raise_for_status()
        except Exception:
            return {"df": None, "url": url, "status": status, "raw": text}

        try:
            data = resp.json()
        except Exception:
            return {"df": None, "url": url, "status": status, "raw": text}

        records = _records_from_json(data)
        if not records:
            return {"df": None, "url": url, "status": status, "raw": data}
        df = pd.DataFrame(records)
        if df.empty:
            return {"df": None, "url": url, "status": status, "raw": data}
        return {"df": df, "url": url, "status": status, "raw": data}
    except Exception as e:
        return {"df": None, "url": url, "status": None, "raw": str(e)}


@st.cache_data(show_spinner=False)
def get_study_title(study_id: str) -> str | None:
    """
    Retrieve study title for a given Metabolomics Workbench study ID.
    """
    if not study_id:
        return None
    base_url = "https://metabolomicsworkbench.org/rest/study/study_id"
    url = f"{base_url}/{study_id}/summary"

    def _records_from_json(obj):
        if obj is None:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # dict-of-dicts with numeric keys
            if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
                return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x)) if isinstance(obj[k], dict)]
            return [obj]
        return []

    try:
        # Small delay to be gentle with the API when called repeatedly.
        time.sleep(0.2)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        records = _records_from_json(data)
        if not records:
            return None
        rec = records[0]
        # Try several title-like keys
        for key in ["Title", "Study Title", "study_title", "title"]:
            if key in rec and isinstance(rec[key], str) and rec[key].strip():
                return rec[key].strip()
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_kegg_pathways(kegg_id: str) -> pd.DataFrame | None:
    """
    Retrieve KEGG pathways involving a compound (by KEGG ID, e.g. 'C00422').
    Uses KEGG REST:
      - link/pathway/cpd:{kegg_id} to get pathway IDs
      - list/{pathway_ids} to get pathway descriptions
    """
    if not kegg_id:
        return None
    kegg_id = kegg_id.strip()
    base_url = "https://rest.kegg.jp"

    # Step 1: link compound to pathways
    link_url = f"{base_url}/link/pathway/cpd:{kegg_id}"
    try:
        resp = requests.get(link_url, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
    except Exception:
        return None

    pathway_ids: list[str] = []
    for line in lines:
        if "\t" not in line:
            continue
        _, path_id = line.split("\t", 1)
        path_id = path_id.strip()
        if path_id:
            pathway_ids.append(path_id)

    if not pathway_ids:
        return None

    pathway_ids = sorted(set(pathway_ids))

    # Step 2: get pathway names/descriptions
    joined_ids = "+".join(pathway_ids)
    list_url = f"{base_url}/list/{joined_ids}"
    try:
        # Small delay to be gentle with the API when called repeatedly
        time.sleep(0.2)
        resp2 = requests.get(list_url, timeout=15)
        resp2.raise_for_status()
        lines2 = resp2.text.strip().splitlines()
    except Exception:
        return None

    rows = []
    for line in lines2:
        if "\t" not in line:
            continue
        pid, desc = line.split("\t", 1)
        rows.append({"pathway_id": pid.strip(), "description": desc.strip()})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    return df


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
                    # Compute per-fold means and immediately switch to display-friendly
                    # lipid names (':' instead of '_') for all downstream rendering.
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
                    # Determine internal (underscore) and display (colon) lipid names
                    try:
                        lipid_internal = row["lipid_internal"].iloc[0]
                    except KeyError:
                        # Backward compatibility in case older cached data lacks lipid_internal
                        lipid_internal = lipid_part.replace(":", "_")
                    lipid_display = row["lipid"].iloc[0] if "lipid" in row.columns else lipid_part
                    raw_df = st.session_state.get("raw_df")
                    if raw_df is None:
                        st.warning("Raw data not available.")
                    elif "sample_id" not in raw_df.columns:
                        st.error("'sample_id' column not found in CSV.")
                    elif lipid_internal not in raw_df.columns:
                        st.error(
                            f"Lipid '{lipid_display}' (internal '{lipid_internal}') not found in CSV columns."
                        )
                    else:
                        # Save selected lipid for external APIs (expects ':' notation)
                        st.session_state["selected_lipid_api"] = lipid_display
                        fold_df = raw_df[raw_df["fold"] == fold_value]
                        vals_df = fold_df[["sample_id", lipid_internal]].dropna()
                        if vals_df.empty:
                            st.warning("No SHAP values to display for the selected lipid/fold.")
                        else:
                            vals_df = vals_df.rename(columns={lipid_internal: "shap_value"})
                            vals_df["sample_id"] = vals_df["sample_id"].astype(str)
                            # Attach original lipid concentration values from Excel, if available
                            try:
                                lipid_excel_name = lipid_display
                                lipid_df = load_lipid_data()
                                if "Sample ID" in lipid_df.columns and lipid_excel_name in lipid_df.columns:
                                    meta_df = lipid_df[["Sample ID", lipid_excel_name]].rename(
                                        columns={"Sample ID": "sample_id", lipid_excel_name: "lipid_value"}
                                    )
                                    vals_df = vals_df.merge(meta_df, on="sample_id", how="left")
                            except Exception:
                                # If anything goes wrong, just fall back to plotting without color
                                pass
                            # Create small vertical jitter so points don't overlap on a single y level
                            seed = abs(hash(f"{lipid_display}-{fold_value}")) % (2**32)
                            rng = np.random.default_rng(seed)
                            vals_df = vals_df.copy()
                            vals_df["jitter"] = rng.uniform(-0.3, 0.3, size=len(vals_df))
                            color_kwargs = {}
                            hover_cols = ["sample_id"]
                            if "lipid_value" in vals_df.columns:
                                color_kwargs = {
                                    "color": "lipid_value",
                                    "color_continuous_scale": "Bluered",
                                }
                                hover_cols.append("lipid_value")
                            fig2 = px.scatter(
                                vals_df,
                                x="shap_value",
                                y="jitter",
                                hover_data=hover_cols,
                                labels={"shap_value": "SHAP value"},
                                title=None,
                                **color_kwargs,
                            )
                            fig2.update_traces(marker=dict(size=8, opacity=0.8))
                            fig2.update_yaxes(showticklabels=False, title=None, showgrid=False, zeroline=False)
                            fig2.update_xaxes(range=[-1.1, 1.1])
                            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220)
                            st.plotly_chart(fig2, use_container_width=True)
                            # Correlation of per-instance SHAP values with other lipids (same fold)
                            lipid_cols_all = infer_lipid_columns(raw_df)
                            fold_raw = raw_df[raw_df["fold"] == fold_value]
                            shap_matrix = fold_raw[lipid_cols_all]
                            # Use internal (underscore) name for correlation computations
                            target_col = lipid_internal
                            if target_col in shap_matrix.columns:
                                target = shap_matrix[target_col]
                                corrs = shap_matrix.corrwith(target, method="spearman")
                                corrs = corrs.drop(labels=[target_col], errors="ignore").dropna()
                                if not corrs.empty:
                                    corr_df = corrs.to_frame(name="corr").reset_index().rename(columns={"index": "lipid"})
                                    corr_df["abs_corr"] = corr_df["corr"].abs()
                                    top5 = corr_df.sort_values("abs_corr", ascending=False).head(10)
                                    top5_disp = top5[["lipid", "corr"]].copy()
                                    # Render correlated lipid names with ':' instead of '_'
                                    top5_disp["lipid"] = top5_disp["lipid"].astype(str).str.replace("_", ":", regex=False)
                                    top5_disp["corr"] = top5_disp["corr"].round(3)
                                    # Store for display in the right pane
                                    st.session_state["corr_table"] = top5_disp
                                    st.session_state["corr_title"] = (
                                        f"Top 10 correlated lipids (Spearman) for {lipid_display} (fold {fold_value})"
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
        # st.table(corr_table.style.hide(axis="index"))
        st.dataframe(corr_table, hide_index=True)
    st.subheader("RefMet annotation")
    selected_api_name = st.session_state.get("selected_lipid_api")
    if selected_api_name:
        with st.spinner(f"Querying RefMet for '{selected_api_name}'..."):
            info = get_refmet_info(selected_api_name)
        if info:
            if isinstance(info, list):
                records = [rec for rec in info if isinstance(rec, dict)]
            elif isinstance(info, dict):
                records = [info]
            else:
                records = []
            if records:
                df_info = pd.DataFrame(records)
                # Keep only a focused subset of informative RefMet columns
                keep_cols = [
                    "refmet_id",
                    "name",
                    "exactmass",
                    "formula",
                    "superclass",
                    "main_class",
                    "sub_class",
                ]
                existing_cols = [c for c in keep_cols if c in df_info.columns]
                if existing_cols:
                    df_info = df_info[existing_cols]
                # Remember last RefMet record(s) for use below
                st.session_state["last_refmet_info"] = records
                # st.table(df_info.style.hide(axis="index"))
                st.dataframe(df_info, hide_index=True)
            else:
                st.info("No RefMet match found or API response not in expected format.")
        else:
            st.info("No RefMet match found or API unavailable.")
    else:
        st.caption("Select a lipid-fold on the left to see RefMet details here.")

# --- Studies mentioning the selected lipid ---
st.divider()
st.header("Studies and KEGG pathways mentioning this lipid")
refmet_info = st.session_state.get("last_refmet_info")
if not refmet_info:
    st.caption("Select a lipid-fold and ensure a RefMet match is available to see related studies.")
    # Clear any previous studies/pathways so the language model does not
    # accidentally use stale information from another lipid.
    st.session_state["last_studies_df"] = None
    st.session_state["last_pathways_df"] = None
else:
    # Support both single-record and multi-record RefMet lookups.
    if isinstance(refmet_info, list):
        refmet_records = [r for r in refmet_info if isinstance(r, dict)]
    elif isinstance(refmet_info, dict):
        refmet_records = [refmet_info]
    else:
        refmet_records = []

    if not refmet_records:
        st.caption("No valid RefMet record available for study lookup.")
        # Clear any previous studies/pathways so the language model does not
        # accidentally use stale information from another lipid.
        st.session_state["last_studies_df"] = None
        st.session_state["last_pathways_df"] = None
    else:
        all_studies: list[pd.DataFrame] = []
        total_refmets = len(refmet_records)

        with st.spinner("Retrieving studies for matched RefMet lipids..."):
            for idx, rec in enumerate(refmet_records, start=1):
                refmet_name = rec.get("name")
                if not isinstance(refmet_name, str) or not refmet_name.strip():
                    continue
                result = get_refmet_studies(refmet_name)
                if not result:
                    continue
                studies_df_single = result.get("df")
                if studies_df_single is None or studies_df_single.empty:
                    continue
                studies_df_single = studies_df_single.copy()
                # Track which RefMet name each study row came from
                studies_df_single["Lipid name"] = refmet_name
                all_studies.append(studies_df_single)

        if not all_studies:
            st.info("No studies found mentioning these lipids (or API unavailable).")
            # Clear any previous studies/pathways so the language model does not
            # accidentally use stale information from another lipid.
            st.session_state["last_studies_df"] = None
            st.session_state["last_pathways_df"] = None
        else:
            studies_df = pd.concat(all_studies, ignore_index=True)
            # Limit to at most 50 rows as early as possible to keep downstream
            # lookups and rendering lightweight.
            max_study_rows = 50
            if len(studies_df) > max_study_rows:
                studies_df = studies_df.head(max_study_rows)

            # Enrich with study titles using the study_id column
            id_col = None
            for cand in ["study_id", "STUDY_ID"]:
                if cand in studies_df.columns:
                    id_col = cand
                    break
            if id_col is not None:
                titles = {}
                sids = sorted(studies_df[id_col].dropna().astype(str).unique())
                # Cap the number of study titles to retrieve to avoid excessive API calls
                n_titles = len(sids)
                title_progress = st.progress(0.0) if n_titles > 1 else None
                with st.spinner("Fetching titles for Metabolomics Workbench studies..."):
                    for i, sid in enumerate(sids, start=1):
                        titles[sid] = get_study_title(sid)
                        if title_progress is not None:
                            title_progress.progress(min(i / n_titles, 1.0))
                if title_progress is not None:
                    title_progress.empty()
                studies_df = studies_df.copy()
                studies_df["Title"] = studies_df[id_col].astype(str).map(titles)

            # For display, keep only the study ID and title (when available)
            if id_col is not None and "Title" in studies_df.columns:
                display_df = studies_df[[id_col, "Title"]].rename(columns={id_col: "study_id"})
            else:
                display_df = studies_df

            # Store for downstream summary generation (use the displayed table)
            st.session_state["last_studies_df"] = display_df

            # Layout: studies on the left, KEGG pathways on the right
            studies_col, pathways_col = st.columns(2)

            with studies_col:
                st.subheader("Studies")
                # Simple pagination for the studies table
                total_rows = len(display_df)
                page_size = 10
                total_pages = max(1, (total_rows + page_size - 1) // page_size)
                if total_pages > 1:
                    page = st.number_input(
                        "Studies page",
                        min_value=1,
                        max_value=int(total_pages),
                        value=1,
                        step=1,
                    )
                else:
                    page = 1
                start = (int(page) - 1) * page_size
                end = start + page_size
                page_df = display_df.iloc[start:end]
                st.caption(
                    f"Showing studies {start + 1}–{min(end, total_rows)} of {total_rows} "
                    f"(page {page}/{total_pages})"
                )
                # st.table(page_df.style.hide(axis="index"))
                st.dataframe(page_df, hide_index=True)

            with pathways_col:
                st.subheader("KEGG pathways for associated KEGG IDs")
                pw_df = None
                if "kegg_id" in studies_df.columns:
                    pw_frames: list[pd.DataFrame] = []
                    kegg_pairs = list(
                        studies_df[["Lipid name", "kegg_id"]]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                    n_pairs = len(kegg_pairs)
                    kegg_progress = st.progress(0.0) if n_pairs > 0 else None
                    try:
                        with st.spinner("Retrieving KEGG pathways for associated KEGG IDs..."):
                            for i, (lipid_nm, kid) in enumerate(kegg_pairs, start=1):
                                if not kid:
                                    continue
                                p = get_kegg_pathways(kid)
                                if p is None or p.empty:
                                    if kegg_progress is not None:
                                        kegg_progress.progress(min(i / n_pairs, 1.0))
                                    continue
                                tmp = p.copy()
                                tmp["kegg_id"] = kid
                                tmp["Lipid name"] = lipid_nm
                                pw_frames.append(tmp)
                                if kegg_progress is not None:
                                    kegg_progress.progress(min(i / n_pairs, 1.0))
                    except Exception:
                        # If KEGG lookup fails for any reason, we still want the
                        # studies table to remain visible; just skip pathways.
                        pw_frames = []
                    finally:
                        if kegg_progress is not None:
                            kegg_progress.empty()
                    if pw_frames:
                        pw_df = pd.concat(pw_frames, ignore_index=True)

                st.session_state["last_pathways_df"] = pw_df if pw_df is not None else None

                if pw_df is not None and not pw_df.empty:
                    # Display only selected columns for the KEGG pathways table
                    pw_cols_order = ["kegg_id", "pathway_id", "description"]
                    pw_cols = [c for c in pw_cols_order if c in pw_df.columns]
                    if not pw_cols:
                        pw_cols = list(pw_df.columns)
                    # st.table(pw_df[pw_cols].style.hide(axis="index"))
                    st.dataframe(pw_df[pw_cols], hide_index=True)
                else:
                    st.caption("No KEGG pathways found for the associated KEGG IDs.")


# --- Language model summary section ---
st.divider()
st.header("Language-model summary")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.caption("Set the OPENAI_API_KEY environment variable to enable language-model summaries.")
else:
    # Gather context pieces
    corr_table = st.session_state.get("corr_table")
    refmet_info_any = st.session_state.get("last_refmet_info")
    if isinstance(refmet_info_any, list):
        refmet_info_list = [r for r in refmet_info_any if isinstance(r, dict)]
        refmet_info = refmet_info_list[0] if refmet_info_list else {}
    elif isinstance(refmet_info_any, dict):
        refmet_info = refmet_info_any
    else:
        refmet_info = {}
    studies_display_df = st.session_state.get("last_studies_df")
    pathways_df = st.session_state.get("last_pathways_df")

    # Build text snippets
    corr_text = "None"
    if corr_table is not None and not corr_table.empty:
        rows = [f"{row['lipid']}: corr={row['corr']:.3f}" for _, row in corr_table.iterrows()]
        corr_text = "; ".join(rows)

    superclass = refmet_info.get("super_class") or refmet_info.get("superclass") or ""
    main_class = refmet_info.get("main_class") or refmet_info.get("mainclass") or ""
    sub_class = refmet_info.get("sub_class") or refmet_info.get("subclass") or ""
    kegg_id = refmet_info.get("kegg_id") or ""
    lipid_name = refmet_info.get("name") or st.session_state.get("selected_lipid_api") or ""

    studies_text = "None"
    if studies_display_df is not None and not studies_display_df.empty:
        study_rows = []
        for _, r in studies_display_df.iterrows():
            sid = r.get("study_id") or r.get("STUDY_ID")
            title = r.get("Title")
            species = r.get("Species")
            parts = [str(sid) if pd.notna(sid) else None,
                     str(title) if pd.notna(title) else None,
                     f"Species: {species}" if pd.notna(species) else None]
            study_rows.append(", ".join([p for p in parts if p]))
        studies_text = " | ".join(study_rows)

    pathways_text = "None"
    if pathways_df is not None and hasattr(pathways_df, "empty") and not pathways_df.empty:
        pw_rows = []
        for _, r in pathways_df.iterrows():
            pid = r.get("pathway_id")
            desc = r.get("description")
            if pd.isna(pid) and pd.isna(desc):
                continue
            pw_rows.append(f"{pid}: {desc}")
        if pw_rows:
            pathways_text = " | ".join(pw_rows)

    prompt = f"""
You are assisting with interpretation of lipidomic SHAP feature importance for predicting the presence of adrenal insufficiency in patients with ALD.

Lipid of interest: {lipid_name}
RefMet classes:
  - Super class: {superclass}
  - Main class: {main_class}
  - Sub class: {sub_class}
  - KEGG ID: {kegg_id}

Top correlated lipids (lipid: correlation):
{corr_text}

Relevant Metabolomics Workbench studies (study id, title, species):
{studies_text}

KEGG pathways involving this lipid (pathway id: description):
{pathways_text}

Write a brief text summarizing the results in the following sections:
- what type of lipid this is and its likely biological role,
- how its SHAP importance and correlations might relate to adrenal insufficiency,
- and how the listed studies and pathways could contextualize or support these interpretations.
Avoid speculation that is not grounded in the provided information; when extrapolating, use cautious language ("may", "could", "suggests"). The overall goal is to contextualize the lipid of interest in the context of adrenal insufficiency in ALD.
"""

    if st.button("Generate summary with language model"):
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://ai-research-proxy.azurewebsites.net",
            )
            placeholder = st.empty()
            full_text = ""
            with st.spinner("Calling language model..."):
                for chunk in client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt.strip()}],
                    stream=True,
                ):
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content = delta.content or ""
                    if not content:
                        continue
                    full_text += content
                    placeholder.markdown(full_text)
        except Exception as e:
            st.error(f"Error while calling language model: {e}")

