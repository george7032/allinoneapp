"""
Autonomous AI Data Scientist
=============================
A Streamlit application that orchestrates a multi-agent pipeline (powered by
the Claude API) to investigate an uploaded dataset end-to-end:

Upload -> Understand -> Audit -> Discover Problem -> Select Paradigm ->
Generate Experiments -> Train -> Critique -> Improve -> Compare -> Explain ->
Recommend -> Report

Design principle
-----------------
Reasoning, hypothesis generation and critique are delegated to Claude
(non-deterministic, exploratory). Actual data profiling, transformations,
model fitting and metric computation are done deterministically in Python
with pandas / scikit-learn, and their real outputs are fed back to Claude for
interpretation. Claude is never allowed to "invent" a metric - every number
in the final report traces back to code that computed it.

Run locally with:
    pip install -r requirements.txt
    streamlit run main.py

You will be prompted for an Anthropic API key in the sidebar (or set the
ANTHROPIC_API_KEY environment variable before launching).
"""

from __future__ import annotations

import io
import json
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# scikit-learn is used for the deterministic modelling stages
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Anthropic client is imported lazily so the app can still start (and show a
# helpful message) even if the package or an API key isn't available yet.
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 3000


# ===========================================================================
# 1. SYSTEM PROMPTS  (one per agent, condensed from the project spec)
# ===========================================================================

MASTER_SYSTEM_PROMPT = """You are an Autonomous AI Data Scientist operating inside an advanced data
science platform. You behave like a senior interdisciplinary data scientist,
ML engineer, statistician and research scientist - not a simple chatbot.

Core principles:
- Never assume ML is always necessary; recommend statistics when that is more appropriate.
- Never force a paradigm (e.g. reinforcement learning) onto data that doesn't support it.
- Every experiment needs: a hypothesis, rationale, methodology, measurable evaluation
  criteria, reproducible configuration and interpretation.
- Never fabricate results, invent metrics, or claim a model was trained if it was not.
- Always distinguish OBSERVED / CALCULATED / INFERRED / HYPOTHESIZED / RECOMMENDED.
- Prioritize methodological correctness over impressive-looking results.
- You will always be given real, computed data (profiling stats, real model metrics).
  Reason ONLY from what is given to you; do not invent numbers.

Always respond with a single valid JSON object matching the schema you are given
in the user message. Do not include markdown code fences, preambles, or any text
outside the JSON object.
"""

STAGE_PROMPTS: Dict[str, str] = {
    "understanding": """You are the Dataset Understanding Agent.
Given dataset profiling results (not raw data), construct a structured understanding.
Do not build a model yet. Do not assume the user's intended objective.
Generate several plausible analytical opportunities with a confidence score each.

Respond as JSON with keys:
dataset_summary (string), variable_classification (object mapping column name to one of
["numeric","categorical","boolean","datetime","text","identifier"]),
potential_targets (array of {column, reason, confidence}),
potential_predictors (array of strings),
temporal_structure (string), possible_problem_types (array of strings),
data_quality_concerns (array of strings), analytical_opportunities (array of strings),
uncertainties (array of strings).""",

    "audit": """You are the Autonomous Data Quality Auditor.
Given real profiling statistics (missingness, cardinality, dtypes, outlier counts,
duplicate counts), identify data quality problems. Do not modify data yourself.

Respond as JSON with key "issues": an array of objects each with:
{column, issue, severity ("LOW"|"MEDIUM"|"HIGH"|"CRITICAL"), evidence,
potential_impact, recommended_treatment, safe_to_auto_correct (boolean)}.""",

    "cleaning": """You are the Data Cleaning Strategist.
Using the audit findings, design a safe preprocessing pipeline. Never delete
information without justification. Never risk target leakage. Do not modify
target variables unless explicitly justified.

Respond as JSON with key "pipeline": an ordered array of objects each with:
{column, problem, transformation, reason, risk, reversible (boolean),
human_approval_recommended (boolean)}.""",

    "problem_discovery": """You are the Problem Discovery Agent.
Do not assume a problem has been defined. Using the dataset understanding,
generate multiple candidate problem formulations across paradigms such as
descriptive statistics, inferential statistics, regression, classification,
clustering, anomaly detection, dimensionality reduction, forecasting,
survival analysis, causal analysis, reinforcement learning, NLP, optimization.

Respond as JSON with key "candidate_problems": an array of objects each with:
{problem_statement, learning_paradigm, target, features (array), feasibility
("LOW"|"MEDIUM"|"HIGH"), confidence (0-1), risks (array), recommended_models (array)}.
Order the array from most to least recommended.""",

    "paradigm_selection": """You are the Modelling Paradigm Selection Agent.
Given the top candidate problem, evaluate suitability of supervised, unsupervised,
time-series, anomaly detection, statistical, causal, reinforcement learning and
hybrid approaches. Do not select reinforcement learning unless the data clearly
supports states/actions/rewards/transitions.

Respond as JSON with keys:
primary_paradigm (one of "supervised_classification","supervised_regression",
"unsupervised_clustering","time_series","anomaly_detection","statistical_only",
"reinforcement_learning","not_feasible"),
alternative_paradigms (array of strings), reasoning (string), confidence (0-1).""",

    "experiment_design": """You are the Autonomous Experiment Designer.
Given the chosen paradigm and dataset understanding, propose a diverse set of
2 to 4 candidate model configurations to be trained deterministically by the
platform (do not train them yourself). Prefer meaningful diversity over
superficial parameter tweaks.

Respond as JSON with key "experiments": an array of objects each with:
{experiment_id, hypothesis, model (string, must be one of the ALLOWED_MODELS
provided to you), features (array of column names to use, or "all"),
reasoning, expected_strength, expected_risk}.""",

    "critic": """You are the Adversarial Model Critic. Your job is to challenge results,
not defend them. You are given REAL computed metrics from real trained models.
Investigate possible overfitting, underfitting, leakage, unstable validation,
class imbalance effects, suspiciously high performance, and interpretability
concerns. Do not accept high performance as proof of quality on its own.

Respond as JSON with key "critiques": an array of objects each with:
{experiment_id, concern, severity ("LOW"|"MEDIUM"|"HIGH"), test_recommended,
verdict ("SURVIVES_SCRUTINY"|"NEEDS_IMPROVEMENT"|"REJECT")}.""",

    "improvement": """You are the Model Improvement Agent. Using the critic's findings,
propose the next round of concrete improvements. Do not just increase model
complexity for its own sake; each change must address a specific weakness found.

Respond as JSON with key "improvements": an array of objects each with:
{problem_detected, proposed_change, expected_effect, risk}.""",

    "comparison": """You are the Model Evaluation Scientist. Given real metrics for all
trained experiments, compare them. Do not rank on a single metric unless that
metric is clearly the right one for the problem type. Never invent a missing metric.

Respond as JSON with keys:
best_performing_model (experiment_id), best_efficiency_model (experiment_id),
overall_recommendation (experiment_id), reasoning (string).""",

    "explainability": """You are the Model Explainability Agent. Explain the selected
model's real, computed feature importances / coefficients to both technical and
non-technical audiences. Never claim causation from predictive/correlational
relationships.

Respond as JSON with keys:
technical_summary (string), plain_language_summary (string),
key_drivers (array of {feature, effect_direction, note}),
limitations (array of strings).""",

    "business_insight": """You are the Business Intelligence Agent. Translate the
validated analytical results into practical insights. Do not invent business
conclusions not supported by the evidence provided.

Respond as JSON with key "insights": an array of objects each with:
{observation, evidence, business_meaning, potential_action, expected_benefit,
risk, category ("FACT"|"INTERPRETATION"|"RECOMMENDATION")}.""",

    "reflection": """You are the Autonomous Research Reflection Agent. Review the
complete analytical process supplied to you and be honest about uncertainty.
It is acceptable and expected to say "we don't know yet" where appropriate.

Respond as JSON with keys:
confirmed_findings (array), uncertain_findings (array), rejected_hypotheses (array),
unresolved_questions (array), recommended_future_experiments (array).""",

    "final_report": """You are the Chief AI Data Scientist. Synthesize the entire
investigation supplied to you into a final narrative report. Clearly label
statements as OBSERVED / CALCULATED / INFERRED / HYPOTHESIZED / RECOMMENDED.
Do not fabricate anything not present in the supplied context.

Respond as JSON with keys:
executive_summary (string), dataset_understanding (string), data_quality (string),
problem_discovered (string), paradigm_selected (string), experiments_performed (string),
best_model (string), key_insights (array of strings), limitations (array of strings),
business_recommendations (array of strings), research_recommendations (array of strings),
overall_confidence ("LOW"|"MEDIUM"|"HIGH"), suggested_next_steps (array of strings).""",
}

ALLOWED_CLASSIFICATION_MODELS = [
    "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier",
]
ALLOWED_REGRESSION_MODELS = [
    "LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor",
]
ALLOWED_CLUSTERING_MODELS = ["KMeans"]


# ===========================================================================
# 2. CLAUDE CALL HELPER
# ===========================================================================

def get_client(api_key: str):
    if anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )
    if not api_key:
        raise RuntimeError("No Anthropic API key provided.")
    return anthropic.Anthropic(api_key=api_key)


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the largest {...} span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {"raw_response": text, "parse_error": True}


def call_agent(
    client,
    model: str,
    stage_key: str,
    context: Dict[str, Any],
    extra_instructions: str = "",
) -> Dict[str, Any]:
    """Call Claude for a given pipeline stage with a JSON context payload."""
    system_prompt = MASTER_SYSTEM_PROMPT + "\n\n" + STAGE_PROMPTS[stage_key]
    user_content = (
        (extra_instructions + "\n\n" if extra_instructions else "")
        + "CONTEXT (JSON):\n"
        + json.dumps(context, default=str, indent=2)
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    full_text = "\n".join(text_parts)
    return _extract_json(full_text)


# ===========================================================================
# 3. DETERMINISTIC DATA PROFILING (no LLM involved)
# ===========================================================================

def profile_dataframe(df: pd.DataFrame, max_categories_sample: int = 8) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    columns_info = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        inferred_kind = "text"
        if pd.api.types.is_bool_dtype(series):
            inferred_kind = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            inferred_kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_kind = "datetime"
        else:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_dt = pd.to_datetime(series, errors="coerce")
            if parsed_dt.notna().mean() > 0.9 and n_unique > 1:
                inferred_kind = "datetime"
            elif n_unique <= max(20, int(0.05 * n_rows)):
                inferred_kind = "categorical"

        is_potential_id = n_unique == n_rows and n_rows > 1

        sample_values = (
            series.dropna().unique()[:max_categories_sample].tolist()
            if inferred_kind in ("categorical", "text", "boolean")
            else []
        )
        sample_values = [str(v) for v in sample_values]

        columns_info.append(
            {
                "name": col,
                "dtype": dtype,
                "inferred_kind": inferred_kind,
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n_rows, 2) if n_rows else 0,
                "n_unique": n_unique,
                "is_potential_id": is_potential_id,
                "sample_values": sample_values,
            }
        )

    numeric_cols = [c["name"] for c in columns_info if c["inferred_kind"] == "numeric"]
    numeric_summary = {}
    if numeric_cols:
        numeric_summary = json.loads(df[numeric_cols].describe().to_json())

    corr_pairs = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        seen = set()
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 == c2 or (c2, c1) in seen:
                    continue
                seen.add((c1, c2))
                val = corr.loc[c1, c2]
                if pd.notna(val) and abs(val) >= 0.5:
                    corr_pairs.append({"col_a": c1, "col_b": c2, "correlation": round(float(val), 3)})
        corr_pairs.sort(key=lambda x: -abs(x["correlation"]))

    n_duplicate_rows = int(df.duplicated().sum())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns_info,
        "numeric_summary": numeric_summary,
        "high_correlation_pairs": corr_pairs[:15],
        "n_duplicate_rows": n_duplicate_rows,
        "pct_duplicate_rows": round(100 * n_duplicate_rows / n_rows, 2) if n_rows else 0,
    }


# ===========================================================================
# 4. DETERMINISTIC MODELLING
# ===========================================================================

@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    task_type: str  # "classification" | "regression" | "clustering"
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers)


def _get_model_instance(name: str, task_type: str):
    registry = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoostingClassifier": lambda: GradientBoostingClassifier(random_state=42),
        "LinearRegression": lambda: LinearRegression(),
        "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoostingRegressor": lambda: GradientBoostingRegressor(random_state=42),
    }
    if name not in registry:
        raise ValueError(f"Unknown or disallowed model: {name}")
    return registry[name]()


def run_supervised_experiment(
    df: pd.DataFrame, target: str, features: List[str], model_name: str, task_type: str, experiment_id: str
) -> ExperimentResult:
    try:
        features = [f for f in features if f in df.columns and f != target]
        if not features:
            raise ValueError("No valid feature columns supplied.")
        data = df[features + [target]].dropna(subset=[target]).copy()
        X = data[features]
        y = data[target]

        if task_type == "classification" and y.nunique() < 2:
            raise ValueError("Target has fewer than 2 classes; classification not possible.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if (task_type == "classification" and y.nunique() > 1) else None,
        )

        preprocessor = _build_preprocessor(X)
        model = _get_model_instance(model_name, task_type)
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics: Dict[str, Any] = {}
        if task_type == "classification":
            metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
            metrics["f1_macro"] = round(
                float(f1_score(y_test, preds, average="macro", zero_division=0)), 4
            )
            try:
                cv_scores = cross_val_score(pipe, X, y, cv=min(5, y.value_counts().min()), scoring="accuracy")
                metrics["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
                metrics["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
            except Exception:
                metrics["cv_accuracy_mean"] = None
        else:
            metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
            try:
                cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
                metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
                metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)
            except Exception:
                metrics["cv_r2_mean"] = None

        metrics["n_train"] = int(len(X_train))
        metrics["n_test"] = int(len(X_test))

        feature_importance: Dict[str, float] = {}
        try:
            fitted_model = pipe.named_steps["model"]
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            if hasattr(fitted_model, "feature_importances_"):
                importances = fitted_model.feature_importances_
            elif hasattr(fitted_model, "coef_"):
                coef = fitted_model.coef_
                importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            else:
                importances = None
            if importances is not None:
                pairs = sorted(zip(feature_names, importances), key=lambda x: -abs(x[1]))[:10]
                feature_importance = {str(k): round(float(v), 4) for k, v in pairs}
        except Exception:
            pass

        return ExperimentResult(experiment_id, model_name, task_type, metrics, feature_importance)
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(experiment_id, model_name, task_type, {}, {}, error=str(exc))


def run_clustering_experiment(
    df: pd.DataFrame, features: List[str], experiment_id: str, k_range=range(2, 8)
) -> ExperimentResult:
    try:
        features = [f for f in features if f in df.columns]
        data = df[features].dropna()
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] == 0:
            raise ValueError("No numeric features available for clustering.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        best_k, best_score, best_labels = None, -1.0, None
        scores_by_k = {}
        for k in k_range:
            if k >= len(X_scaled):
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            scores_by_k[k] = round(float(score), 4)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        metrics = {
            "best_k": best_k,
            "best_silhouette_score": round(float(best_score), 4) if best_k else None,
            "silhouette_by_k": scores_by_k,
            "n_samples_used": int(len(X_scaled)),
        }
        return ExperimentResult(experiment_id, "KMeans", "clustering", metrics, {})
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(experiment_id, "KMeans", "clustering", {}, {}, error=str(exc))


# ===========================================================================
# 5. STREAMLIT APP
# ===========================================================================

def init_state():
    defaults = {
        "df": None,
        "profile": None,
        "results": {},
        "experiment_results": [],
        "pipeline_ran": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar_config():
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Not stored anywhere except this browser session.",
    )
    model = st.sidebar.selectbox(
        "Claude model",
        ["claude-sonnet-4-6", "claude-opus-4-1", "claude-haiku-4-5"],
        index=0,
    )
    st.sidebar.caption(
        "The reasoning/critique stages call this model. Model training and "
        "metrics are always computed locally with scikit-learn, never by the LLM."
    )
    return api_key, model


def show_json(obj: Any):
    st.json(obj)


def run_pipeline(df: pd.DataFrame, api_key: str, model: str):
    results = st.session_state.results
    progress = st.progress(0.0, text="Starting pipeline...")
    steps_total = 12
    step = 0

    def bump(msg):
        nonlocal step
        step += 1
        progress.progress(min(step / steps_total, 1.0), text=msg)

    try:
        client = get_client(api_key)
    except Exception as exc:
        st.error(f"Could not initialize Anthropic client: {exc}")
        return

    # 1. Profiling (deterministic)
    bump("Profiling dataset...")
    profile = profile_dataframe(df)
    st.session_state.profile = profile

    # 2. Understanding
    bump("Understanding dataset (Claude)...")
    results["understanding"] = call_agent(client, model, "understanding", profile)

    # 3. Audit
    bump("Auditing data quality (Claude)...")
    results["audit"] = call_agent(client, model, "audit", profile)

    # 4. Cleaning strategy
    bump("Designing cleaning strategy (Claude)...")
    results["cleaning"] = call_agent(
        client, model, "cleaning", {"profile": profile, "audit": results["audit"]}
    )

    # 5. Problem discovery
    bump("Discovering candidate problems (Claude)...")
    results["problem_discovery"] = call_agent(
        client, model, "problem_discovery", {"understanding": results["understanding"]}
    )

    top_problem = None
    candidates = results["problem_discovery"].get("candidate_problems", [])
    if candidates:
        top_problem = candidates[0]

    # 6. Paradigm selection
    bump("Selecting modelling paradigm (Claude)...")
    results["paradigm_selection"] = call_agent(
        client, model, "paradigm_selection", {"top_candidate_problem": top_problem}
    )
    paradigm = results["paradigm_selection"].get("primary_paradigm", "not_feasible")

    # 7. Experiment design
    bump("Designing experiments (Claude)...")
    allowed_models = ALLOWED_CLASSIFICATION_MODELS
    task_type = "classification"
    if paradigm == "supervised_regression":
        allowed_models, task_type = ALLOWED_REGRESSION_MODELS, "regression"
    elif paradigm == "unsupervised_clustering":
        allowed_models, task_type = ALLOWED_CLUSTERING_MODELS, "clustering"

    target_col = (top_problem or {}).get("target")
    exp_context = {
        "paradigm": paradigm,
        "top_problem": top_problem,
        "ALLOWED_MODELS": allowed_models,
        "columns_available": [c["name"] for c in profile["columns"]],
    }
    results["experiment_design"] = call_agent(client, model, "experiment_design", exp_context)
    experiments_spec = results["experiment_design"].get("experiments", [])

    # 8. Train experiments (deterministic)
    bump("Training candidate models locally (scikit-learn)...")
    exp_results: List[ExperimentResult] = []
    if paradigm in ("supervised_classification", "supervised_regression") and target_col:
        for i, spec in enumerate(experiments_spec[:4]):
            model_name = spec.get("model")
            if model_name not in allowed_models:
                continue
            feats = spec.get("features")
            if feats == "all" or not feats:
                feature_names = [c["name"] for c in profile["columns"] if c["name"] != target_col]
            else:
                # Claude may return column names, or occasionally column-info dicts.
                feature_names = [
                    (f["name"] if isinstance(f, dict) else f) for f in feats
                ]
                feature_names = [f for f in feature_names if f != target_col]
            exp_id = spec.get("experiment_id", f"exp_{i+1}")
            exp_results.append(
                run_supervised_experiment(df, target_col, feature_names, model_name, task_type, exp_id)
            )
    elif paradigm == "unsupervised_clustering":
        feature_names = [c["name"] for c in profile["columns"] if c["inferred_kind"] == "numeric"]
        exp_results.append(run_clustering_experiment(df, feature_names, "exp_1"))
    else:
        st.info(
            f"Paradigm selected was '{paradigm}' - no automatic local training routine "
            "is wired up for this paradigm yet, or no target column was identified. "
            "Reasoning stages below still ran on the profiling data."
        )

    st.session_state.experiment_results = exp_results
    exp_results_serializable = [vars(r) for r in exp_results]

    # 9. Critic
    bump("Critiquing model results (Claude)...")
    results["critic"] = call_agent(
        client, model, "critic", {"experiment_results": exp_results_serializable}
    )

    # 10. Improvement
    bump("Proposing improvements (Claude)...")
    results["improvement"] = call_agent(
        client, model, "improvement", {"critic": results["critic"], "experiment_results": exp_results_serializable}
    )

    # 11. Comparison + explainability + business insight (combined step)
    bump("Comparing models, generating explanations & insights (Claude)...")
    results["comparison"] = call_agent(
        client, model, "comparison", {"experiment_results": exp_results_serializable}
    )
    best_exp = None
    best_id = results["comparison"].get("overall_recommendation")
    for r in exp_results:
        if r.experiment_id == best_id:
            best_exp = vars(r)
    results["explainability"] = call_agent(
        client, model, "explainability", {"best_model": best_exp}
    )
    results["business_insight"] = call_agent(
        client,
        model,
        "business_insight",
        {
            "understanding": results["understanding"],
            "comparison": results["comparison"],
            "explainability": results["explainability"],
        },
    )

    # 12. Reflection + final report
    bump("Writing final report (Claude)...")
    full_context = {
        "understanding": results["understanding"],
        "audit": results["audit"],
        "problem_discovery": results["problem_discovery"],
        "paradigm_selection": results["paradigm_selection"],
        "experiment_results": exp_results_serializable,
        "critic": results["critic"],
        "comparison": results["comparison"],
        "explainability": results["explainability"],
        "business_insight": results["business_insight"],
    }
    results["reflection"] = call_agent(client, model, "reflection", full_context)
    results["final_report"] = call_agent(client, model, "final_report", full_context)

    progress.progress(1.0, text="Done.")
    st.session_state.pipeline_ran = True
    st.session_state.results = results


def render_final_report():
    report = st.session_state.results.get("final_report")
    if not report or report.get("parse_error"):
        return
    st.subheader("📄 Final Report")
    st.markdown(f"**Executive summary:** {report.get('executive_summary', '')}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Dataset understanding:** {report.get('dataset_understanding', '')}")
        st.markdown(f"**Data quality:** {report.get('data_quality', '')}")
        st.markdown(f"**Problem discovered:** {report.get('problem_discovered', '')}")
        st.markdown(f"**Paradigm selected:** {report.get('paradigm_selected', '')}")
    with cols[1]:
        st.markdown(f"**Experiments performed:** {report.get('experiments_performed', '')}")
        st.markdown(f"**Best model:** {report.get('best_model', '')}")
        st.markdown(f"**Overall confidence:** {report.get('overall_confidence', '')}")

    if report.get("key_insights"):
        st.markdown("**Key insights**")
        for item in report["key_insights"]:
            st.markdown(f"- {item}")
    if report.get("limitations"):
        st.markdown("**Limitations**")
        for item in report["limitations"]:
            st.markdown(f"- {item}")
    if report.get("business_recommendations"):
        st.markdown("**Business recommendations**")
        for item in report["business_recommendations"]:
            st.markdown(f"- {item}")
    if report.get("research_recommendations"):
        st.markdown("**Research recommendations**")
        for item in report["research_recommendations"]:
            st.markdown(f"- {item}")
    if report.get("suggested_next_steps"):
        st.markdown("**Suggested next steps**")
        for item in report["suggested_next_steps"]:
            st.markdown(f"- {item}")

    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(st.session_state.results, indent=2, default=str),
        file_name="ai_data_scientist_report.json",
        mime="application/json",
    )


def main():
    st.set_page_config(page_title="Autonomous AI Data Scientist", layout="wide")
    init_state()

    st.title("🔬 Autonomous AI Data Scientist")
    st.caption(
        "Upload -> Understand -> Audit -> Discover Problem -> Select Paradigm -> "
        "Generate Experiments -> Train -> Critique -> Improve -> Compare -> "
        "Explain -> Recommend -> Report"
    )

    api_key, model = sidebar_config()

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return

    if st.session_state.df is None:
        st.info("Upload a CSV file to begin.")
        return

    df = st.session_state.df
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")

    run_clicked = st.button("▶ Run full autonomous pipeline", type="primary")

    if run_clicked:
        if not api_key:
            st.error("Please provide an Anthropic API key in the sidebar first.")
        else:
            try:
                run_pipeline(df, api_key, model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")
                st.code(traceback.format_exc())

    if st.session_state.pipeline_ran:
        render_final_report()

        st.divider()
        st.subheader("🔎 Inspect every stage")
        stage_labels = {
            "understanding": "1. Dataset Understanding",
            "audit": "2. Data Quality Audit",
            "cleaning": "3. Cleaning Strategy",
            "problem_discovery": "4. Problem Discovery",
            "paradigm_selection": "5. Paradigm Selection",
            "experiment_design": "6. Experiment Design",
            "critic": "7. Model Critic",
            "improvement": "8. Model Improvement",
            "comparison": "9. Model Comparison",
            "explainability": "10. Explainability",
            "business_insight": "11. Business Insight",
            "reflection": "12. Self-Reflection",
        }
        for key, label in stage_labels.items():
            with st.expander(label):
                show_json(st.session_state.results.get(key, {}))

        if st.session_state.experiment_results:
            st.subheader("🧪 Real trained-model results (scikit-learn)")
            for r in st.session_state.experiment_results:
                with st.expander(f"{r.experiment_id} — {r.model_name} ({r.task_type})"):
                    if r.error:
                        st.error(r.error)
                    else:
                        st.json(r.metrics)
                        if r.feature_importance:
                            st.markdown("**Top feature importances**")
                            st.bar_chart(pd.Series(r.feature_importance))


if __name__ == "__main__":
    main()
