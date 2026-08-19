"""
Autonomous AI Data Science Platform
------------------------------------
An orchestration engine that inspects a dataset, forms a hypothesis about
what kind of problem it represents, runs several competing modelling
experiments, critiques the results, retries with improvements, and explains
its final recommendation in plain English.

Architecture (mirrors the multi-agent design):
    Data Detective        -> understands the dataset
    Problem Formulator     -> decides what kind of problem this is
    Experiment Designer     -> builds competing model configurations
    Model Builder            -> trains & evaluates each configuration
    Model Critic               -> looks for weaknesses, triggers a second round
    Explainability Agent        -> extracts feature importances
    Business Analyst              -> turns results into a plain-English narrative
                                     (rule-based always; LLM-backed if an API key is set)

Run with:
    streamlit run autonomous_ai_data_scientist.py

Dependencies:
    pip install streamlit pandas numpy scikit-learn joblib
Optional (auto-detected, app degrades gracefully if absent):
    pip install xgboost statsmodels anthropic
"""

import io
import os
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
)

import joblib

# --- Optional libraries -----------------------------------------------------
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

def get_anthropic_api_key():
    """Read the API key from Streamlit secrets first, then the environment."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


try:
    import anthropic
    GENAI_AVAILABLE = bool(get_anthropic_api_key())
except ImportError:
    GENAI_AVAILABLE = False


# =============================================================================
# AGENT 1 — DATA DETECTIVE
# =============================================================================
@st.cache_data(show_spinner=False)
def data_detective(df: pd.DataFrame) -> dict:
    """Inspect the raw dataset and return a structured profile."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum()
    missing_pct = (missing / n_rows * 100).round(2) if n_rows else missing

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # datetime detection (declared dtype OR object columns that parse cleanly)
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    for col in categorical_cols[:]:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.9:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    duplicates = int(df.duplicated().sum())

    # crude outlier count (IQR method) per numeric column
    outliers = {}
    for c in numeric_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers[c] = int(((df[c] < lo) | (df[c] > hi)).sum())

    corr = df[numeric_cols].corr(numeric_only=True) if len(numeric_cols) > 1 else pd.DataFrame()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": dtypes,
        "missing": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": list(dict.fromkeys(datetime_cols)),
        "cardinality": cardinality,
        "duplicates": duplicates,
        "outliers": outliers,
        "correlation": corr,
    }


# =============================================================================
# AGENT 2 — PROBLEM FORMULATOR
# =============================================================================
def formulate_problem(df: pd.DataFrame, profile: dict, target_col: str | None) -> dict:
    """
    Decide what kind of analytical problem this dataset represents.
    Returns a dict describing the recommended paradigm + reasoning + confidence.
    """
    n_rows = profile["n_rows"]
    n_cols = profile["n_cols"]

    # --- Reject modelling outright if the data can't support it -----------
    if n_rows < 30:
        return {
            "paradigm": "insufficient_data",
            "reasoning": (
                f"Only {n_rows} rows are available. This is too small a sample for "
                "reliable machine learning. A descriptive/statistical summary is "
                "recommended instead of predictive modelling."
            ),
            "confidence": 0.95,
        }

    avg_missing = np.mean(list(profile["missing_pct"].values())) if profile["missing_pct"] else 0
    if avg_missing > 50:
        return {
            "paradigm": "insufficient_data",
            "reasoning": (
                f"Average missingness across columns is {avg_missing:.1f}%. Modelling "
                "on data this incomplete risks unreliable, misleading results. Address "
                "data collection gaps before modelling."
            ),
            "confidence": 0.85,
        }

    if n_cols > n_rows:
        return {
            "paradigm": "insufficient_data",
            "reasoning": (
                f"The dataset has more columns ({n_cols}) than rows ({n_rows}). This "
                "high-dimensional, low-sample-size setting is prone to severe overfitting. "
                "Dimensionality reduction or additional data collection is recommended "
                "before predictive modelling is attempted."
            ),
            "confidence": 0.8,
        }

    # --- No target selected -> unsupervised territory ----------------------
    if target_col is None or target_col == "(none — explore structure)":
        # time-series signal
        if profile["datetime_cols"] and profile["numeric_cols"]:
            return {
                "paradigm": "time_series",
                "reasoning": (
                    f"A datetime-like column ({profile['datetime_cols'][0]}) was detected "
                    "alongside numeric variables, suggesting a time-ordered forecasting "
                    "problem rather than i.i.d. tabular data."
                ),
                "confidence": 0.6,
            }
        return {
            "paradigm": "unsupervised",
            "reasoning": (
                "No target variable was specified. The strongest analytical opportunity "
                "with unlabeled tabular data is typically clustering (to discover natural "
                "groupings) or anomaly detection (to flag unusual records)."
            ),
            "confidence": 0.7,
        }

    # --- Target selected -> supervised: classification vs regression -------
    target = df[target_col]
    if target.dropna().empty:
        return {
            "paradigm": "insufficient_data",
            "reasoning": f"Target column '{target_col}' contains no non-missing values.",
            "confidence": 0.99,
        }
    n_unique = target.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(target)

    if not is_numeric or (is_numeric and n_unique <= max(20, int(0.05 * n_rows))):
        # looks categorical
        class_counts = target.value_counts(normalize=True)
        imbalance_ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1
        reasoning = (
            f"Target column '{target_col}' has {n_unique} distinct value(s), consistent "
            "with a classification problem."
        )
        if imbalance_ratio > 3:
            reasoning += f" Class imbalance detected (ratio ≈ {imbalance_ratio:.1f}:1) — this will inform model selection."
        return {
            "paradigm": "classification",
            "reasoning": reasoning,
            "confidence": 0.9 if n_unique <= 10 else 0.65,
            "imbalance_ratio": float(imbalance_ratio),
        }
    else:
        return {
            "paradigm": "regression",
            "reasoning": (
                f"Target column '{target_col}' is continuous numeric with {n_unique} distinct "
                "values, consistent with a regression problem."
            ),
            "confidence": 0.85,
        }


# =============================================================================
# Shared preprocessing pipeline builder
# =============================================================================
def build_preprocessor(df: pd.DataFrame, feature_cols: list):
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return preprocessor


# =============================================================================
# AGENT 3 + 4 — EXPERIMENT DESIGNER + MODEL BUILDER (classification)
# =============================================================================
def design_classification_experiments(class_weight=None):
    experiments = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weight),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight=class_weight, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(probability=True, class_weight=class_weight),
        "k-NN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }
    if XGBOOST_AVAILABLE:
        experiments["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=42,
            scale_pos_weight=1 if class_weight is None else None,
        )
    return experiments


def run_classification(df, target_col, feature_cols, class_weight=None):
    X, y = df[feature_cols], df[target_col]
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes with non-missing values.")

    preprocessor = build_preprocessor(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    results = []
    fitted_pipelines = {}
    for name, model in design_classification_experiments(class_weight).items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = None
            if hasattr(pipe.named_steps["model"], "predict_proba") and y.nunique() == 2:
                try:
                    proba = pipe.predict_proba(X_test)[:, 1]
                except Exception:
                    proba = None

            metrics = {
                "Experiment": name,
                "Accuracy": round(accuracy_score(y_test, preds), 4),
                "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
                "Recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
                "F1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
            }
            if proba is not None:
                try:
                    metrics["ROC-AUC"] = round(roc_auc_score(y_test, proba), 4)
                except Exception:
                    metrics["ROC-AUC"] = None
            results.append(metrics)
            fitted_pipelines[name] = pipe
        except Exception as e:
            results.append({"Experiment": name, "Accuracy": None, "Error": str(e)})

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False, na_position="last")
    return results_df, fitted_pipelines, (X_test, y_test)


# =============================================================================
# AGENT 3 + 4 — EXPERIMENT DESIGNER + MODEL BUILDER (regression)
# =============================================================================
def design_regression_experiments():
    experiments = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Elastic Net": ElasticNet(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
    }
    if XGBOOST_AVAILABLE:
        experiments["XGBoost"] = XGBRegressor(random_state=42)
    return experiments


def run_regression(df, target_col, feature_cols):
    X, y = df[feature_cols], df[target_col]
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    if len(y) < 10:
        raise ValueError("Regression requires at least 10 non-missing target observations.")

    preprocessor = build_preprocessor(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    results = []
    fitted_pipelines = {}
    for name, model in design_regression_experiments().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            metrics = {
                "Experiment": name,
                "MAE": round(mean_absolute_error(y_test, preds), 4),
                "RMSE": round(mean_squared_error(y_test, preds) ** 0.5, 4),
                "R2": round(r2_score(y_test, preds), 4),
            }
            results.append(metrics)
            fitted_pipelines[name] = pipe
        except Exception as e:
            results.append({"Experiment": name, "MAE": None, "Error": str(e)})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False, na_position="last")
    return results_df, fitted_pipelines, (X_test, y_test)


# =============================================================================
# AGENT 3 + 4 — EXPERIMENT DESIGNER + MODEL BUILDER (unsupervised)
# =============================================================================
def run_unsupervised(df, feature_cols):
    X = df[feature_cols]
    preprocessor = build_preprocessor(df, feature_cols)
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    results = []
    labels_by_experiment = {}

    # Clustering experiments across a small k grid for KMeans
    for k in (2, 3, 4, 5):
        try:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X_processed)
            if len(set(labels)) > 1:
                score = silhouette_score(X_processed, labels)
                results.append({"Experiment": f"KMeans (k={k})", "Method": "Clustering",
                                 "Silhouette Score": round(score, 4), "Clusters found": len(set(labels))})
                labels_by_experiment[f"KMeans (k={k})"] = labels
        except Exception as e:
            results.append({"Experiment": f"KMeans (k={k})", "Method": "Clustering", "Error": str(e)})

    try:
        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X_processed)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(X_processed, labels)
            results.append({"Experiment": "DBSCAN", "Method": "Clustering",
                             "Silhouette Score": round(score, 4), "Clusters found": n_clusters})
            labels_by_experiment["DBSCAN"] = labels
    except Exception as e:
        results.append({"Experiment": "DBSCAN", "Method": "Clustering", "Error": str(e)})

    try:
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        results.append({"Experiment": "Agglomerative (k=3)", "Method": "Clustering",
                         "Silhouette Score": round(score, 4), "Clusters found": 3})
        labels_by_experiment["Agglomerative (k=3)"] = labels
    except Exception as e:
        results.append({"Experiment": "Agglomerative (k=3)", "Method": "Clustering", "Error": str(e)})

    try:
        model = GaussianMixture(n_components=3, random_state=42)
        labels = model.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        results.append({"Experiment": "Gaussian Mixture (k=3)", "Method": "Clustering",
                         "Silhouette Score": round(score, 4), "Clusters found": 3})
        labels_by_experiment["Gaussian Mixture (k=3)"] = labels
    except Exception as e:
        results.append({"Experiment": "Gaussian Mixture (k=3)", "Method": "Clustering", "Error": str(e)})

    # Anomaly detection experiments
    for name, model in {
        "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.05),
    }.items():
        try:
            preds = model.fit_predict(X_processed)
            n_anomalies = int((preds == -1).sum())
            results.append({"Experiment": name, "Method": "Anomaly Detection",
                             "Anomalies flagged": n_anomalies,
                             "Anomaly rate %": round(100 * n_anomalies / len(preds), 2)})
            labels_by_experiment[name] = preds
        except Exception as e:
            results.append({"Experiment": name, "Method": "Anomaly Detection", "Error": str(e)})

    try:
        lof = LocalOutlierFactor(n_neighbors=20)
        preds = lof.fit_predict(X_processed)
        n_anomalies = int((preds == -1).sum())
        results.append({"Experiment": "Local Outlier Factor", "Method": "Anomaly Detection",
                         "Anomalies flagged": n_anomalies,
                         "Anomaly rate %": round(100 * n_anomalies / len(preds), 2)})
        labels_by_experiment["Local Outlier Factor"] = preds
    except Exception as e:
        results.append({"Experiment": "Local Outlier Factor", "Method": "Anomaly Detection", "Error": str(e)})

    # Dimensionality reduction, reported separately
    dim_reduction_note = None
    try:
        n_components = min(5, X_processed.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_processed)
        explained = pca.explained_variance_ratio_.cumsum()
        dim_reduction_note = (
            f"PCA: the first {n_components} component(s) explain "
            f"{explained[-1] * 100:.1f}% of total variance."
        )
    except Exception:
        pass

    results_df = pd.DataFrame(results)
    return results_df, labels_by_experiment, dim_reduction_note


# =============================================================================
# TIME SERIES (lightweight, honest-about-limits implementation)
# =============================================================================
def run_time_series(df, date_col, value_col, forecast_periods=10):
    ts_df = df[[date_col, value_col]].dropna().copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.dropna().sort_values(date_col)
    series = ts_df.set_index(date_col)[value_col]

    if len(series) < 15:
        return None, "Not enough time-ordered observations (need at least 15) for a meaningful forecast."

    split_point = int(len(series) * 0.85)
    train, test = series.iloc[:split_point], series.iloc[split_point:]

    results = []

    # Naive baseline: last value carried forward
    naive_preds = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    results.append({
        "Experiment": "Naive (last value)",
        "MAE": round(mean_absolute_error(test, naive_preds), 4),
    })

    # Moving-average baseline
    window = max(2, min(7, len(train) // 4))
    ma_value = train.rolling(window).mean().iloc[-1]
    ma_preds = pd.Series([ma_value] * len(test), index=test.index)
    results.append({
        "Experiment": f"Moving Average (window={window})",
        "MAE": round(mean_absolute_error(test, ma_preds), 4),
    })

    forecast_series = None
    if STATSMODELS_AVAILABLE:
        try:
            model = ARIMA(train, order=(1, 1, 1)).fit()
            arima_preds = model.forecast(steps=len(test))
            results.append({
                "Experiment": "ARIMA(1,1,1)",
                "MAE": round(mean_absolute_error(test, arima_preds), 4),
            })
            full_model = ARIMA(series, order=(1, 1, 1)).fit()
            forecast_series = full_model.forecast(steps=forecast_periods)
        except Exception as e:
            results.append({"Experiment": "ARIMA(1,1,1)", "Error": str(e)})
    else:
        results.append({
            "Experiment": "ARIMA(1,1,1)",
            "Note": "statsmodels not installed — install it to enable ARIMA forecasting.",
        })

    results_df = pd.DataFrame(results)
    return {"results": results_df, "series": series, "forecast": forecast_series}, None


# =============================================================================
# AGENT 6 — MODEL CRITIC (one round of criticize -> retry)
# =============================================================================
def model_critic_classification(results_df, imbalance_ratio):
    """If imbalance is significant and results are mediocre, trigger a
    second round of experiments with class_weight='balanced'."""
    critique = []
    retry = False
    best_f1 = results_df["F1"].max() if "F1" in results_df and not results_df["F1"].isna().all() else 0

    if imbalance_ratio > 3:
        critique.append(
            f"Class imbalance ratio of {imbalance_ratio:.1f}:1 detected. Weighted F1 can mask "
            "poor minority-class recall. Retrying key models with class_weight='balanced'."
        )
        retry = True
    if best_f1 < 0.6:
        critique.append(
            f"Best F1 score ({best_f1:.2f}) is modest. This may indicate the features have limited "
            "predictive signal for this target, or that further feature engineering is needed."
        )
    if not critique:
        critique.append("No major weaknesses detected in this round of experiments.")
    return critique, retry


# =============================================================================
# AGENT 7 — EXPLAINABILITY AGENT
# =============================================================================
def extract_feature_importance(pipe, feature_cols, df):
    """Best-effort feature importance extraction across model types."""
    try:
        model = pipe.named_steps["model"]
        preprocessor = pipe.named_steps["prep"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(15)
        return imp_df
    except Exception:
        return None


# =============================================================================
# AGENT 8 — BUSINESS ANALYST (rule-based, always available)
# =============================================================================
def rule_based_narrative(profile, problem, results_df, paradigm, critique=None):
    lines = []
    lines.append(f"The dataset contains **{profile['n_rows']} records** and **{profile['n_cols']} variables**.")
    lines.append(f"**Diagnosis:** {problem['reasoning']} (confidence ≈ {problem['confidence']*100:.0f}%).")

    if paradigm in ("classification", "regression") and not results_df.empty:
        top = results_df.iloc[0]
        metric_name = "F1" if paradigm == "classification" else "R2"
        if metric_name in top and pd.notna(top[metric_name]):
            lines.append(
                f"**Recommended model:** {top['Experiment']} achieved the strongest "
                f"{metric_name} score ({top[metric_name]}) among {len(results_df)} experiments."
            )
    elif paradigm == "unsupervised" and not results_df.empty:
        cluster_rows = results_df[results_df.get("Method") == "Clustering"] if "Method" in results_df else pd.DataFrame()
        if not cluster_rows.empty and "Silhouette Score" in cluster_rows:
            best = cluster_rows.sort_values("Silhouette Score", ascending=False).iloc[0]
            lines.append(
                f"**Recommended clustering:** {best['Experiment']} produced the most "
                f"well-separated groups (silhouette score {best['Silhouette Score']})."
            )

    if critique:
        lines.append("**Model critique:** " + " ".join(critique))

    return lines


def generate_ai_narrative(profile, problem, results_df, paradigm):
    if not GENAI_AVAILABLE:
        return None
    client = anthropic.Anthropic(api_key=get_anthropic_api_key())
    summary = textwrap.dedent(f"""
        Dataset shape: {profile['n_rows']} rows x {profile['n_cols']} columns
        Detected paradigm: {paradigm}
        Reasoning: {problem['reasoning']}
        Experiment results:
        {results_df.to_string(index=False)}
    """)
    prompt = (
        "You are a senior data scientist presenting results to a business stakeholder. "
        "Based on this experiment summary, write a concise (5-6 bullet points) business-facing "
        "interpretation: what was found, which model/approach is recommended and why, and one "
        "concrete next step. Avoid jargon where possible.\n\n" + summary
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title="Autonomous AI Data Science Platform", layout="wide")
    st.title("🧠 Autonomous AI Data Science Platform")
    st.caption(
        "Upload a dataset. The system inspects it, decides what kind of problem it is, "
        "runs competing model experiments, critiques itself, and explains the result."
    )

    if GENAI_AVAILABLE:
        st.sidebar.success("AI narrative layer: enabled")
    else:
        st.sidebar.info("AI narrative layer: disabled\n\nSet ANTHROPIC_API_KEY to enable LLM-written summaries.")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="latin-1")
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The uploaded CSV contains no rows.")
        st.stop()

    st.subheader("① Raw Data")
    st.dataframe(df.head(15), use_container_width=True)

    # --- Agent 1: Data Detective -------------------------------------------
    st.subheader("② AI Data Understanding")
    profile = data_detective(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", profile["n_rows"])
    c2.metric("Columns", profile["n_cols"])
    c3.metric("Duplicate rows", profile["duplicates"])
    c4.metric("Datetime columns", len(profile["datetime_cols"]))

    with st.expander("Full data profile"):
        st.write("**Missing % by column:**", profile["missing_pct"])
        st.write("**Cardinality by column:**", profile["cardinality"])
        st.write("**Outlier count (IQR method):**", profile["outliers"])
        if not profile["correlation"].empty:
            st.dataframe(profile["correlation"].style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))

    # --- Target selection (user supplies business context) -----------------
    st.subheader("③ Problem Formulation")
    target_options = ["(none — explore structure)"] + df.columns.tolist()
    target_col = st.selectbox(
        "Select a target variable if you have one in mind (or let the system explore unsupervised):",
        target_options,
    )
    target_col = None if target_col == "(none — explore structure)" else target_col

    problem = formulate_problem(df, profile, target_col)
    paradigm = problem["paradigm"]

    st.markdown(f"**Detected paradigm:** `{paradigm}`")
    st.markdown(f"**Reasoning:** {problem['reasoning']}")
    st.progress(min(1.0, problem["confidence"]))

    if paradigm == "insufficient_data":
        st.warning(
            "The AI recommends **against** predictive modelling on this dataset in its current "
            "state. This is an intentional design choice — a credible system should be able to "
            "say 'no' rather than force a bad model. Consider a descriptive/statistical summary instead."
        )
        return

    feature_cols = [c for c in df.columns if c != target_col and c not in profile["datetime_cols"]]

    # --- Agent 3/4: Experiment Designer + Model Builder ---------------------
    st.subheader("④ Modelling Laboratory")

    if paradigm == "classification":
        with st.spinner("Running competing classification experiments..."):
            results_df, pipelines, test_data = run_classification(df, target_col, feature_cols)
        st.dataframe(results_df, use_container_width=True)

        imbalance_ratio = problem.get("imbalance_ratio", 1.0)
        critique, retry = model_critic_classification(results_df, imbalance_ratio)
        st.markdown("**🔎 Model Critic:**")
        for c in critique:
            st.write("- " + c)

        if retry:
            st.markdown("**Round 2 — retrying with `class_weight='balanced'`:**")
            with st.spinner("Retraining with class balancing..."):
                results_df2, pipelines2, _ = run_classification(
                    df, target_col, feature_cols, class_weight="balanced"
                )
            st.dataframe(results_df2, use_container_width=True)
            if results_df2["F1"].max() > results_df["F1"].max():
                st.success("Round 2 improved the best F1 score — using the balanced models going forward.")
                results_df, pipelines = results_df2, pipelines2
            else:
                st.info("Round 2 did not improve results — keeping round 1 models.")

        best_name = results_df.iloc[0]["Experiment"]
        best_pipe = pipelines.get(best_name)

        st.subheader("⑤ Explainability")
        if best_pipe is not None:
            imp_df = extract_feature_importance(best_pipe, feature_cols, df)
            if imp_df is not None:
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.write(f"Feature importance is not directly available for {best_name}.")

    elif paradigm == "regression":
        with st.spinner("Running competing regression experiments..."):
            results_df, pipelines, test_data = run_regression(df, target_col, feature_cols)
        st.dataframe(results_df, use_container_width=True)

        best_r2 = results_df["R2"].max() if "R2" in results_df else None
        critique = []
        if best_r2 is not None and best_r2 < 0.4:
            critique.append(
                f"Best R² ({best_r2:.2f}) is low — the available features explain relatively "
                "little of the variance in the target. Consider additional features or a "
                "different target formulation."
            )
        else:
            critique.append("No major weaknesses detected in this round of experiments.")
        st.markdown("**🔎 Model Critic:**")
        for c in critique:
            st.write("- " + c)

        best_name = results_df.iloc[0]["Experiment"]
        best_pipe = pipelines.get(best_name)

        st.subheader("⑤ Explainability")
        if best_pipe is not None:
            imp_df = extract_feature_importance(best_pipe, feature_cols, df)
            if imp_df is not None:
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.write(f"Feature importance is not directly available for {best_name}.")

    elif paradigm == "unsupervised":
        with st.spinner("Running clustering and anomaly detection experiments..."):
            results_df, labels_by_experiment, dim_note = run_unsupervised(df, feature_cols)
        st.dataframe(results_df, use_container_width=True)
        if dim_note:
            st.info(dim_note)
        critique = ["Unsupervised results were evaluated using silhouette score (clustering) "
                    "and contamination-based anomaly rate (anomaly detection). No ground truth "
                    "exists, so results should be validated against business judgment."]
        st.markdown("**🔎 Model Critic:**")
        for c in critique:
            st.write("- " + c)
        best_pipe = None

    elif paradigm == "time_series":
        date_col = profile["datetime_cols"][0]
        if not profile["numeric_cols"]:
            st.warning("A datetime column was detected, but there are no numeric columns available to forecast.")
            st.stop()
        value_col = st.selectbox("Select the numeric value to forecast:", profile["numeric_cols"])
        ts_result, err = run_time_series(df, date_col, value_col)
        if err:
            st.warning(err)
            return
        results_df = ts_result["results"]
        st.dataframe(results_df, use_container_width=True)
        st.line_chart(ts_result["series"])
        if ts_result["forecast"] is not None:
            st.markdown("**Forecast (next periods):**")
            st.write(ts_result["forecast"])
        critique = ["Only lightweight baselines and a single ARIMA configuration were tried. "
                    "A production system would also evaluate seasonal models, Prophet, and "
                    "sequence models (LSTM/Transformer) before finalizing a forecasting approach."]
        st.markdown("**🔎 Model Critic:**")
        for c in critique:
            st.write("- " + c)
        best_pipe = None

    # --- Agent 8: Business Analyst narrative --------------------------------
    st.subheader("⑥ AI Recommendation & Narrative")
    narrative_lines = rule_based_narrative(profile, problem, results_df, paradigm, critique)
    for line in narrative_lines:
        st.markdown("- " + line)

    if GENAI_AVAILABLE:
        if st.button("Generate AI-written business narrative"):
            with st.spinner("Consulting the language model..."):
                ai_text = generate_ai_narrative(profile, problem, results_df, paradigm)
            st.markdown(ai_text)
    else:
        st.caption("Set ANTHROPIC_API_KEY to also get an LLM-written narrative here.")

    # --- Downloads -----------------------------------------------------------
    st.subheader("⑦ Downloads")
    colA, colB = st.columns(2)
    with colA:
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download experiment report (CSV)", csv_buf.getvalue(),
                            file_name="experiment_report.csv", mime="text/csv")
    with colB:
        if paradigm in ("classification", "regression") and 'best_pipe' in dir() and best_pipe is not None:
            model_buf = io.BytesIO()
            joblib.dump(best_pipe, model_buf)
            st.download_button("⬇️ Download best model (.joblib)", model_buf.getvalue(),
                                file_name="best_model.joblib")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Deployment-ready: API keys can be supplied through Streamlit Secrets "
        "or environment variables. No API key is embedded in the source code."
    )


if __name__ == "__main__":
    main()
