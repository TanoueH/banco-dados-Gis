"""
Machine Learning MVP para o dataset de suscetibilidade a deslizamentos.

Uso principal:
    python scripts/run_ml_ubatuba_mvp.py --target instability_label

Uso alternativo, multiclasse:
    python scripts/run_ml_ubatuba_mvp.py --target susceptibility_class

O script tenta ler o CSV em outputs/tables/dataset_ml_ubatuba_mvp.csv.
Se o CSV não existir, ele busca os dados diretamente da view PostGIS
vw_ubatuba_susceptibility_points e salva o CSV automaticamente.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DEFAULT_DATASET_PATH = Path("outputs/tables/dataset_ml_ubatuba_mvp.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/ml")

QUERY_DATASET_FROM_POSTGIS = """
SELECT
    slope_code,
    event_code,
    event_date,
    municipality,
    neighborhood,
    elevation_m,
    slope_angle_deg,
    aspect_deg,
    curvature,
    ndvi,
    land_use_class,
    distance_to_drainage_m,
    distance_to_road_m,
    soil_type,
    lithology,
    rainfall_24h_mm,
    rainfall_7d_mm,
    soil_moisture_pct,
    instability_label,
    susceptibility_score,
    susceptibility_class,
    ST_X(geom) AS longitude,
    ST_Y(geom) AS latitude
FROM vw_ubatuba_susceptibility_points
ORDER BY slope_code;
"""


def get_database_url() -> str:
    load_dotenv()

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5434")
    db_name = os.getenv("DB_NAME", "gisdb")
    db_user = os.getenv("DB_USER", "gis_user")
    db_password = os.getenv("DB_PASSWORD", "gis_password")

    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def load_dataset(dataset_path: Path, force_from_db: bool = False) -> pd.DataFrame:
    if dataset_path.exists() and not force_from_db:
        print(f"Loading dataset from CSV: {dataset_path}")
        return pd.read_csv(dataset_path)

    print("Loading dataset from PostGIS...")
    engine = create_engine(get_database_url(), pool_pre_ping=True)

    with engine.connect() as conn:
        df = pd.read_sql(text(QUERY_DATASET_FROM_POSTGIS), conn)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to: {dataset_path}")
    return df


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    df = df.copy()
    df = df.dropna(subset=[target])
    df = df.dropna(axis=1, how="all")

    if target == "instability_label":
        df[target] = df[target].astype(int)

    if len(df) < 20:
        raise ValueError(f"Dataset has only {len(df)} rows. Generate more samples first.")

    return df


def feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    # Não usar susceptibility_score como entrada: ele é a regra que originou o alvo.
    # Usá-lo como feature seria vazamento de informação.
    numeric_candidates = [
        "elevation_m",
        "slope_angle_deg",
        "aspect_deg",
        "curvature",
        "ndvi",
        "distance_to_drainage_m",
        "distance_to_road_m",
        "rainfall_24h_mm",
        "rainfall_7d_mm",
        "soil_moisture_pct",
    ]

    categorical_candidates = [
        "land_use_class",
        "soil_type",
        "lithology",
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns and c != target]
    categorical_features = [c for c in categorical_candidates if c in df.columns and c != target]

    if "soil_moisture_pct" not in numeric_features:
        print("WARNING: soil_moisture_pct não foi encontrado. Rode o ETL corrigido antes do ML.")

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found.")

    return numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale: bool) -> ColumnTransformer:
    if scale:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers)


def make_models(numeric_features: list[str], categorical_features: list[str], random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor(numeric_features, categorical_features, scale=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "decision_tree": Pipeline([
            ("preprocess", make_preprocessor(numeric_features, categorical_features, scale=False)),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor(numeric_features, categorical_features, scale=False)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", make_preprocessor(numeric_features, categorical_features, scale=False)),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]),
    }


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\nTraining: {model_name}")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    average_mode = "binary" if target == "instability_label" else "weighted"

    metrics: dict[str, Any] = {
        "model": model_name,
        "target": target,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average_mode, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average_mode, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average_mode, zero_division=0),
    }

    if target == "instability_label" and hasattr(pipeline, "predict_proba") and y_test.nunique() == 2:
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = output_dir / f"classification_report_{model_name}_{target}.txt"
    report_path.write_text(report, encoding="utf-8")

    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{model_name}_{target}.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    model_path = output_dir / f"model_{model_name}_{target}.joblib"
    joblib.dump(pipeline, model_path)

    metrics["classification_report_path"] = str(report_path)
    metrics["confusion_matrix_path"] = str(cm_path)
    metrics["model_path"] = str(model_path)

    return metrics


def save_feature_importance(model_name: str, pipeline: Pipeline, output_dir: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    df_imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    df_imp = df_imp.sort_values("importance", ascending=False)
    df_imp.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)

    top = df_imp.head(min(15, len(df_imp))).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=300)
    plt.close(fig)


def run_ml(dataset_path: Path, output_dir: Path, target: str, test_size: float, random_state: int, from_db: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, force_from_db=from_db)
    df = clean_dataset(df, target)

    numeric_features, categorical_features = feature_columns(df, target)

    print("\nDataset summary")
    print(f"Rows: {len(df)}")
    print(f"Target: {target}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print("\nTarget distribution:")
    print(df[target].value_counts())

    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models = make_models(numeric_features, categorical_features, random_state)
    results: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        metrics = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target=target,
            output_dir=output_dir,
        )
        results.append(metrics)
        save_feature_importance(model_name, pipeline, output_dir)

    metrics_df = pd.DataFrame(results)
    metrics_path = output_dir / f"metrics_summary_{target}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"All outputs saved in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML MVP para suscetibilidade a deslizamentos.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", choices=["instability_label", "susceptibility_class"], default="instability_label")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--from-db", action="store_true", help="Ler direto do PostGIS e recriar o CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ml(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        from_db=args.from_db,
    )


if __name__ == "__main__":
    main()
