import os
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


def get_database_url() -> str:
    load_dotenv()

    db_host = os.getenv("DB_HOST", "postgis_db")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "gisdb")
    db_user = os.getenv("DB_USER", "gis_user")
    db_password = os.getenv("DB_PASSWORD", "gis_password")

    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def wait_for_database(engine, retries: int = 30, delay_s: int = 2) -> None:
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Connection to PostGIS established successfully.")
            return
        except OperationalError:
            print(f"Waiting for PostGIS... attempt {attempt}/{retries}")
            time.sleep(delay_s)

    raise RuntimeError("Could not connect to PostGIS after several attempts.")


def insert_demo_features(conn) -> None:
    existing = conn.execute(
        text("SELECT COUNT(*) FROM slope_features")
    ).scalar_one()

    if existing > 0:
        print("Demonstration slope features already exist. Skipping insertion.")
        return

    print("Inserting demonstration GIS/ML features...")

    rows = [
        {
            "slope_code": "SLOPE_001",
            "elevation_m": 690.0,
            "slope_angle_deg": 12.5,
            "aspect_deg": 80.0,
            "curvature": 0.021,
            "ndvi": 0.62,
            "land_use_class": "vegetation",
            "distance_to_drainage_m": 180.0,
            "distance_to_road_m": 250.0,
            "soil_type": "residual_soil",
            "lithology": "weathered_rock",
            "rainfall_24h_mm": 18.0,
            "rainfall_7d_mm": 65.0,
        },
        {
            "slope_code": "SLOPE_002",
            "elevation_m": 710.0,
            "slope_angle_deg": 24.8,
            "aspect_deg": 135.0,
            "curvature": -0.014,
            "ndvi": 0.38,
            "land_use_class": "sparse_vegetation",
            "distance_to_drainage_m": 95.0,
            "distance_to_road_m": 80.0,
            "soil_type": "colluvial_soil",
            "lithology": "sedimentary",
            "rainfall_24h_mm": 32.0,
            "rainfall_7d_mm": 110.0,
        },
        {
            "slope_code": "SLOPE_003",
            "elevation_m": 735.0,
            "slope_angle_deg": 38.2,
            "aspect_deg": 210.0,
            "curvature": -0.087,
            "ndvi": 0.21,
            "land_use_class": "exposed_soil",
            "distance_to_drainage_m": 35.0,
            "distance_to_road_m": 22.0,
            "soil_type": "clayey_soil",
            "lithology": "weathered_rock",
            "rainfall_24h_mm": 58.0,
            "rainfall_7d_mm": 185.0,
        },
    ]

    for row in rows:
        slope_id = conn.execute(
            text("""
                SELECT id
                FROM slope_units
                WHERE slope_code = :slope_code
                LIMIT 1
            """),
            {"slope_code": row["slope_code"]},
        ).scalar_one()

        conn.execute(
            text("""
                INSERT INTO slope_features (
                    slope_id,
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
                    rainfall_7d_mm
                )
                VALUES (
                    :slope_id,
                    :elevation_m,
                    :slope_angle_deg,
                    :aspect_deg,
                    :curvature,
                    :ndvi,
                    :land_use_class,
                    :distance_to_drainage_m,
                    :distance_to_road_m,
                    :soil_type,
                    :lithology,
                    :rainfall_24h_mm,
                    :rainfall_7d_mm
                )
            """),
            {**row, "slope_id": slope_id},
        )

    print("Demonstration GIS/ML features inserted successfully.")


def insert_demo_training_samples(conn) -> None:
    existing = conn.execute(
        text("SELECT COUNT(*) FROM training_samples")
    ).scalar_one()

    if existing > 0:
        print("Training samples already exist. Skipping insertion.")
        return

    print("Inserting demonstration training samples...")

    labels = {
        "SLOPE_001": 0,
        "SLOPE_002": 0,
        "SLOPE_003": 1,
    }

    for slope_code, label in labels.items():
        slope_id = conn.execute(
            text("""
                SELECT id
                FROM slope_units
                WHERE slope_code = :slope_code
                LIMIT 1
            """),
            {"slope_code": slope_code},
        ).scalar_one()

        conn.execute(
            text("""
                INSERT INTO training_samples (
                    slope_id,
                    instability_label,
                    label_source,
                    confidence_level,
                    notes
                )
                VALUES (
                    :slope_id,
                    :instability_label,
                    :label_source,
                    :confidence_level,
                    :notes
                )
            """),
            {
                "slope_id": slope_id,
                "instability_label": label,
                "label_source": "demonstration_data",
                "confidence_level": "medium",
                "notes": "Demonstration label for MVP validation.",
            },
        )

    print("Demonstration training samples inserted successfully.")


def insert_demo_susceptibility_results(conn) -> None:
    existing = conn.execute(
        text("SELECT COUNT(*) FROM susceptibility_results")
    ).scalar_one()

    if existing > 0:
        print("Susceptibility results already exist. Skipping insertion.")
        return

    print("Inserting demonstration susceptibility results...")

    results = {
        "SLOPE_001": (0.18, "Low"),
        "SLOPE_002": (0.54, "Medium"),
        "SLOPE_003": (0.87, "High"),
    }

    for slope_code, (score, susceptibility_class) in results.items():
        slope_id = conn.execute(
            text("""
                SELECT id
                FROM slope_units
                WHERE slope_code = :slope_code
                LIMIT 1
            """),
            {"slope_code": slope_code},
        ).scalar_one()

        conn.execute(
            text("""
                INSERT INTO susceptibility_results (
                    slope_id,
                    model_name,
                    susceptibility_score,
                    susceptibility_class,
                    probability_instability,
                    run_id
                )
                VALUES (
                    :slope_id,
                    :model_name,
                    :susceptibility_score,
                    :susceptibility_class,
                    :probability_instability,
                    :run_id
                )
            """),
            {
                "slope_id": slope_id,
                "model_name": "demo_rule_based_model",
                "susceptibility_score": score,
                "susceptibility_class": susceptibility_class,
                "probability_instability": score,
                "run_id": "demo_run_001",
            },
        )

    conn.execute(
        text("""
            INSERT INTO ml_runs (
                run_id,
                model_name,
                algorithm,
                accuracy,
                precision_score,
                recall_score,
                f1_score,
                roc_auc,
                notes
            )
            VALUES (
                'demo_run_001',
                'demo_rule_based_model',
                'rule_based_demo',
                0.850000,
                0.800000,
                0.750000,
                0.774000,
                0.880000,
                'Demonstration ML run for MVP validation.'
            )
            ON CONFLICT (run_id) DO NOTHING
        """)
    )

    print("Demonstration susceptibility results inserted successfully.")


def print_database_summary(conn) -> None:
    tables = [
        "projects",
        "study_areas",
        "slope_units",
        "geospatial_layers",
        "slope_features",
        "training_samples",
        "susceptibility_results",
        "ml_runs",
    ]

    print("\nDatabase summary:")

    for table in tables:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        print(f"- {table}: {count} record(s)")

    print("\nSpatial check:")

    result = conn.execute(
        text("""
            SELECT
                slope_code,
                ST_AsText(geom) AS geometry_wkt
            FROM slope_units
            ORDER BY slope_code
        """)
    ).fetchall()

    for row in result:
        print(f"- {row.slope_code}: {row.geometry_wkt}")


def main() -> None:
    print("Starting GIS ETL for slope instability susceptibility project...")

    database_url = get_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    wait_for_database(engine)

    with engine.begin() as conn:
        insert_demo_features(conn)
        insert_demo_training_samples(conn)
        insert_demo_susceptibility_results(conn)
        print_database_summary(conn)

    print("\nGIS ETL finished successfully.")


if __name__ == "__main__":
    main()