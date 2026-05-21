"""
ETL geoespacial para dataset MVP de suscetibilidade a deslizamentos.

Estratégia metodológica do MVP:
- Trabalhar com um único evento/coroa de ruptura para reduzir ambiguidade.
- Gerar pontos amostrais em anéis concêntricos ao redor da coroa.
- Usar buffers de 5 m, 15 m, 25 m e 50 m.
- Incorporar a variável soil_moisture_pct ao dataset.
- Produzir uma view consolidada para ArcGIS/QGIS e exportação para Machine Learning.

Arquivo sugerido:
    scripts/etl_ubatuba_event.py

Execução:
    python scripts/etl_ubatuba_event.py

Observação metodológica:
Este dataset é experimental e heurístico. Ele não substitui inventário real de
movimentos de massa nem medições de campo. Sua função é demonstrar a
infraestrutura Python + PostGIS + SIG + preparação de dataset para Machine Learning.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


# -----------------------------------------------------------------------------
# 1. Configuração do evento real
# -----------------------------------------------------------------------------

EVENT_CODE = "UBATUBA_2023_02_19_PEREQUE_ACU"
EVENT_DATE = "2023-02-19"
MUNICIPALITY = "Ubatuba"
STATE = "SP"
NEIGHBORHOOD = "Perequê-Açu"
ADDRESS_REFERENCE = "Coroa da ruptura georreferenciada"
EVENT_TYPE = "deslizamento/rolamento de bloco"
SOURCE_NAME = "Corpo de Bombeiros / CNN Brasil / Agência Brasil / imprensa local"
SOURCE_URL = "https://www.cnnbrasil.com.br/nacional/crianca-de-7-anos-morre-em-ubatuba-apos-deslizamento-de-pedra-causado-pela-chuva/"
DAMAGE_DESCRIPTION = (
    "Deslizamento/rolamento de bloco associado ao evento extremo de chuvas "
    "entre 18 e 19 de fevereiro de 2023 no Litoral Norte de São Paulo."
)
CONFIDENCE_LEVEL = "high"

# Coordenada da coroa da ruptura utilizada como ponto central da análise.
# Ordem correta em PostGIS: longitude, latitude.
# Sistema: WGS 84 / EPSG:4326.
EVENT_LON = -45.665458
EVENT_LAT = -23.784755

# Chuva acumulada de referência do evento.
# Mantida como NULL até confirmação em base pluviométrica oficial específica.
RAINFALL_72H_MM = None

# Sistema projetado em metros para Ubatuba/SP.
# SIRGAS 2000 / UTM zone 23S.
PROJECTED_SRID = 31983
GEOGRAPHIC_SRID = 4326

RUN_ID = "ubatuba_2023_mvp_ml_dataset_001"
MODEL_NAME = "ubatuba_rule_based_susceptibility_mvp"


# -----------------------------------------------------------------------------
# 2. Estratégia de amostragem do MVP
# -----------------------------------------------------------------------------

# Buffers e anéis de geração de pontos.
# A geometria dos buffers é criada a partir da coroa da ruptura.
BUFFER_DISTANCES_M = (5, 15, 25, 50)

# Cada anel gera N pontos ao redor da coroa.
# 4 anéis x 20 pontos = 80 pontos amostrais.
N_POINTS_PER_RING = 20


@dataclass(frozen=True)
class AnalysisPoint:
    slope_code: str
    center_code: str
    center_name: str
    center_lon: float
    center_lat: float
    ring_m: float
    angle_deg: float
    dx_m: float
    dy_m: float
    distance_to_center_m: float
    elevation_m: float
    slope_angle_deg: float
    aspect_deg: float
    curvature: float
    ndvi: float
    land_use_class: str
    distance_to_drainage_m: float
    distance_to_road_m: float
    soil_type: str
    lithology: str
    rainfall_24h_mm: float
    rainfall_7d_mm: float
    soil_moisture_pct: float
    instability_label: int
    susceptibility_score: float
    susceptibility_class: str


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def classify_susceptibility(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def base_score_by_ring(ring_m: float) -> float:
    """Define criticidade média por anel.

    A lógica do MVP assume maior criticidade no entorno imediato da coroa
    e redução progressiva com o afastamento espacial.
    """
    if ring_m <= 5:
        return 0.86
    if ring_m <= 15:
        return 0.68
    if ring_m <= 25:
        return 0.50
    return 0.30


def generate_analysis_points() -> list[AnalysisPoint]:
    """Gera dataset amostral em anéis concêntricos ao redor da coroa.

    A variação angular evita que todos os pontos de um mesmo anel tenham valores
    idênticos. A geração é determinística, isto é, o resultado se repete a cada execução.
    """
    points: list[AnalysisPoint] = []

    center_code = "COROA"
    center_name = "Coroa da ruptura"

    point_global_id = 1

    for ring_m in BUFFER_DISTANCES_M:
        base_score = base_score_by_ring(ring_m)

        for point_index in range(N_POINTS_PER_RING):
            angle_deg = point_index * (360.0 / N_POINTS_PER_RING)
            angle_rad = math.radians(angle_deg)

            dx_m = round(ring_m * math.cos(angle_rad), 3)
            dy_m = round(ring_m * math.sin(angle_rad), 3)
            distance_to_center_m = round((dx_m ** 2 + dy_m ** 2) ** 0.5, 3)

            # Variação determinística por direção para simular heterogeneidade local.
            angular_variation = 0.06 * math.sin(angle_rad) + 0.04 * math.cos(2 * angle_rad)
            score = round(clamp(base_score + angular_variation, 0.10, 0.95), 3)

            susceptibility_class = classify_susceptibility(score)
            instability_label = 1 if score >= 0.60 else 0

            land_use_class = (
                "exposed_soil"
                if score >= 0.75
                else "sparse_vegetation"
                if score >= 0.45
                else "vegetation"
            )

            soil_type = (
                "clayey_soil"
                if score >= 0.80
                else "colluvial_soil"
                if score >= 0.45
                else "residual_soil"
            )

            # Variáveis heurísticas do MVP.
            # Em versão futura, substituir por MDE, sensoriamento remoto, sensores ou campo.
            elevation_m = round(35 + 45 * score + 0.04 * ring_m + (point_index % 3), 2)
            slope_angle_deg = round(8 + 34 * score - 0.04 * ring_m, 2)
            aspect_deg = round((angle_deg + 90) % 360, 2)
            curvature = round(-0.10 * score + 0.003 * math.sin(3 * angle_rad), 4)
            ndvi = round(clamp(0.78 - 0.55 * score + 0.0015 * ring_m, 0.18, 0.82), 3)

            distance_to_drainage_m = round(max(8, 120 - 80 * score + 0.75 * ring_m), 2)
            distance_to_road_m = round(max(8, 160 - 105 * score + 0.90 * ring_m), 2)

            rainfall_24h_mm = round(45 + 55 * score, 2)
            rainfall_7d_mm = round(170 + 115 * score, 2)

            # Nova variável hidrológica do dataset.
            # No MVP ela é estimada heurísticamente; não é medição direta de campo.
            soil_moisture_pct = round(clamp(18 + 52 * score + 0.08 * rainfall_24h_mm, 15, 85), 2)

            slope_code = f"UBATUBA_PT_{point_global_id:03d}"

            points.append(
                AnalysisPoint(
                    slope_code=slope_code,
                    center_code=center_code,
                    center_name=center_name,
                    center_lon=EVENT_LON,
                    center_lat=EVENT_LAT,
                    ring_m=ring_m,
                    angle_deg=angle_deg,
                    dx_m=dx_m,
                    dy_m=dy_m,
                    distance_to_center_m=distance_to_center_m,
                    elevation_m=elevation_m,
                    slope_angle_deg=slope_angle_deg,
                    aspect_deg=aspect_deg,
                    curvature=curvature,
                    ndvi=ndvi,
                    land_use_class=land_use_class,
                    distance_to_drainage_m=distance_to_drainage_m,
                    distance_to_road_m=distance_to_road_m,
                    soil_type=soil_type,
                    lithology="weathered_rock",
                    rainfall_24h_mm=rainfall_24h_mm,
                    rainfall_7d_mm=rainfall_7d_mm,
                    soil_moisture_pct=soil_moisture_pct,
                    instability_label=instability_label,
                    susceptibility_score=score,
                    susceptibility_class=susceptibility_class,
                )
            )

            point_global_id += 1

    return points


UBATUBA_POINTS: list[AnalysisPoint] = generate_analysis_points()


# -----------------------------------------------------------------------------
# 3. Conexão com o banco
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 4. Criação/ajuste de tabelas
# -----------------------------------------------------------------------------

def ensure_schema(conn) -> None:
    print("Ensuring database schema...")

    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS landslide_events (
            id SERIAL PRIMARY KEY,
            event_code TEXT UNIQUE NOT NULL,
            event_date DATE NOT NULL,
            municipality TEXT,
            state TEXT,
            neighborhood TEXT,
            address_reference TEXT,
            event_type TEXT,
            source_name TEXT,
            source_url TEXT,
            damage_description TEXT,
            rainfall_72h_mm NUMERIC,
            confidence_level TEXT,
            geom geometry(Point, 4326),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("ALTER TABLE landslide_events ADD COLUMN IF NOT EXISTS state TEXT;"))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS landslide_event_buffers (
            id SERIAL PRIMARY KEY,
            event_code TEXT NOT NULL,
            buffer_m INTEGER NOT NULL,
            geom geometry(Polygon, 4326),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS slope_units (
            id SERIAL PRIMARY KEY,
            slope_code TEXT NOT NULL,
            event_code TEXT,
            description TEXT,
            center_code TEXT,
            center_name TEXT,
            center_lon NUMERIC,
            center_lat NUMERIC,
            ring_m NUMERIC,
            angle_deg NUMERIC,
            dx_m NUMERIC,
            dy_m NUMERIC,
            distance_to_center_m NUMERIC,
            geom geometry(Point, 4326),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS event_code TEXT;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS description TEXT;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS center_code TEXT;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS center_name TEXT;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS center_lon NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS center_lat NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS ring_m NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS angle_deg NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS dx_m NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS dy_m NUMERIC;"))
    conn.execute(text("ALTER TABLE slope_units ADD COLUMN IF NOT EXISTS distance_to_center_m NUMERIC;"))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS slope_features (
            id SERIAL PRIMARY KEY,
            slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,
            elevation_m NUMERIC,
            slope_angle_deg NUMERIC,
            aspect_deg NUMERIC,
            curvature NUMERIC,
            ndvi NUMERIC,
            land_use_class TEXT,
            distance_to_drainage_m NUMERIC,
            distance_to_road_m NUMERIC,
            soil_type TEXT,
            lithology TEXT,
            rainfall_24h_mm NUMERIC,
            rainfall_7d_mm NUMERIC,
            soil_moisture_pct NUMERIC,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("""
        ALTER TABLE slope_features
        ADD COLUMN IF NOT EXISTS soil_moisture_pct NUMERIC;
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS training_samples (
            id SERIAL PRIMARY KEY,
            slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,
            instability_label INTEGER,
            label_source TEXT,
            confidence_level TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS susceptibility_results (
            id SERIAL PRIMARY KEY,
            slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,
            model_name TEXT,
            susceptibility_score NUMERIC,
            susceptibility_class TEXT,
            probability_instability NUMERIC,
            run_id TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ml_runs (
            id SERIAL PRIMARY KEY,
            run_id TEXT,
            model_name TEXT,
            algorithm TEXT,
            accuracy NUMERIC,
            precision_score NUMERIC,
            recall_score NUMERIC,
            f1_score NUMERIC,
            roc_auc NUMERIC,
            notes TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_landslide_events_geom
        ON landslide_events USING GIST (geom);
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_slope_units_geom
        ON slope_units USING GIST (geom);
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_event_buffers_geom
        ON landslide_event_buffers USING GIST (geom);
    """))


# -----------------------------------------------------------------------------
# 5. Limpeza somente do estudo de Ubatuba
# -----------------------------------------------------------------------------

def clear_ubatuba_case(conn) -> None:
    print("Cleaning previous Ubatuba MVP records...")

    conn.execute(text("""
        DELETE FROM susceptibility_results
        WHERE slope_id IN (
            SELECT id FROM slope_units WHERE event_code = :event_code
        );
    """), {"event_code": EVENT_CODE})

    conn.execute(text("""
        DELETE FROM training_samples
        WHERE slope_id IN (
            SELECT id FROM slope_units WHERE event_code = :event_code
        );
    """), {"event_code": EVENT_CODE})

    conn.execute(text("""
        DELETE FROM slope_features
        WHERE slope_id IN (
            SELECT id FROM slope_units WHERE event_code = :event_code
        );
    """), {"event_code": EVENT_CODE})

    conn.execute(text("DELETE FROM slope_units WHERE event_code = :event_code;"), {"event_code": EVENT_CODE})
    conn.execute(text("DELETE FROM landslide_event_buffers WHERE event_code = :event_code;"), {"event_code": EVENT_CODE})
    conn.execute(text("DELETE FROM landslide_events WHERE event_code = :event_code;"), {"event_code": EVENT_CODE})
    conn.execute(text("DELETE FROM ml_runs WHERE run_id = :run_id;"), {"run_id": RUN_ID})


# -----------------------------------------------------------------------------
# 6. Inserção do evento, buffers, pontos e atributos
# -----------------------------------------------------------------------------

def insert_landslide_event(conn) -> None:
    print("Inserting landslide event...")

    conn.execute(text("""
        INSERT INTO landslide_events (
            event_code,
            event_date,
            municipality,
            state,
            neighborhood,
            address_reference,
            event_type,
            source_name,
            source_url,
            damage_description,
            rainfall_72h_mm,
            confidence_level,
            geom
        )
        VALUES (
            :event_code,
            :event_date,
            :municipality,
            :state,
            :neighborhood,
            :address_reference,
            :event_type,
            :source_name,
            :source_url,
            :damage_description,
            :rainfall_72h_mm,
            :confidence_level,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
        );
    """), {
        "event_code": EVENT_CODE,
        "event_date": EVENT_DATE,
        "municipality": MUNICIPALITY,
        "state": STATE,
        "neighborhood": NEIGHBORHOOD,
        "address_reference": ADDRESS_REFERENCE,
        "event_type": EVENT_TYPE,
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "damage_description": DAMAGE_DESCRIPTION,
        "rainfall_72h_mm": RAINFALL_72H_MM,
        "confidence_level": CONFIDENCE_LEVEL,
        "lon": EVENT_LON,
        "lat": EVENT_LAT,
    })


def insert_event_buffers(conn, buffers_m: Iterable[int] = BUFFER_DISTANCES_M) -> None:
    print("Creating event buffers...")

    for buffer_m in buffers_m:
        conn.execute(text("""
            INSERT INTO landslide_event_buffers (event_code, buffer_m, geom)
            SELECT
                event_code,
                :buffer_m AS buffer_m,
                ST_Transform(
                    ST_Buffer(ST_Transform(geom, :projected_srid), :buffer_m),
                    4326
                ) AS geom
            FROM landslide_events
            WHERE event_code = :event_code;
        """), {
            "event_code": EVENT_CODE,
            "buffer_m": buffer_m,
            "projected_srid": PROJECTED_SRID,
        })


def insert_slope_units(conn) -> None:
    print("Inserting slope units around the rupture crown...")

    for point in UBATUBA_POINTS:
        conn.execute(text("""
            WITH event_geom AS (
                SELECT ST_Transform(geom, :projected_srid) AS geom_utm
                FROM landslide_events
                WHERE event_code = :event_code
            )
            INSERT INTO slope_units (
                slope_code,
                event_code,
                description,
                center_code,
                center_name,
                center_lon,
                center_lat,
                ring_m,
                angle_deg,
                dx_m,
                dy_m,
                distance_to_center_m,
                geom
            )
            SELECT
                :slope_code,
                :event_code,
                :description,
                :center_code,
                :center_name,
                :center_lon,
                :center_lat,
                :ring_m,
                :angle_deg,
                :dx_m,
                :dy_m,
                :distance_to_center_m,
                ST_Transform(
                    ST_Translate(geom_utm, :dx_m, :dy_m),
                    4326
                ) AS geom
            FROM event_geom;
        """), {
            "event_code": EVENT_CODE,
            "slope_code": point.slope_code,
            "description": "Ponto amostral gerado em anel concêntrico ao redor da coroa da ruptura.",
            "center_code": point.center_code,
            "center_name": point.center_name,
            "center_lon": point.center_lon,
            "center_lat": point.center_lat,
            "ring_m": point.ring_m,
            "angle_deg": point.angle_deg,
            "dx_m": point.dx_m,
            "dy_m": point.dy_m,
            "distance_to_center_m": point.distance_to_center_m,
            "projected_srid": PROJECTED_SRID,
        })


def get_slope_id(conn, slope_code: str) -> int:
    return conn.execute(text("""
        SELECT id
        FROM slope_units
        WHERE slope_code = :slope_code
          AND event_code = :event_code
        LIMIT 1;
    """), {
        "slope_code": slope_code,
        "event_code": EVENT_CODE,
    }).scalar_one()


def insert_slope_features(conn) -> None:
    print("Inserting slope features...")

    for point in UBATUBA_POINTS:
        slope_id = get_slope_id(conn, point.slope_code)

        conn.execute(text("""
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
                rainfall_7d_mm,
                soil_moisture_pct
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
                :rainfall_7d_mm,
                :soil_moisture_pct
            );
        """), {
            "slope_id": slope_id,
            "elevation_m": point.elevation_m,
            "slope_angle_deg": point.slope_angle_deg,
            "aspect_deg": point.aspect_deg,
            "curvature": point.curvature,
            "ndvi": point.ndvi,
            "land_use_class": point.land_use_class,
            "distance_to_drainage_m": point.distance_to_drainage_m,
            "distance_to_road_m": point.distance_to_road_m,
            "soil_type": point.soil_type,
            "lithology": point.lithology,
            "rainfall_24h_mm": point.rainfall_24h_mm,
            "rainfall_7d_mm": point.rainfall_7d_mm,
            "soil_moisture_pct": point.soil_moisture_pct,
        })


def insert_training_samples(conn) -> None:
    print("Inserting training samples...")

    for point in UBATUBA_POINTS:
        slope_id = get_slope_id(conn, point.slope_code)

        conn.execute(text("""
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
            );
        """), {
            "slope_id": slope_id,
            "instability_label": point.instability_label,
            "label_source": "ubatuba_2023_mvp_heuristic_dataset",
            "confidence_level": "medium",
            "notes": "Amostra heurística para MVP acadêmico de Machine Learning aplicado a suscetibilidade.",
        })


def insert_susceptibility_results(conn) -> None:
    print("Inserting susceptibility results...")

    for point in UBATUBA_POINTS:
        slope_id = get_slope_id(conn, point.slope_code)

        conn.execute(text("""
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
            );
        """), {
            "slope_id": slope_id,
            "model_name": MODEL_NAME,
            "susceptibility_score": point.susceptibility_score,
            "susceptibility_class": point.susceptibility_class,
            "probability_instability": point.susceptibility_score,
            "run_id": RUN_ID,
        })

    conn.execute(text("""
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
            :run_id,
            :model_name,
            'rule_based_mvp_dataset_generation',
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            :notes
        );
    """), {
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "notes": (
            "Dataset MVP gerado por classificação heurística. Não representa "
            "validação estatística com inventário histórico completo."
        ),
    })


# -----------------------------------------------------------------------------
# 7. Views e relatórios
# -----------------------------------------------------------------------------

def create_analysis_view(conn) -> None:
    print("Creating analysis views...")

    conn.execute(text("DROP VIEW IF EXISTS vw_ubatuba_susceptibility_points;"))
    conn.execute(text("DROP VIEW IF EXISTS vw_ubatuba_analysis_centers;"))

    conn.execute(text("""
        CREATE VIEW vw_ubatuba_susceptibility_points AS
        SELECT
            su.id AS slope_id,
            su.slope_code,
            su.event_code,
            le.event_date,
            le.municipality,
            le.state,
            le.neighborhood,
            le.address_reference,
            su.center_code,
            su.center_name,
            su.center_lon,
            su.center_lat,
            su.ring_m,
            su.angle_deg,
            su.dx_m,
            su.dy_m,
            su.distance_to_center_m,
            sf.elevation_m,
            sf.slope_angle_deg,
            sf.aspect_deg,
            sf.curvature,
            sf.ndvi,
            sf.land_use_class,
            sf.distance_to_drainage_m,
            sf.distance_to_road_m,
            sf.soil_type,
            sf.lithology,
            sf.rainfall_24h_mm,
            sf.rainfall_7d_mm,
            sf.soil_moisture_pct,
            ts.instability_label,
            sr.model_name,
            sr.susceptibility_score,
            sr.susceptibility_class,
            sr.probability_instability,
            sr.run_id,
            su.geom
        FROM slope_units su
        JOIN landslide_events le
            ON su.event_code = le.event_code
        LEFT JOIN slope_features sf
            ON sf.slope_id = su.id
        LEFT JOIN training_samples ts
            ON ts.slope_id = su.id
        LEFT JOIN susceptibility_results sr
            ON sr.slope_id = su.id
        WHERE su.event_code = 'UBATUBA_2023_02_19_PEREQUE_ACU';
    """))

    conn.execute(text("""
        CREATE VIEW vw_ubatuba_analysis_centers AS
        SELECT DISTINCT
            center_code,
            center_name,
            center_lon,
            center_lat,
            event_code,
            ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326) AS geom
        FROM slope_units
        WHERE event_code = 'UBATUBA_2023_02_19_PEREQUE_ACU'
          AND center_code IS NOT NULL;
    """))


def print_database_summary(conn) -> None:
    print("\nDatabase summary for Ubatuba MVP dataset:")

    queries = {
        "landslide_events": "SELECT COUNT(*) FROM landslide_events WHERE event_code = :event_code",
        "landslide_event_buffers": "SELECT COUNT(*) FROM landslide_event_buffers WHERE event_code = :event_code",
        "slope_units": "SELECT COUNT(*) FROM slope_units WHERE event_code = :event_code",
        "slope_features": """
            SELECT COUNT(*)
            FROM slope_features sf
            JOIN slope_units su ON su.id = sf.slope_id
            WHERE su.event_code = :event_code
        """,
        "training_samples": """
            SELECT COUNT(*)
            FROM training_samples ts
            JOIN slope_units su ON su.id = ts.slope_id
            WHERE su.event_code = :event_code
        """,
        "susceptibility_results": """
            SELECT COUNT(*)
            FROM susceptibility_results sr
            JOIN slope_units su ON su.id = sr.slope_id
            WHERE su.event_code = :event_code
        """,
    }

    for name, query in queries.items():
        count = conn.execute(text(query), {"event_code": EVENT_CODE}).scalar_one()
        print(f"- {name}: {count} record(s)")

    print("\nSusceptibility class distribution:")

    rows = conn.execute(text("""
        SELECT
            susceptibility_class,
            COUNT(*) AS n,
            ROUND(AVG(susceptibility_score)::numeric, 3) AS avg_score,
            ROUND(AVG(soil_moisture_pct)::numeric, 2) AS avg_soil_moisture_pct
        FROM vw_ubatuba_susceptibility_points
        GROUP BY susceptibility_class
        ORDER BY avg_score DESC;
    """)).fetchall()

    for row in rows:
        print(
            f"- {row.susceptibility_class}: {row.n} point(s), "
            f"avg_score={row.avg_score}, "
            f"avg_soil_moisture_pct={row.avg_soil_moisture_pct}%"
        )

    print("\nRing distribution:")

    ring_rows = conn.execute(text("""
        SELECT
            ring_m,
            COUNT(*) AS n,
            ROUND(AVG(susceptibility_score)::numeric, 3) AS avg_score,
            ROUND(AVG(soil_moisture_pct)::numeric, 2) AS avg_soil_moisture_pct
        FROM vw_ubatuba_susceptibility_points
        GROUP BY ring_m
        ORDER BY ring_m;
    """)).fetchall()

    for row in ring_rows:
        print(
            f"- ring {row.ring_m} m: {row.n} point(s), "
            f"avg_score={row.avg_score}, "
            f"avg_soil_moisture_pct={row.avg_soil_moisture_pct}%"
        )

    print("\nSpatial check:")

    spatial_rows = conn.execute(text("""
        SELECT
            slope_code,
            ring_m,
            ROUND(ST_X(geom)::numeric, 6) AS lon,
            ROUND(ST_Y(geom)::numeric, 6) AS lat,
            susceptibility_class,
            susceptibility_score,
            soil_moisture_pct
        FROM vw_ubatuba_susceptibility_points
        ORDER BY slope_code
        LIMIT 20;
    """)).fetchall()

    for row in spatial_rows:
        print(
            f"- {row.slope_code}: ring={row.ring_m}m, "
            f"lon={row.lon}, lat={row.lat}, "
            f"class={row.susceptibility_class}, "
            f"score={row.susceptibility_score}, "
            f"soil_moisture={row.soil_moisture_pct}%"
        )


def print_sig_instructions() -> None:
    print("\nQGIS/ArcGIS layers to load from PostGIS:")
    print("1. landslide_events")
    print("2. landslide_event_buffers")
    print("3. vw_ubatuba_analysis_centers")
    print("4. vw_ubatuba_susceptibility_points")
    print("\nSuggested symbology:")
    print("- landslide_events: red star or red triangle for the rupture crown")
    print("- landslide_event_buffers: transparent fill with dashed outline; categorize by buffer_m")
    print("- vw_ubatuba_analysis_centers: central point of the crown")
    print("- vw_ubatuba_susceptibility_points: categorize by susceptibility_class")


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Starting Ubatuba MVP landslide dataset ETL...")

    database_url = get_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    wait_for_database(engine)

    with engine.begin() as conn:
        ensure_schema(conn)
        clear_ubatuba_case(conn)
        insert_landslide_event(conn)
        insert_event_buffers(conn)
        insert_slope_units(conn)
        insert_slope_features(conn)
        insert_training_samples(conn)
        insert_susceptibility_results(conn)
        create_analysis_view(conn)
        print_database_summary(conn)

    print_sig_instructions()
    print("\nUbatuba MVP landslide dataset ETL finished successfully.")


if __name__ == "__main__":
    main()
