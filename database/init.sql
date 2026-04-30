-- ============================================================
-- Projeto GIS / Taludes
-- Banco: PostgreSQL + PostGIS
-- Objetivo: estrutura inicial para análise de suscetibilidade
--           a instabilidades em taludes.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------
-- 1. Projetos
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL,
    student_name TEXT,
    institution TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 2. Áreas de estudo
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS study_areas (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    area_name TEXT NOT NULL,
    municipality TEXT,
    state_code TEXT,
    country TEXT DEFAULT 'Brazil',
    srid INTEGER DEFAULT 4326,
    geom GEOMETRY(POLYGON, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_study_areas_geom
ON study_areas
USING GIST (geom);

-- ------------------------------------------------------------
-- 3. Taludes / pontos ou unidades de análise
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS slope_units (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    slope_code TEXT NOT NULL,
    slope_type TEXT,                -- natural, cut_slope, embankment
    description TEXT,
    geom GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_slope_units_geom
ON slope_units
USING GIST (geom);

-- ------------------------------------------------------------
-- 4. Camadas geoespaciais de referência
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS geospatial_layers (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    layer_name TEXT NOT NULL,
    layer_type TEXT,                -- raster, vector, dem, remote_sensing
    source_name TEXT,
    source_path TEXT,
    acquisition_date DATE,
    srid INTEGER DEFAULT 4326,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 5. Features geoespaciais para análise/ML
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS slope_features (
    id SERIAL PRIMARY KEY,
    slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,

    elevation_m NUMERIC(12, 4),
    slope_angle_deg NUMERIC(10, 4),
    aspect_deg NUMERIC(10, 4),
    curvature NUMERIC(14, 6),

    ndvi NUMERIC(10, 6),
    land_use_class TEXT,
    distance_to_drainage_m NUMERIC(12, 4),
    distance_to_road_m NUMERIC(12, 4),

    soil_type TEXT,
    lithology TEXT,
    rainfall_24h_mm NUMERIC(12, 4),
    rainfall_7d_mm NUMERIC(12, 4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 6. Amostras rotuladas para treinamento
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS training_samples (
    id SERIAL PRIMARY KEY,
    slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,

    instability_label INTEGER NOT NULL,  -- 0 = stable/no event, 1 = unstable/event
    label_source TEXT,                   -- field_mapping, visual_interpretation, report
    confidence_level TEXT,               -- low, medium, high
    notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 7. Resultados de suscetibilidade
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS susceptibility_results (
    id SERIAL PRIMARY KEY,
    slope_id INTEGER REFERENCES slope_units(id) ON DELETE CASCADE,

    model_name TEXT NOT NULL,
    susceptibility_score NUMERIC(10, 6),
    susceptibility_class TEXT,           -- Low, Medium, High, Critical

    probability_instability NUMERIC(10, 6),
    run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 8. Execuções de modelos de Machine Learning
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ml_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    algorithm TEXT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    accuracy NUMERIC(10, 6),
    precision_score NUMERIC(10, 6),
    recall_score NUMERIC(10, 6),
    f1_score NUMERIC(10, 6),
    roc_auc NUMERIC(10, 6),

    notes TEXT
);

-- ------------------------------------------------------------
-- 9. Dados demonstrativos iniciais
-- ------------------------------------------------------------

INSERT INTO projects (
    project_name,
    student_name,
    institution,
    description
)
VALUES (
    'Slope Instability Susceptibility Mapping - MVP',
    'Ricardo',
    'UNICAMP',
    'Initial MVP for a GIS-based intelligent system for slope instability susceptibility analysis using remote sensing, PostGIS and machine learning.'
);

INSERT INTO study_areas (
    project_id,
    area_name,
    municipality,
    state_code,
    geom
)
SELECT
    id,
    'Demonstration Study Area',
    'Campinas',
    'SP',
    ST_GeomFromText(
        'POLYGON((-47.10 -22.95, -47.00 -22.95, -47.00 -22.85, -47.10 -22.85, -47.10 -22.95))',
        4326
    )
FROM projects
WHERE project_name = 'Slope Instability Susceptibility Mapping - MVP'
LIMIT 1;

INSERT INTO slope_units (
    project_id,
    slope_code,
    slope_type,
    description,
    geom
)
SELECT
    id,
    'SLOPE_001',
    'cut_slope',
    'Demonstration slope unit with low susceptibility.',
    ST_SetSRID(ST_MakePoint(-47.070, -22.920), 4326)
FROM projects
WHERE project_name = 'Slope Instability Susceptibility Mapping - MVP'
LIMIT 1;

INSERT INTO slope_units (
    project_id,
    slope_code,
    slope_type,
    description,
    geom
)
SELECT
    id,
    'SLOPE_002',
    'natural',
    'Demonstration slope unit with medium susceptibility.',
    ST_SetSRID(ST_MakePoint(-47.055, -22.905), 4326)
FROM projects
WHERE project_name = 'Slope Instability Susceptibility Mapping - MVP'
LIMIT 1;

INSERT INTO slope_units (
    project_id,
    slope_code,
    slope_type,
    description,
    geom
)
SELECT
    id,
    'SLOPE_003',
    'cut_slope',
    'Demonstration slope unit with high susceptibility.',
    ST_SetSRID(ST_MakePoint(-47.035, -22.890), 4326)
FROM projects
WHERE project_name = 'Slope Instability Susceptibility Mapping - MVP'
LIMIT 1;

-- ============================================================
-- Fim do init.sql
-- ============================================================