CREATE OR REPLACE VIEW public.vw_slope_susceptibility AS
SELECT
    su.id AS objectid,
    su.slope_code,
    su.slope_type,
    su.description,

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

    sr.model_name,
    sr.susceptibility_score,
    sr.susceptibility_class,
    sr.probability_instability,
    sr.run_id,

    su.geom
FROM public.slope_units su
LEFT JOIN public.slope_features sf
    ON su.id = sf.slope_id
LEFT JOIN public.susceptibility_results sr
    ON su.id = sr.slope_id;