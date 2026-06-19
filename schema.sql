-- ================================================================
-- Crop Disease Detection — PostgreSQL Schema
-- ================================================================
-- Run this file once to set up the database:
--   psql -U postgres -d crop_disease_db -f schema.sql
--
-- Or run each block manually in pgAdmin / DBeaver
-- ================================================================


-- ── 1. Create the database (run this as postgres superuser) ──────
-- CREATE DATABASE crop_disease_db;
-- \c crop_disease_db


-- ── 2. Users table (for future login feature) ────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(50)  UNIQUE NOT NULL,
    email         VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,           -- store bcrypt hash, never plain text
    full_name     VARCHAR(100),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login    TIMESTAMP,
    is_active     BOOLEAN DEFAULT TRUE
);


-- ── 3. Detections table (prediction history) ─────────────────────
CREATE TABLE IF NOT EXISTS detections (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE SET NULL,  -- NULL = anonymous
    detected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Model output
    disease_name  VARCHAR(100) NOT NULL,           -- raw class name e.g. Tomato___Early_blight
    display_name  VARCHAR(100),                    -- formatted   e.g. Tomato Early Blight
    confidence    NUMERIC(5,2) NOT NULL,           -- 0.00 to 100.00
    plant_type    VARCHAR(50),                     -- e.g. Tomato, Apple, Potato

    -- Disease info (stored so it's available even if model_handler changes)
    description   TEXT,
    symptoms      JSONB,                           -- list of symptom strings
    treatment     JSONB,                           -- list of treatment strings
    prevention    JSONB,                           -- list of prevention strings

    -- Image reference
    image_path    VARCHAR(255)                     -- e.g. static/uploads/abc123.png
);


-- ── 4. Indexes for fast queries ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_detections_user_id     ON detections(user_id);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_disease     ON detections(disease_name);
CREATE INDEX IF NOT EXISTS idx_detections_plant_type  ON detections(plant_type);


-- ── 5. Useful views ───────────────────────────────────────────────

-- Summary: count of each disease detected
CREATE OR REPLACE VIEW disease_summary AS
SELECT
    display_name,
    plant_type,
    COUNT(*)              AS total_detections,
    ROUND(AVG(confidence),2) AS avg_confidence,
    MAX(detected_at)      AS last_detected
FROM detections
GROUP BY display_name, plant_type
ORDER BY total_detections DESC;


-- Recent 50 detections with user info
CREATE OR REPLACE VIEW recent_detections AS
SELECT
    d.id,
    d.detected_at,
    d.display_name,
    d.plant_type,
    d.confidence,
    d.image_path,
    u.username
FROM detections d
LEFT JOIN users u ON d.user_id = u.id
ORDER BY d.detected_at DESC
LIMIT 50;


-- ================================================================
-- Sample queries you will use in app.py
-- ================================================================

-- Get all detections (history page):
-- SELECT * FROM detections ORDER BY detected_at DESC;

-- Get detections for one user:
-- SELECT * FROM detections WHERE user_id = %s ORDER BY detected_at DESC;

-- Get single detection by id:
-- SELECT * FROM detections WHERE id = %s;

-- Get related detections (same disease or same plant):
-- SELECT * FROM detections
-- WHERE (disease_name = %s OR plant_type = %s) AND id != %s
-- ORDER BY detected_at DESC LIMIT 4;

-- Delete a detection:
-- DELETE FROM detections WHERE id = %s;

-- Most common diseases:
-- SELECT disease_name, COUNT(*) as count FROM detections
-- GROUP BY disease_name ORDER BY count DESC LIMIT 10;