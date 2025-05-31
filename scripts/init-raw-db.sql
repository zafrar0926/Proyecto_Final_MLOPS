DROP TABLE IF EXISTS raw_properties;

CREATE TABLE IF NOT EXISTS raw_properties (
    id SERIAL PRIMARY KEY,
    bed INTEGER,
    bath INTEGER,
    acre_lot NUMERIC,
    house_size NUMERIC,
    price NUMERIC,
    total_rooms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_properties_price ON raw_properties(price); 