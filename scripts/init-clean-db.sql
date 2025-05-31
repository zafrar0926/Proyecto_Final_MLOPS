DROP TABLE IF EXISTS clean_properties;

CREATE TABLE IF NOT EXISTS clean_properties (
    id SERIAL PRIMARY KEY,
    bed INTEGER,
    bath INTEGER,
    acre_lot NUMERIC,
    house_size NUMERIC,
    price NUMERIC,
    total_rooms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clean_properties_price ON clean_properties(price); 