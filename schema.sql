-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS scores;


CREATE TABLE scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT,
  time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  false_positive_rate Float,
  true_positive_rate Float
);
