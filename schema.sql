-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS scores;


CREATE TABLE scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT,
  time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  HP TEXT,
  Loss float,
  PD float,
  SD float,
  F1 float,
  UD float,
  ND float,
  TNR float,
  FNR float,
  FPR Float,
  TPR Float,
  TPR_T Float
);

