CREATE TABLE user_data (
    ID SERIAL PRIMARY KEY,
    acne_cells INT,
    acne_coverage FLOAT,
    date DATETIME,
    filename VARCHAR(255),
    user_id VARCHAR(255)
);