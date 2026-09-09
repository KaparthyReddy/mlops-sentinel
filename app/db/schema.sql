CREATE TABLE prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    model_alias VARCHAR(20) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    features JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE feedback_labels (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL REFERENCES prediction_logs(request_id),
    actual_label INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE drift_scores (
    id BIGSERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    psi_score DOUBLE PRECISION NOT NULL,
    sample_size INTEGER NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE retraining_events (
    id BIGSERIAL PRIMARY KEY,
    trigger_reason VARCHAR(200) NOT NULL,
    max_psi_score DOUBLE PRECISION,
    triggered_at TIMESTAMP NOT NULL DEFAULT now(),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
);

CREATE INDEX idx_prediction_logs_model_alias ON prediction_logs(model_alias);
CREATE INDEX idx_prediction_logs_created_at ON prediction_logs(created_at);
CREATE INDEX idx_drift_scores_feature ON drift_scores(feature_name);
CREATE INDEX idx_drift_scores_computed_at ON drift_scores(computed_at);
