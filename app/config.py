from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://mlops_user:change_me_locally@localhost:5433/mlops_sentinel"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "fraud-detector"
    drift_psi_threshold: float = 0.2
    retraining_check_interval_minutes: int = 60
    ab_test_challenger_traffic_percent: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
