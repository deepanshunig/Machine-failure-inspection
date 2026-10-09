"""Local configuration and safe MySQL URL construction."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = PROJECT_ROOT
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_name: str = "machine_failure_dashboard"
    db_user: str = "machine_failure_app"
    db_password: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)
    db_ssl_ca: str = ""

    @property
    def db_configured(self) -> bool:
        return bool(self.db_password and self.db_password != "YOUR_APP_PASSWORD")

    @property
    def database_url(self) -> URL:
        return URL.create(
            "mysql+pymysql",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            query={"charset": "utf8mb4"},
        )


def read_settings(root: Path = PROJECT_ROOT) -> Settings:
    load_dotenv(root / ".env", override=False)
    return Settings(
        root=root,
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=int(os.getenv("DB_PORT", "3306")),
        db_name=os.getenv("DB_NAME", "machine_failure_dashboard"),
        db_user=os.getenv("DB_USER", "machine_failure_app"),
        db_password=os.getenv("DB_PASSWORD", ""),
        api_key=os.getenv("DEMO_API_KEY", ""),
        db_ssl_ca=os.getenv("DB_SSL_CA", ""),
    )
