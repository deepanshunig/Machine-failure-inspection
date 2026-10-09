from alembic import context

from machine_failure.config import read_settings
from machine_failure.db import Base, make_engine

config = context.config
settings = config.attributes.get("settings") or read_settings()
engine = make_engine(settings)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
