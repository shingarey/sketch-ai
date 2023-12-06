def create_sql_engine():
    from sqlalchemy import create_engine

    engine = create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
    )
    return engine
