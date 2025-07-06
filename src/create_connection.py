from sqlalchemy import create_engine
from sqlalchemy.engine import URL

def create_connection():
    url = URL.create(
            drivername="postgresql",
            username="porto-seguro-db",
            password="porto-seguro-db",
            host="localhost",
            port=5431,
            database="porto-seguro-db"
        )

    engine = create_engine(url)

    return engine