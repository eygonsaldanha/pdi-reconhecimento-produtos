import pandas as pd
from sqlalchemy import create_engine

POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_USER = 'admin'
POSTGRES_PASSWORD = 'admin123'
POSTGRES_DB = 'app_db'

engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")


def __convert_dict__(data_list):
    result = {}

    for item in data_list:
        for key, value in item.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    return result


def insert_data(table, data, need_convert=True):
    if need_convert:
        data = __convert_dict__(data)

    pd.DataFrame(data).to_sql(table.lower(), engine, if_exists="append", index=False)


select = lambda sql: pd.read_sql(sql, engine)
