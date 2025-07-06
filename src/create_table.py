import pandas as pd
from sqlalchemy import text
from create_connection import create_connection
import yaml
import logging
import logging.config
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)

with open("config/logging.yaml", "r") as file:
    logging.config.dictConfig(yaml.safe_load(file))

logger = logging.getLogger(__name__)

def create_table(schema_name, table_name, data, cols_schema, process='append'):

    def generate_create_table_sql(schema_name, table_name):

        columns_sql = []
        for col_name, col_type in cols_schema.items():

            if table_name == 'test' and col_name == 'target':
                continue
            
            columns_sql.append(f"{col_name} {col_type}")
        
        return f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            {',\n        '.join(columns_sql)}
        )
        """

    create_schema = text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
    create_table = text(generate_create_table_sql(schema_name, table_name))

    try:
        logger.info("Connecting to the database")
        engine = create_connection()

        with engine.begin() as conn:

            logger.info("Creating schema and table structure")
            conn.execute(create_schema)
            conn.execute(create_table)

            if process == 'overwrite':
                logger.info(f"Overwriting table {schema_name}.{table_name}")
                conn.execute(text(f"TRUNCATE TABLE {schema_name}.{table_name}"))
            elif process == 'append':
                logger.info(f"Appending data to table {schema_name}.{table_name}")
            else:
                raise ValueError(f"{process} is not a valid process to do in the database")
  
            columns = ', '.join(data.columns)
            placeholders = ', '.join([':' + col for col in data.columns])
            insert_query = text(f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({placeholders})")
            
            records = data.to_dict('records')
            
            chunk_size = 80000
            logger.info(f"Starting data load into {schema_name}.{table_name} ...")
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                conn.execute(insert_query, chunk)
            
            logger.info(f"Successfully loaded all {len(records)} records into {schema_name}.{table_name}")
            engine.dispose()
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':

    data_files = [(r'original_data\train.csv', 'train'), (r'original_data\test.csv', 'test')]

    schema_name_raw = "raw"

    cols_schema_raw = {
        'id': 'INTEGER',
        'target': 'INTEGER',
        'ps_ind_01': 'INTEGER',
        'ps_ind_02_cat': 'INTEGER',
        'ps_ind_03': 'INTEGER',
        'ps_ind_04_cat': 'INTEGER',
        'ps_ind_05_cat': 'INTEGER',
        'ps_ind_06_bin': 'INTEGER',
        'ps_ind_07_bin': 'INTEGER',
        'ps_ind_08_bin': 'INTEGER',
        'ps_ind_09_bin': 'INTEGER',
        'ps_ind_10_bin': 'INTEGER',
        'ps_ind_11_bin': 'INTEGER',
        'ps_ind_12_bin': 'INTEGER',
        'ps_ind_13_bin': 'INTEGER',
        'ps_ind_14': 'INTEGER',
        'ps_ind_15': 'INTEGER',
        'ps_ind_16_bin': 'INTEGER',
        'ps_ind_17_bin': 'INTEGER',
        'ps_ind_18_bin': 'INTEGER',
        'ps_reg_01': 'FLOAT',
        'ps_reg_02': 'FLOAT',
        'ps_reg_03': 'FLOAT',
        'ps_car_01_cat': 'INTEGER',
        'ps_car_02_cat': 'INTEGER',
        'ps_car_03_cat': 'INTEGER',
        'ps_car_04_cat': 'INTEGER',
        'ps_car_05_cat': 'INTEGER',
        'ps_car_06_cat': 'INTEGER',
        'ps_car_07_cat': 'INTEGER',
        'ps_car_08_cat': 'INTEGER',
        'ps_car_09_cat': 'INTEGER',
        'ps_car_10_cat': 'INTEGER',
        'ps_car_11_cat': 'INTEGER',
        'ps_car_11': 'INTEGER',
        'ps_car_12': 'FLOAT',
        'ps_car_13': 'FLOAT',
        'ps_car_14': 'FLOAT',
        'ps_car_15': 'FLOAT',
        'ps_calc_01': 'FLOAT',
        'ps_calc_02': 'FLOAT',
        'ps_calc_03': 'FLOAT',
        'ps_calc_04': 'INTEGER',
        'ps_calc_05': 'INTEGER',
        'ps_calc_06': 'INTEGER',
        'ps_calc_07': 'INTEGER',
        'ps_calc_08': 'INTEGER',
        'ps_calc_09': 'INTEGER',
        'ps_calc_10': 'INTEGER',
        'ps_calc_11': 'INTEGER',
        'ps_calc_12': 'INTEGER',
        'ps_calc_13': 'INTEGER',
        'ps_calc_14': 'INTEGER',
        'ps_calc_15_bin': 'INTEGER',
        'ps_calc_16_bin': 'INTEGER',
        'ps_calc_17_bin': 'INTEGER',
        'ps_calc_18_bin': 'INTEGER',
        'ps_calc_19_bin': 'INTEGER',
        'ps_calc_20_bin': 'INTEGER'
    }

    for path, table_name in data_files:
        df = pd.read_csv(f'{path}')
        create_table(schema_name_raw, table_name, df, cols_schema_raw)


