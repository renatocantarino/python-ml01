import os
import sys
import pandas as pd
import yaml
import psycopg2


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_yaml_config

def fetch_data_from_db(sql_query):
    try:        
        yaml_loaded = load_yaml_config()
        if not yaml_loaded:
            raise ValueError("YAML configuration could not be loaded.")
        
        con = psycopg2.connect(
            dbname=yaml_loaded['database_config']['dbname'], 
            user=yaml_loaded['database_config']['user'], 
            password=yaml_loaded['database_config']['password'], 
            host=yaml_loaded['database_config']['host']
        )

        cursor = con.cursor()
        cursor.execute(sql_query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    return df
                                                            



                