import pandas as pd
from sqlalchemy import text
from src.create_connection import create_connection

def extract_data(select_query):

    '''
    Function that serves to extract data
    from a postgres database based on a 
    input query    
    '''

    try:
        engine = create_connection()

        with engine.begin() as connection:
            result = connection.execute(text(select_query))
            
            rows = result.fetchall()
            
            df = pd.DataFrame(rows, columns=result.keys())

        engine.dispose()

        return df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
        
    finally:
        engine.dispose()