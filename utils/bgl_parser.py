# parser for BGL dataset which returns pandas dataframe

import pandas as pd

def parse_bgl_dataset(file_path):
    headers = [
        "Label",          
        "Timestamp",      
        "Date",           
        "Node",           
        "DateFull",      
        "NodeRepeat",   
        "Type",          
        "Component",     
        "Level",          
        "Content"        
    ]

    parsed_data = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # maxsplit=9 ensures the CONTENT column spaces are ignored
                parts = line.strip().split(maxsplit=9)
                if len(parts) == 10:
                    parsed_data.append(parts)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data, columns=headers).drop(['Date', 'DateFull', 'NodeRepeat'], axis=1) # remove unutilized columns
    
    df['Timestamp'] = pd.to_numeric(df['Timestamp'])
    
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    cols_to_categorical = ['Type', 'Component', 'Level', 'Node']
    for col in cols_to_categorical:
        df[col] = df[col].astype('category')
        
    return df
