import os 
import pandas as pd
from src.ml_agent.schemas.eda_spec import EDAReport
from dotenv import load_dotenv

load_dotenv()

def analyse_dataset(
        df:pd.DataFrame,
        target_column: str|None
                    ) -> dict:
    #df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include=["object","category","bool"])

    missing = df.isna().sum()
    missing = missing[missing>0]
    duplicated_rows = int(df.duplicated().sum())

    quality = {
        "raw_count": len(df),
        "column_count": len(df.columns),
        "columns":[{"name":col, "dtype":str(df[col].dtype)} for col in df.columns],
        "missing_values": {
            col: {
                "count": int(count),
                "percentage": round(count / len(df)*100,2) if len(df) else 0,
            }
            for col, count in missing.items()
        },
        "duplicate_rows":duplicated_rows,

    }

    return {
        "data_quality":quality
    }