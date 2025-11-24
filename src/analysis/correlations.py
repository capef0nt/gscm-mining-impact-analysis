import pandas as pd

def correlation_table(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr().round(2)
    return corr
