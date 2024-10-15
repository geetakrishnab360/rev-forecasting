import pandas as pd
from functools import reduce
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def preprocess(
        dataframe,
):
    def convert_columns_to_lower(df):
        df.columns = [col.lower() for col in df.columns]
        return df

    def convert_to_dates(df):
        df['date'] = pd.to_datetime(df['date'])
        return df

    def convert_to_float(df):
        df['revenue'] = df['revenue'].astype(float)
        return df

    def rename_columns(df):
        df.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)
        return df

    def trim_date(df):
        df['ds'] = df.ds.apply(lambda x: datetime(x.year, x.month, 1))
        return df

    return reduce(
        lambda df, func: func(df),
        [
            convert_columns_to_lower,
            convert_to_dates,
            convert_to_float,
            rename_columns,
            trim_date,
        ],
        dataframe,
    )
