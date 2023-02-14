import dask.dataframe as dd
from dask.dataframe import to_numeric

df = dd.read_csv("D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\CRMS_Continuous_Hydrographic.csv",
                 encoding="unicode_escape",  dtype=object).repartition(npartitions=10)
# df = to_numeric(df, errors='coerce')
df.to_csv("D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\export-*.csv")

