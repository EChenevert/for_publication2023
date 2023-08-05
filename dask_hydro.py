import dask.dataframe as dd

# Here I use dask to break up the full hourly hydrographic data from the CIMS website into manageable portions.
# I then save to my local computer to re-load in the compute_hydro_hourly_vars.py file.
# DISCLAIMER:
# Since the full hourly hydrologic dataset is too large to host on GitHub, this code will not run if one downloads it.
# However, one could use it as a guide to my workflow in preprocessing the hourly hydrologic data.
df = dd.read_csv("D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\CRMS_Continuous_Hydrographic.csv",
                 encoding="unicode_escape",  dtype=object).repartition(npartitions=10)
df.to_csv("D:\Etienne\PAPER_2023\CRMS_Continuous_Hydrographic\export-*.csv")

