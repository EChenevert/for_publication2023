# for_publication2023

## Directions for Use

To replicate the study, run the BLR_results.py file for the Bayesian Linear regression portion of the study. For the 
Gaussian Process regression portion of the study, run GPR_results file. Make sure to download all .py files in the same 
repository. Particularly, the main.py and funcs.py files.

## Descriptions of Datasets

CRMS_Accretion.csv: This dataset holds all vertical accretion measurements taken at all CRMS stations. This is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/

CRMS_Discrete_Hydrographic.csv: Contains monthly averages of hourly measurements of hydrologic data. These measurements are taken from an underwater sonde placed in a nearby body of water to the CRMS station. This is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/

CRMS_GEE60pfrom2007to2022.csv: Dataset compiled from google earth engine and ...

CRMS_Marsh_Vegetation: Dataset of descriptions regarding the vegetation around nearby a CRMS station. This is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/

CRMS Soil Properties: Dataset of sedimentologic characteristics taken from cores upon site initialization and subsequently every 6 to 10 years. This is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/

final_flooddepths.py: This is final product of preprocessing hourly hydrologic measurements of water depths. Since the full dataset of hourly hydrologic data is too large to host on GitHub, we only use the final product here. The raw hourly hydrologic data is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/. See the dask_hydro.py and compute_hydro_hourly_vars.py for our methodology of preprocessing the hourly hydrologic data.

final_floodfreq.py: This is also the final product of preprocessing hourly hydrologic measurements of water depths. However, instead of recording teh depth of flooding we only record the number of times a site experiences a flood per year. Since the full dataset of hourly hydrologic data is too large to host on GitHub, we only use the final product here. The raw hourly hydrologic data is openly available at https://cims.coastal.louisiana.gov/ and https://www.lacoast.gov/CRMS/. See the dask_hydro.py and compute_hydro_hourly_vars.py for our methodology of preprocessing the hourly hydrologic data.

for_distanceWater_ex.py: A dataset describing the distance a CRMS station is from a persistent body of water. Distance were calcualted using ArcMap.

percentflooded.csv: Dataset taken from https://www.lacoast.gov/CRMS/ calculated from hourly measurements waterlevel relative to the marsh elevation. It describes the amount of percent time a CRMS station is inundated with water a year.  

## Descriptions of .py files

main.py: This file is used to compile the raw data from the Coastal Information Management System website. 
Accretion rates are derived in this file along with averages of hydrologic data, soil properties, and marsh vegetation.

funcs.py: This script is used to hold any functions called in other files within the project.

BLR_results.py: Here we load the files compiled in main.py as well as files compiled elsewhere in google earth engine 
and ArcMap. Then we run the Bayesian Linear regression portion of the study, printing all plots used in the 
investigation.

GPR_results.py: We load the files ompiled in main.py as well as files compiled elsewhere in google earth engine 
and ArcMap. Then we run the Gaussian Process regression portion of the study, with all relevant plots.

dask_hydro.py: This file was used in the preprocessing of the hourly hydrologic data, since the original data was too 
large to handle. Here we split the dataset into smaller, more manageable chunks.

compute_hydro_hourly_vars.py: Here is where we calculate the flood frequency and flood depth variables from the 
hydrologic data split in dask_hydro.py. 

final_average_hourly_vars.py: Here we average the hourly hydrologic variables per site to make a final dataset of hourly 
hydrologic variables.

data_vis_paper.py: Here we simply make visualizations and some exploratory work our final dataset used for the machine 
learning experiments.

data_vis_for_GPR.py: In this file we create some figures used in the GPR results.

explore_NDVI_Biomass.py: Used to make plots of NDVI versus organic matter % and aboveground biomass. The aboveground 
biomass data is taken from the Coastal Information Monitoring System website and is freely available.


