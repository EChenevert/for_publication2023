from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import main
import pandas as pd
import numpy as np
import funcs
import seaborn as sns


# Everything I need for this should be within the file "D:\Etienne\fall2022\agu_data"
## Data from CIMS
data = main.load_data()
bysite = main.average_bysite(data)


## Data from CRMS
url_perc = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/percentflooded.csv"
perc = pd.read_csv(url_perc, encoding="unicode escape")
perc['Simple site'] = [i[:8] for i in perc['Station_ID']]
perc = perc.groupby('Simple site').median()

url_wl = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/waterlevelrange.csv"
wl = pd.read_csv(url_wl, encoding="unicode escape")[['Station_ID', 'Tide_Amp (ft)']]
wl['Simple site'] = [i[:8] for i in wl['Station_ID']]
wl = wl.groupby('Simple site').median()

## Data from Gee and Arc
url_jrc = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/CRMS_GEE_JRCCOPY2.csv"
jrc = pd.read_csv(url_jrc, encoding="unicode_escape")[
    ['Simple_sit', 'Land_Lost_m2']
].set_index('Simple_sit')

url_gee = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/CRMS_GEE60pfrom2007to2022.csv"
gee = pd.read_csv(url_gee, encoding="unicode escape")[['Simple_sit', 'NDVI', 'tss_med', 'windspeed']]\
    .groupby('Simple_sit').median().fillna(0)  # filling nans with zeros cuz all nans are in tss because some sites are not near water


url_distRiver = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/totalDataAndRivers.csv"
distRiver = pd.read_csv(url_distRiver, encoding="unicode escape")[['Field1', 'distance_to_river_m', 'width_mean']].groupby('Field1').median()

url_nearWater = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/for_distanceWater_ex.csv"
nearWater = pd.read_csv(url_nearWater, encoding="unicode_escape")[
    ['Simple site', 'Distance_to_Water_m']
].set_index('Simple site')
# Add flooding frequency
url_floodfreq = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/final_floodfreq.csv"
floodfreq = pd.read_csv(url_floodfreq, encoding="unicode_escape")[[
    'Simple site', 'Flood Freq (Floods/yr)'
]].set_index('Simple site')
# add flood depth when flooded
url_floodDepth = "https://raw.githubusercontent.com/EChenevert/for_publication2023/main/final_flooddepths.csv"
floodDepth = pd.read_csv(url_floodDepth, encoding="unicode_escape")[[
    'Simple site', 'Avg. Flood Depth when Flooded (ft)', '90th Percentile Flood Depth when Flooded (ft)',
    '10th Percentile Flood Depth when Flooded (ft)', 'Std. Deviation Flood Depth when Flooded '
]].set_index('Simple site')

# Concatenate
df = pd.concat([bysite, distRiver, nearWater, gee, jrc, wl, perc, floodfreq, floodDepth],
               axis=1, join='outer')

# Now clean the columns
# First delete columns that are more than 1/2 nans
tdf = df.dropna(thresh=df.shape[0]*0.5, how='all', axis=1)
# tdf = df.dropna(thresh=df.shape[0]*0.3, how='all', axis=1)  # this threshold lets sand, silt, clay terms stay
# Drop uninformative features
udf = tdf.drop([
    'Year (yyyy)', 'Accretion Measurement 1 (mm)', 'Year',
    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    'Accretion Measurement 4 (mm)', 'Measurement Depth (ft)',
    'Month (mm)', 'Average Accretion (mm)', 'Delta time (days)', 'Wet Volume (cm3)',
    'Delta Time (decimal_years)', 'Wet Soil pH (pH units)', 'Dry Soil pH (pH units)', 'Dry Volume (cm3)',
    'percent_waterlevel_complete'
], axis=1)


# Address the vertical measurement for mass calculation (wit the potential of switching between my accretion and
# CRMS accretion)
vertical = 'Accretion Rate (mm/yr)'

####### Define outcome as vertical component
outcome = 'Accretion Rate (mm/yr)'

# Try to semi-standardize variables
des = udf.describe()  # just to identify which variables are way of the scale
udf['distance_to_river_km'] = udf['distance_to_river_m']/1000  # convert to km
udf['river_width_mean_km'] = udf['width_mean']/1000
udf['distance_to_water_km'] = udf['Distance_to_Water_m']/1000
udf['land_lost_km2'] = udf['Land_Lost_m2']*0.000001  # convert to km2

# Drop remade variables
udf = udf.drop(['distance_to_river_m', 'width_mean', 'Distance_to_Water_m',
                'Soil Specific Conductance (uS/cm)',
                'Land_Lost_m2'], axis=1)
udf = udf.rename(columns={'tss_med': 'TSS (mg/L)'})

# Delete the swamp sites and unammed basin
udf.drop(udf.index[udf['Community'] == 'Swamp'], inplace=True)
udf = udf.drop('Basins', axis=1)
# ----
udf = udf.drop([
    'Staff Gauge (ft)', 'Soil Porewater Temperature (¡C)', 'Soil Porewater Specific Conductance (uS/cm)',
    'Soil Salinity (ppt)',
    'river_width_mean_km',   # 'log_river_width_mean_km',  # i just dont like this variable because it has a sucky distribution
    # Delete the dominant herb cuz of rendundancy with dominant veg
    'Average Height Herb (cm)',
    'Organic Density (g/cm3)',
    'Soil Moisture Content (%)',
    'land_lost_km2'
], axis=1)
# conduct outlier removal which drops all nans
rdf = funcs.max_interquartile_outlierrm(udf.drop(['Community', 'Latitude', 'Longitude', 'Bulk Density (g/cm3)',
                                                  'Organic Matter (%)'], axis=1).dropna(), outcome)
# transformations (basically log transforamtions) --> the log actually kinda regularizes too
rdf['log_distance_to_water_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_water_km']]
rdf['log_distance_to_river_km'] = [np.log(val) if val > 0 else 0 for val in rdf['distance_to_river_km']]

# drop the old features
rdf = rdf.drop(['distance_to_water_km', 'distance_to_river_km'], axis=1)

# Rename some variables for better text wrapping
rdf = rdf.rename(columns={
    'Tide_Amp (ft)': 'Tide Amp (ft)',
    'avg_percentflooded (%)': 'Avg. Time Flooded (%)',
    'windspeed': 'Windspeed (m/s)',
    # 'log_distance_to_ocean_km': 'Log Distance to Ocean (km)',
    'log_distance_to_water_km': 'Log Distance to Water (km)',
    'log_distance_to_river_km': 'Log Distance to River (km)',
    # My flood depth vars
    '90th Percentile Flood Depth when Flooded (ft)': '90th Percentile Flood Depth (ft)',
    '10th Percentile Flood Depth when Flooded (ft)': '10th Percentile Flood Depth (ft)',
    'Avg. Flood Depth when Flooded (ft)': 'Avg. Flood Depth (ft)',
    'Std. Deviation Flood Depth when Flooded ': 'Std. Deviation Flood Depth (ft)'
})

gdf = pd.concat([rdf, udf[['Community', 'Latitude', 'Longitude', 'Organic Matter (%)', 'Bulk Density (g/cm3)']]],
                axis=1, join='inner')
# Transform all units to SI units
gdf['Tidal Amplitude (cm)'] = gdf['Tide Amp (ft)'] * 30.48
gdf['90th Percentile Flood Depth (cm)'] = gdf['90th Percentile Flood Depth (ft)'] * 30.48
gdf['10th Percentile Flood Depth (cm)'] = gdf['10th Percentile Flood Depth (ft)'] * 30.48
gdf['Avg. Flood Depth (cm)'] = gdf['Avg. Flood Depth (ft)'] * 30.48
gdf['Std. Deviation Flood Depth (cm)'] = gdf['Std. Deviation Flood Depth (ft)'] * 30.48

# Delete the old non SI unit variables
# gdf = gdf.drop(['Std. Deviation Flood Depth (ft)', 'Avg. Flood Depth (ft)', 'Tide Amp (ft)'], axis=1)
gdf = gdf.drop(['Std. Deviation Flood Depth (ft)', 'Avg. Flood Depth (ft)', '10th Percentile Flood Depth (ft)',
                '90th Percentile Flood Depth (ft)', 'Tide Amp (ft)'], axis=1)

brackdf = gdf[gdf['Community'] == 'Brackish']
saldf = gdf[gdf['Community'] == 'Saline']
freshdf = gdf[gdf['Community'] == 'Freshwater']
interdf = gdf[gdf['Community'] == 'Intermediate']
combined = gdf[(gdf['Community'] == 'Intermediate') | (gdf['Community'] == 'Brackish')]
freshinter = gdf[(gdf['Community'] == 'Intermediate') | (gdf['Community'] == 'Freshwater')]
bracksal = gdf[(gdf['Community'] == 'Saline') | (gdf['Community'] == 'Brackish')]
# Exclude swamp
marshdic = {'All': gdf, 'Brackish': brackdf, 'Saline': saldf, 'Freshwater': freshdf, 'Intermediate': interdf,
            'Intermediate and Brackish': combined, 'Freshwater and Intermediate': freshinter,
            'Brackish and Saline': bracksal}


hold_marsh_weights = {}
hold_unscaled_weights = {}
hold_intercept = {}
hold_marsh_regularizors = {}
hold_marsh_weight_certainty = {}
hold_prediction_certainty = {}

for key in marshdic:
    print(key)
    mdf = marshdic[key]
    t = mdf[outcome].reset_index().drop('index', axis=1)
    phi = mdf.drop([outcome, 'Community', 'Latitude', 'Longitude',  'Organic Matter (%)', 'Bulk Density (g/cm3)',
                    ],
                   axis=1).reset_index().drop('index', axis=1)
    # Scale: because I want feature importances
    scalar_Xmarsh = StandardScaler()
    predictors_scaled = pd.DataFrame(scalar_Xmarsh.fit_transform(phi), columns=phi.columns.values)

    bestfeaturesM = funcs.backward_elimination(predictors_scaled, t, num_feats=20, significance_level=0.05)

    X, y = predictors_scaled[bestfeaturesM], t

    baymod = linear_model.BayesianRidge(fit_intercept=True)

    results_dict = funcs.cv_results_and_plot(baymod, bestfeaturesM, phi, X, y, {'cmap': 'YlOrRd', 'line': "r--"}, str(key))

    hold_marsh_weights[key] = results_dict["Scaled Weights"]
    hold_unscaled_weights[key] = results_dict["Unscaled Weights"]
    hold_marsh_regularizors[key] = results_dict["Scaled regularizors"]
    hold_marsh_weight_certainty[key] = results_dict["# Well Determined Weights"]
    hold_prediction_certainty[key] = results_dict["Standard Deviations of Predictions"]
    hold_intercept[key] = results_dict["Unscaled Intercepts"]

# Make a colormap so all each weight will have a specific color
colormap = {
'Soil Porewater Salinity (ppt)': '#DD8A8A',
'Average Height Dominant (cm)': '#137111',
'NDVI': '#0AFF06',
'TSS (mg/L)': '#8E6C02',
'Windspeed (m/s)': '#70ECE3',
'Tidal Amplitude (cm)': '#434F93',
'Avg. Flood Depth (cm)': '#087AFA',
'SAVI':  '#087AFD',
'90th Percentile Flood Depth (cm)': '#D000E1',
'10th Percentile Flood Depth (cm)': '#73ACAE',
'Std. Deviation Flood Depth (cm)': '#DE5100',
'Avg. Time Flooded (%)': '#970CBD',
'Flood Freq (Floods/yr)': '#EB0000',
'Log Distance to Water (km)': '#442929',
'Log Distance to River (km)': '#045F38',
'Log Distance to Ocean (km)': '#045F27'
}

for key in hold_marsh_weights:
    d = pd.DataFrame(hold_marsh_weights[key].mean().reset_index()).rename(columns={0: 'Means'})
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_ylabel("Relative Feature Importance")
    # my_cmap = plt.get_cmap("cool")
    # ax.bar(list(d['index']), list(d['Means']), color='Blue')
    ax.set_title(str(key) + " CRMS Stations", fontsize=21)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # sns.barplot(data=hold_marsh_weights[key], palette="Blues")
    palette_ls = []
    for weight in d['index']:
        palette_ls.append(colormap[weight])
    sns.barplot(list(d['index']), list(d['Means']), palette=palette_ls)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    # fig.savefig("D:\\Etienne\\PAPER_2023\\results_BLR\\" + str(key) +
    #             "_scaledX_nolog_boxplot_human.eps", format='eps',
    #             dpi=300,
    #             bbox_inches='tight')
    plt.show()

# Plot the distribution of weight parameters for the marsh runs
for key in hold_unscaled_weights:
    print("Unscaled Weights for " + str(key))
    print(hold_unscaled_weights[key].mean())
    sns.set_theme(style='white', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_ylabel("Rescaled Weight Coefficients")
    # matplotlib.rcParams['pdf.fonttype'] = 42
    ax.set_title(str(key) + " CRMS Stations")
    ax.axhline(0, ls='--')
    # if key != 'Saline':
    #     ax.axhline(0, ls='--')
    palette_ls = []
    for weight in hold_unscaled_weights[key].keys():
        palette_ls.append(colormap[weight])
    boxplot = sns.boxplot(data=hold_unscaled_weights[key], notch=True, showfliers=False, palette=palette_ls, width=0.4)
    funcs.wrap_labels(ax, 10)
    fig.subplots_adjust(bottom=0.3)
    # fig.savefig("D:\\Etienne\\PAPER_2023\\results_BLR\\" + str(
    #     key) + "_unscaledWeights_nolog_boxplot_human.eps", format='eps',
    #             dpi=300,
    #             bbox_inches='tight')
    plt.show()


# Plot the distribution of the eff_reg parameter for each run
eff_reg_df = pd.DataFrame(hold_marsh_regularizors)
sns.set_theme(style='white', font_scale=1)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Learned Effective Regularization Parameters')
sns.boxplot(data=eff_reg_df, notch=True, showfliers=False, palette="YlOrBr")
funcs.wrap_labels(ax, 10)
# fig.savefig("D:\\Etienne\\PAPER_2023\\results_BLR\\regularization_scaledX_nolog_boxplot_human.eps",
#             format='eps',
#             dpi=300,
#             bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of parameters for each run
certainty_df = pd.DataFrame(hold_marsh_weight_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Calculated Number of Well Determined Parameters')
sns.boxplot(data=certainty_df, notch=True, showfliers=False, palette="Blues")
funcs.wrap_labels(ax, 10)
# fig.savefig("__ENTER LOCATION TO SAVE__",
#             format='eps',
#             dpi=300,
#             bbox_inches='tight')
plt.show()



# Plot the distribution calculated intercepts
intercept_df = pd.DataFrame(hold_intercept)
sns.set_theme(style='white', rc={'figure.dpi': 147}, font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Intercepts [Unscaled]:')
ax.axhline(0, ls='--')
sns.boxplot(data=intercept_df, notch=True, showfliers=False, palette="coolwarm")
funcs.wrap_labels(ax, 10)
# fig.savefig("__ENTER LOCATION TO SAVE__", dpi=300,
#             format='eps',
#             bbox_inches='tight')
plt.show()


# Plot the distribution of the certainty of predictions for each run
pred_certainty_df = pd.DataFrame(hold_prediction_certainty)
sns.set_theme(style='white', rc={'figure.dpi': 147},
              font_scale=0.7)
fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams['pdf.fonttype'] = 42
ax.set_title('Distribution of Bayesian Uncertainty in Predictions')
sns.boxplot(data=pred_certainty_df, notch=True, showfliers=False, palette="Reds")
funcs.wrap_labels(ax, 10)
# fig.savefig("__ENTER LOCATION TO SAVE__",
#             dpi=300, format='eps',
#             bbox_inches='tight')
plt.show()

