import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fall2021 = pd.read_csv(r"D:\Etienne\PAPER_2023\explore_NDVI_biomass\DeltaX_Aboveground_Biomass_Necromass_Fall2021.csv",
                       encoding="unicode_escape")
spring2021 = pd.read_csv(r"D:\Etienne\PAPER_2023\explore_NDVI_biomass\DeltaX_Aboveground_Biomass_Necromass_Spring2021.csv",
                         encoding="unicode_escape")
df = pd.read_csv(r"D:\Etienne\fall2022\agu_data\results\AGU_dataset.csv", encoding="unicode_escape")

print("NDVI CRMS0399: ", df[df['Unnamed: 0'] == 'CRMS0399']['NDVI'])
print("NDVI CRMS0322: ", df[df['Unnamed: 0'] == 'CRMS0322']['NDVI'])
print("NDVI CRMS0399: ", df[df['Unnamed: 0'] == 'CRMS0294']['NDVI'])
print("NDVI CRMS0399: ", df[df['Unnamed: 0'] == 'CRMS0396']['NDVI'])
print("NDVI CRMS0399: ", df[df['Unnamed: 0'] == 'CRMS0421']['NDVI'])

# NEXT: Use the CRMS aboveground biomass dataset instead, ( easier to work with and more data etc )
# Then do an analysis to correlate the NDVI to aboveground biomass of specific species...
agb = pd.read_csv(r"D:\Etienne\PAPER_2023\explore_NDVI_biomass\CRMS_Biomass\CRMS_Biomass.csv",
                  encoding="unicode_escape")

## Combine the agb dataset with accretion to see how it relates
agb['Simple site'] = [i[:8] for i in agb['Station ID']]
agb_gb = agb.groupby("Simple site").agg({'Aboveground Live Biomass (g/m2)': 'mean',
                                         'Common Name as Currently Recognized': lambda x: x.mode()})
agb_gb = agb_gb.dropna()
agb_gb['Common Name as Currently Recognized'] = [x[0] if type(x) != str else x for x in
                                                 agb_gb['Common Name as Currently Recognized']]
new = pd.concat([agb_gb, df.set_index("Unnamed: 0")], axis=1)

# 1. There is no relationship between NDVI and Organic Matter in the soil
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(ax=ax, data=new, y='NDVI', hue='Aboveground Live Biomass (g/m2)', x='Organic Matter (%)',
                size='Aboveground Live Biomass (g/m2)')
ax.set_xlabel('Organic Matter (%)', fontsize=21)
ax.set_ylabel("NDVI", fontsize=21)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Aboveground Live Biomass (g/m2)', title_fontsize=18)
plt.show()
fig.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_org_scatterplot.eps",
            dpi=300, format="eps")

# 2. Aboveground biomass seems to have a slight negative trend with NDVI for some reason
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(ax=ax1, data=new, y='NDVI', x='Aboveground Live Biomass (g/m2)', hue='Organic Matter (%)',
                size='Organic Matter (%)')
ax1.set_xlabel('Aboveground Live Biomass (g/m2)', fontsize=21)
ax1.set_ylabel("NDVI", fontsize=21)
ax1.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Organic Matter (%)', title_fontsize=18)
plt.show()
fig1.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_biomass_scatterplot.eps",
            dpi=300, format="eps")

# 3. Flooding influences the NDVI signiture / recording
# State that three might be a tidal range dependece of vegetation species that effects the NDVI signiture
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=new, x='90th Percentile Flood Depth (cm)', y='NDVI', hue='Tidal Amplitude (cm)',
                size='Tidal Amplitude (cm)')
ax2.set_xlabel("90th Percentile Flood Depth (cm)", fontsize=21)
ax2.set_ylabel("NDVI", fontsize=21)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Tidal Amplitude (cm)', title_fontsize=18)
plt.show()
fig2.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_flooddepth_tides_scatterplot.eps",
            dpi=300, format="eps")

# State that this is not related to different flood inundation times because the highest inundations are on the
# shallower NDVI slope
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='90th Percentile Flood Depth (cm)', y='NDVI', hue='Avg. Time Flooded (%)',
                size='Avg. Time Flooded (%)')
ax3.set_xlabel("90th Percentile Flood Depth (cm)", fontsize=21)
ax3.set_ylabel("NDVI", fontsize=21)
ax3.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Avg. Time Flooded (%)', title_fontsize=18)
plt.show()
fig3.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_flooddepth_floodtime_scatterplot.eps",
            dpi=300, format="eps")

# State there are transitions across marsh communities (supporting vegetation species arg) in the stteper negative
# relationship
palette = {'Brackish': '#ADD8E6', 'Saline': '#032180', 'Intermediate': '#5DC069', 'Freshwater': '#006E0D'}
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='90th Percentile Flood Depth (cm)', y='NDVI', hue='Community',
                palette=palette)
ax4.set_xlabel("90th Percentile Flood Depth (cm)", fontsize=21)
ax4.set_ylabel("NDVI", fontsize=21)
ax4.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Community', title_fontsize=18)
plt.show()
fig4.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_flooddepth_community_scatterplot.eps",
            dpi=300, format="eps")

# State that accretion rate is highest where NDVI is highest and lowest
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=new, x='90th Percentile Flood Depth (cm)', y='NDVI', hue='Accretion Rate (mm/yr)',
                size='Accretion Rate (mm/yr)')
ax5.set_xlabel("90th Percentile Flood Depth (cm)", fontsize=21)
ax5.set_ylabel("NDVI", fontsize=21)
ax5.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Accretion Rate (mm/yr)', title_fontsize=18)
plt.show()
fig5.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_flooddepth_accretion_scatterplot.eps",
            dpi=300, format="eps")

# # State that maybe it is related to salinity gradient
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=new, x='90th Percentile Flood Depth (cm)', y='NDVI', hue='Soil Porewater Salinity (ppt)',
                size='Soil Porewater Salinity (ppt)')
ax2.set_xlabel("90th Percentile Flood Depth (cm)", fontsize=21)
ax2.set_ylabel("NDVI", fontsize=21)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.legend(fontsize=14, title='Soil Porewater Salinity (ppt)', title_fontsize=18)
plt.show()
fig2.savefig("D:\\Etienne\\PAPER_2023\\data_vis\\ndvi_flooddepth_salinity_scatterplot.eps",
            dpi=300, format="eps")



