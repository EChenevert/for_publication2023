import pandas as pd
import numpy as np
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

# ndvi_df = {'NDVI': [df[df['Unnamed: 0'] == 'CRMS0399']['NDVI'],
#                     df[df['Unnamed: 0'] == 'CRMS0322']['NDVI'],
#                     df[df['Unnamed: 0'] == 'CRMS0294']['NDVI'],
#                     df[df['Unnamed: 0'] == 'CRMS0396']['NDVI'],
#                     df[df['Unnamed: 0'] == 'CRMS0421']['NDVI']]}
# ndvi_df = pd.DataFrame(ndvi_df)
# plt.plot(ndvi_df['NDVI'])
# plt.show()

# test plots for fall2021 site CRMS0399 (Brackish to saline (from GEE map))
sns.boxplot(data=fall2021[fall2021['site_id'] == 'CRMS0399'],
            x='species',
            y='total_dry_mass', hue='vegetation_class')
plt.title('CRMS0399')
plt.show()
print("Total Mass:", fall2021[fall2021['site_id'] == 'CRMS0399']['total_dry_mass'].sum())

# test plots for fall2021 site CRMS0322 (saline (from GEE map))
sns.boxplot(data=fall2021[fall2021['site_id'] == 'CRMS0322'],
            x='species',
            y='total_dry_mass', hue='vegetation_class')
plt.title('CRMS0322')
plt.show()
print("Total Mass:", fall2021[fall2021['site_id'] == 'CRMS0322']['total_dry_mass'].sum())

# test plots for fall2021 site CRMS0322 (fresh (from GEE map))
sns.boxplot(data=fall2021[fall2021['site_id'] == 'CRMS0294'],
            x='species',
            y='total_dry_mass', hue='vegetation_class')
plt.title('CRMS0294')
plt.show()
print("Total Mass:", fall2021[fall2021['site_id'] == 'CRMS0294']['total_dry_mass'].sum())

# test plots for fall2021 site CRMS0322 (intermediate (from GEE map))
sns.boxplot(data=fall2021[fall2021['site_id'] == 'CRMS0396'],
            x='species',
            y='total_dry_mass', hue='vegetation_class')
plt.title('CRMS0396')
plt.show()
print("Total Mass:", fall2021[fall2021['site_id'] == 'CRMS0396']['total_dry_mass'].sum())

# test plots for fall2021 site CRMS0322 (strong saline (from GEE map))
sns.boxplot(data=fall2021[fall2021['site_id'] == 'CRMS0421'],
            x='species',
            y='total_dry_mass', hue='vegetation_class')
plt.title('CRMS0421')
plt.show()
print("Total Mass:", fall2021[fall2021['site_id'] == 'CRMS0421']['total_dry_mass'].sum())


