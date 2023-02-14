import glob
import pandas as pd

### --- Working on the flooding frequency compilation ---###
path1 = "D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\floodingsplits\\"
files1 = glob.glob(path1 + "/*.csv")

df1 = pd.read_csv(files1[0], encoding="unicode_escape")
for i in range(1, len(files1)):
    add1 = pd.read_csv(files1[i], encoding="unicode_escape")
    df1 = pd.concat([df1, add1])

final_floodfreq = df1.groupby(['Simple site']).median()
final_floodfreq.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\floodingsplits\\final_floodfreq.csv")


### --- Working on the flood depth vars compilation --- ###
path2 = "D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\flooddepthsplits\\"
files2 = glob.glob(path2 + "/*.csv")

df2 = pd.read_csv(files2[0], encoding="unicode_escape")
for i in range(1, len(files2)):
    add2 = pd.read_csv(files2[i], encoding="unicode_escape")
    df2 = pd.concat([df2, add2])

final_flooddepth = df2.groupby(['Simple site']).median()
final_flooddepth.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\flooddepthsplits\\final_flooddepths.csv")
