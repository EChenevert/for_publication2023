import glob

import numpy as np
import pandas as pd


def split_by_intrinsic_var(df, var_str):
    unq_vals = df[var_str].unique().tolist()
    hold_splits = []
    for value in unq_vals:
        hold_splits.append(df.groupby(var_str).get_group(value).reset_index().drop('index', axis=1))
    return hold_splits

def save_to_folder(folder_path, df_list, group_col):
    """

    :param folder_path: path to a folder. Must have \\ path separation if in windows
    :param df_list: list of grouped dataframes
    :return: None, Exports files to place on computer
    """
    folder_path = folder_path
    for df in df_list:
        group_str = df[group_col][0]
        df.to_csv(folder_path+"\\"+group_str+".csv")


for i in range(0, 9):
    idx = str(i)
    floodingdf = pd.read_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx + ".csv",
                             encoding="unicode_escape")[['Station ID', 'Adjusted Water Elevation to Marsh (ft)',
                                      'Date (mm/dd/yyyy)']]

    # Make the simple site site name
    floodingdf['Simple site'] = [i[:8] for i in floodingdf['Station ID']]
    floodingdf = floodingdf.drop('Station ID', axis=1)

    split_bysite_ls = split_by_intrinsic_var(floodingdf, "Simple site")
    save_to_folder("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx, split_bysite_ls,
                   "Simple site")

    ### --- Re-Load them back in ---###
    path = "D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx
    files = glob.glob(path + "/*.csv")

    # checking all the csv files in the
    # specified path
    listdfs = []
    for filename in files:
        df = pd.read_csv(filename, encoding="unicode_escape")
        listdfs.append(df)

    dictdf = {}
    arraydf = {}
    flooddepthdf = {}
    for d in listdfs:
        dfname = d['Simple site'][0]  # name of marsh type used to name dataframe and associated features
        dictdf[dfname] = d.dropna().reset_index()
        if len(dictdf[dfname]) > 0:

            wlarray = dictdf[dfname]['Adjusted Water Elevation to Marsh (ft)'].to_numpy()
            dictdf[dfname]['date in datetime'] = pd.to_datetime(dictdf[dfname]['Date (mm/dd/yyyy)'], format='%m/%d/%Y')
            start = dictdf[dfname]['date in datetime'][0]
            end = dictdf[dfname]['date in datetime'][len(dictdf[dfname]) - 1]
            timedays = end - start
            timedays = timedays.days
            decimalyears = timedays / 365
            arraydf[dfname] = np.zeros((len(wlarray),))
            flooddepthdf[dfname] = np.zeros((len(wlarray, )))
            for i in range(len(wlarray)):
                if wlarray[i] > 0:
                    flooddepthdf[dfname][i] = wlarray[i]
                if (wlarray[i] <= 0 and wlarray[i - 1] > 0) or (wlarray[i] > 0 and wlarray[i - 1] <= 0):
                    if decimalyears > 0:
                        arraydf[dfname][
                            i] = 0.5 / decimalyears  # 0.5 cuz I will be appending every time a flood comes in and then out (so i divide)
                    else:
                        arraydf[dfname][i] = 0.5
            # else:
            #     arraydf[dfname][i] = 0.0

        npcolstack_floodFreq = {}
        npcolstack_floodDepth = {}
        for key in arraydf:
            npcolstack_floodFreq[key] = np.column_stack((dictdf[key]['Simple site'].to_numpy(), arraydf[key]))
        for key in flooddepthdf:
            npcolstack_floodDepth[key] = np.column_stack((dictdf[key]['Simple site'].to_numpy(), flooddepthdf[key],
                                                      np.full(shape=len(flooddepthdf[key]),
                                                              fill_value=np.percentile(a=flooddepthdf[key], q=90)),
                                                      np.full(shape=len(flooddepthdf[key]),
                                                              fill_value=np.percentile(a=flooddepthdf[key], q=10)),
                                                      np.full(shape=len(flooddepthdf[key]),
                                                              fill_value=np.std(flooddepthdf[key]))
                                                      ))

        stackeddf_floodFreq = {}
        stackeddf_floodDepth = {}
        for key in npcolstack_floodFreq:
            stackeddf_floodFreq[key] = pd.DataFrame(npcolstack_floodFreq[key],
                                                columns=['Simple site', 'Flood Freq (Floods/yr)']) \
                .groupby('Simple site').sum().reset_index()
        for key in npcolstack_floodDepth:
            stackeddf_floodDepth[key] = pd.DataFrame(npcolstack_floodDepth[key], columns=['Simple site',
                                                                                      'Avg. Flood Depth when Flooded (ft)',
                                                                                      '90th Percentile Flood Depth when Flooded (ft)',
                                                                                      '10th Percentile Flood Depth when Flooded (ft)',
                                                                                      'Std. Deviation Flood Depth when Flooded ']) \
                .groupby('Simple site').median().reset_index()

        check_freq = pd.concat(stackeddf_floodFreq.values(), ignore_index=True)
        check_freq.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\floodingsplits\\floodFrequencySitePerYear-" + idx +
                      ".csv")
        check_depth = pd.concat(stackeddf_floodDepth.values(), ignore_index=True)
        check_depth.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\flooddepthsplits\\floodDepthSitePerYear-"
                       + idx + ".csv")

