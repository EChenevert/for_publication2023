import glob
import numpy as np
import pandas as pd


# DISCLAIMER:
# Since the full hourly hydrologic dataset is too large to host on GitHub, this code will not run if one downloads it.
# However, one could use it as a guide to my workflow in preprocessing the hourly hydrologic data.
def split_by_intrinsic_var(df, var_str):
    """
    Split the DataFrame 'df' into multiple DataFrames based on an identifying categorical variable given by the user
    through a column named 'var_str'.
    @params:
        df (DataFrame): The original DataFrame to be split.
        var_str (str): The column name of the variable used for splitting.
    @returns:
        list: A list of DataFrames, each corresponding to a distinct value of the specified variable.
    """
    unq_vals = df[var_str].unique().tolist()
    hold_splits = []
    for value in unq_vals:
        hold_splits.append(df.groupby(var_str).get_group(value).reset_index().drop('index', axis=1))
    return hold_splits

def save_to_folder(folder_path, df_list, group_col):
    """
    Save a list of grouped DataFrames into separate CSV files based on the value of 'group_col' column, which
    correponds to a categorical variable. Each value in 'group_col' should be the same.

    @params:
        folder_path (str): Path to a folder where CSV files will be saved. Use \\ path separation if in Windows.
        df_list (list): A list of DataFrames, each corresponding to a different group.
        group_col (str): The column name used for grouping the DataFrames.
    @returns:
        None: Exports CSV files to the specified folder.
    """
    folder_path = folder_path
    for df in df_list:
        group_str = df[group_col][0]
        df.to_csv(folder_path+"\\"+group_str+".csv")


for i in range(0, 9):
    idx = str(i)
    # Reads data stored in file specified by the value in 'i' in local computer.
    floodingdf = pd.read_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx + ".csv",
                             encoding="unicode_escape")[['Station ID', 'Adjusted Water Elevation to Marsh (ft)',
                                      'Date (mm/dd/yyyy)']]

    # Make the simple site name. Leads to common grounds for identifying CRMS stations
    floodingdf['Simple site'] = [i[:8] for i in floodingdf['Station ID']]
    floodingdf = floodingdf.drop('Station ID', axis=1)
    # Split the DataFrame into multiple DataFrames based on the CRMS station name
    split_bysite_ls = split_by_intrinsic_var(floodingdf, "Simple site")
    # Save the separated DataFrames locally
    save_to_folder("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx, split_bysite_ls,
                   "Simple site")
    # Re-Load the DataFrames back in
    path = "D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\export-" + idx
    files = glob.glob(path + "/*.csv")

    # checking all the csv files in the specified path
    listdfs = []
    for filename in files:
        df = pd.read_csv(filename, encoding="unicode_escape")
        listdfs.append(df)
    # Create containers to hold processed data
    dictdf = {}
    arraydf = {}
    flooddepthdf = {}
    # Loop through the CSV files in the folder and read them back into DataFrames
    for d in listdfs:
        dfname = d['Simple site'][0]  # name of CRMS station used to name dataframe and associated features
        dictdf[dfname] = d.dropna().reset_index()
        # Calculating flood frequency and flood depths
        if len(dictdf[dfname]) > 0:  # Check that DataFrame has enough values in it
            wlarray = dictdf[dfname]['Adjusted Water Elevation to Marsh (ft)'].to_numpy()  # hold water elevation relative to marsh elevation
            dictdf[dfname]['date in datetime'] = pd.to_datetime(dictdf[dfname]['Date (mm/dd/yyyy)'], format='%m/%d/%Y')
            start = dictdf[dfname]['date in datetime'][0]  # Identify the date of first recorded water elevation
            end = dictdf[dfname]['date in datetime'][len(dictdf[dfname]) - 1]  # Identify date of the last recorded water elevation
            timedays = end - start  # Calculate the time in days of monitoring
            timedays = timedays.days
            decimalyears = timedays / 365  # Convert to decimal years for calculating floods/year
            arraydf[dfname] = np.zeros((len(wlarray),))  # initialize an array of zeros to hold the flood counts
            flooddepthdf[dfname] = np.zeros((len(wlarray,)))
            # Begin counting the number of floods and their depths that occur throughout the stations monitoring history
            for i in range(len(wlarray)):
                # flood depth
                if wlarray[i] > 0:  # Check if water level is greater than the marsh elevation (this means "flood").
                    flooddepthdf[dfname][i] = wlarray[i]  # add the depth of the flood to the flood depth array
                # flood frequency
                if (wlarray[i] <= 0 and wlarray[i - 1] > 0) or (wlarray[i] > 0 and wlarray[i - 1] <= 0):
                    if decimalyears > 0:
                        arraydf[dfname][
                            i] = 0.5 / decimalyears  # 0.5 cuz I will be appending every time a flood comes in and then out (so i divide)
                    else:
                        arraydf[dfname][i] = 0.5

        npcolstack_floodFreq = {}
        npcolstack_floodDepth = {}
        for key in arraydf:
            npcolstack_floodFreq[key] = np.column_stack((dictdf[key]['Simple site'].to_numpy(), arraydf[key]))
        for key in flooddepthdf:
            npcolstack_floodDepth[key] = np.column_stack((dictdf[key]['Simple site'].to_numpy(), flooddepthdf[key],
                                                      np.full(shape=len(flooddepthdf[key]),  # Find the 90th percentile of flood depth
                                                              fill_value=np.percentile(a=flooddepthdf[key], q=90)),
                                                      np.full(shape=len(flooddepthdf[key]),  # Find the 10th percentile of flood depth
                                                              fill_value=np.percentile(a=flooddepthdf[key], q=10)),
                                                      np.full(shape=len(flooddepthdf[key]),  # find the standard deviation of flood depth
                                                              fill_value=np.std(flooddepthdf[key]))
                                                      ))

        stackeddf_floodFreq = {}
        stackeddf_floodDepth = {}
        # Average all values into a single dataframe
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

        # Export to local computer for use in ML experiments
        check_freq = pd.concat(stackeddf_floodFreq.values(), ignore_index=True)
        check_freq.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\floodingsplits\\floodFrequencySitePerYear-" + idx +
                      ".csv")
        check_depth = pd.concat(stackeddf_floodDepth.values(), ignore_index=True)
        check_depth.to_csv("D:\\Etienne\\PAPER_2023\\CRMS_Continuous_Hydrographic\\flooddepthsplits\\floodDepthSitePerYear-"
                       + idx + ".csv")

