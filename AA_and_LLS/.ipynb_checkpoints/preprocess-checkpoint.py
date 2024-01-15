import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import glob
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import pandas as pd
import json
import cftime
#from itertools import product
from cftime import DatetimeNoLeap
#import Gridding
#from nc_processing import *
#from JASMIN_utils import *
#from analysis import * 
#from plotting import *
#import esmvalcore.preprocessor
import xesmf as xe
import warnings
#%matplotlib inline
#import seaborn as sns
#sns.set()
from tqdm import tqdm
#from scipy.stats import linregress
from nc_processing import calc_spatial_mean
from xmip.preprocessing import rename_cmip6



def read_in(dir, plev_val=None, rename=None):
    files = []
    for x in os.listdir(dir):
        files.append(dir + x)
    ds = rename_cmip6(xr.open_mfdataset(files))
    
    if plev_val:
        ds = ds.sel(plev=plev_val, method="nearest", tolerance=1000)

    if rename:
        ds = ds.rename(rename)

    return ds


### SETTINGS

arctic_bnd = 66
high_arctic_bnd = 75
tropics_bnds = [-30, 30]

### cell makes a dataframe containing paths to air temp data (below is just 1st ens mem)
def get_ta_tas_dirs(exps = ["historical", "ssp245"]):
    
    #exps = ['ssp245']
    #exps = ['ssp126']
    var_path = "Amon/ta"
    #var_path = "SImon/siconca"
    dfs = []
    
    for experiment in exps:
        dirs = []
        if experiment == "historical":
            exp_set = "CMIP"
        else:
            exp_set = "ScenarioMIP"
        for x in glob.glob('/badc/cmip6/data/CMIP6/{es}/*/*/{e}/r1i*/{v}/*/latest/'.format(es=exp_set, e=experiment, v=var_path)):
            dirs.append(x)
        model = []
        ensemble_member = []
        for dir in dirs:
            model.append(dir.split('/')[7])
            ensemble_member.append(dir.split('/')[9])
            
        #dirs.reverse()
        print(len(dirs))
        df = pd.DataFrame({'ta_dirs': dirs, 
                           'Model': model,
                           'Ensemble_member': ensemble_member})
        df['Experiment']=experiment
        df['tas_dirs'] = df['ta_dirs'].str.replace('Amon/ta', 'Amon/tas')
        for dir in df['tas_dirs']:
            if not os.path.isdir(dir):
                print('no tas path: {}'.format(dir))
        
        
        rows_to_drop = []
        for ind in df.index:
            dir = df['tas_dirs'][ind]
            if not os.path.isdir(dir):
                rows_to_drop.append(ind)
        
        df.drop(rows_to_drop, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)  
        
        #check for any remaning errors:
        for dir in df['tas_dirs']:
            if not os.path.isdir(dir):
                print('drop failed, error on: {}'.format(dir))
        dfs.append(df)
    
    DF = pd.concat(dfs).reset_index()
    print(len(DF))
    DF = DF.drop_duplicates(subset=['Model', 'Experiment'])
    print(len(DF))
    
    return DF
    
DF = get_ta_tas_dirs(exps = ["historical", "ssp245"])
models =DF['Model'].unique()
exps = DF['Experiment'].unique()

## only run over models with both historical and ssp245:
mods_ssp245 = DF[DF['Experiment']=='ssp245']['Model'].unique()
mods_hist = DF[DF['Experiment']=='historical']['Model'].unique()
mods_to_run = [x for x in mods_hist if x in mods_ssp245]
print(len(mods_to_run))

### the functions below are used to read in ta and tas from a 
### dir, and process into DJF. If multiple exps are used,
### they must be contiguous in time

def get_ds_mod(mod, 
               exps = ['historical', 'ssp245'],
               plev=85000):
                   
    """ returns a dataset containing both tas and ta
    (at a given pressure level), concated on time into
    one 1850-2100 timeseries. keep spatial dims for now """

    dss = []
    for exp in exps:
        print(exp)
        dir_ta = DF[DF['Model']==mod][DF['Experiment']==exp]['ta_dirs'].item()
        dir_tas = DF[DF['Model']==mod][DF['Experiment']==exp]['tas_dirs'].item()
        
        ds_ta = read_in(dir_ta, plev_val=plev, rename={'ta':'ta_850'}) 
        ds_tas = read_in(dir_tas)
        
        ds = xr.merge([ds_ta, ds_tas])
        dss.append(ds)
    DS = xr.concat(dss, dim='time')
    ## make sure no duplicate times from any pesky runs with 
    ## overlapping ranges (looking at you FGOALS..)
    _, index = np.unique(DS['time'], return_index=True)
    DS = DS.isel(time=index)
    return DS

def get_ds_DJF(ds):
    # the dataset below is quarterly seasonal averages by year, labeled by start month
    DS_quarterly = ds.resample(time='QS-DEC').mean(dim="time")
    
    # selecting 12 is actually selecting the 3month average for DJF
    # note that here, the data is labelled by the year in which December happens, not January.
    DS_DJF = DS_quarterly.sel(time=DS_quarterly.time.dt.month==12) 
    
    # now label by year (at this point, still based on december labelling)
    DS_DJF = DS_DJF.groupby(DS_DJF.time.dt.year).mean(dim='time')
    
    # switch to january labelling
    DS_DJF = DS_DJF.assign_coords(year=(DS_DJF.year + 1))
    
    # drop the incomplete years 1850 (just jan-feb) and 2100 (just dec)
    DS_DJF = DS_DJF.sel(year=DS_DJF.year>1850)
    DS_DJF = DS_DJF.sel(year=DS_DJF.year<2101)
    return DS_DJF
    



def flatten_to_timeseries(ds):
    """ take input spatial ds for ta_850 and tas, and replace 
    with output ds made up of a set of time-series data,
    each of which is some spatial mean over a domain """

    global_mean_tas = calc_spatial_mean(ds.tas, 
                                        lon_name="x", lat_name="y").rename(
                                        'Global_tas')

    arctic_mean_tas = calc_spatial_mean(ds.sel(y=slice(arctic_bnd, 91)).tas,
                                        lon_name="x", lat_name="y").rename(
                                        'Arctic_{}_tas'.format(str(arctic_bnd)))
    
    LLS_high_arctic_mean = calc_spatial_mean(ds.sel(y=slice(high_arctic_bnd, 91)).LLS,
                                        lon_name="x", lat_name="y").rename(
                                        'LLS_{}'.format(str(high_arctic_bnd)))
    
    tropics_mean_tas = calc_spatial_mean(ds.sel(y=slice(tropics_bnds[0], tropics_bnds[1])).tas,
                                        lon_name="x", lat_name="y").rename(
                                        'Tropics_{a}_{b}_tas'.format(
                                            a=str(tropics_bnds[0]),
                                            b=str(tropics_bnds[1])))
    out_ds = xr.merge([global_mean_tas, 
                       arctic_mean_tas,
                       LLS_high_arctic_mean,
                       tropics_mean_tas
                      ])
    return out_ds
    
def main(model):
    ds = get_ds_mod(mod=model)
    ds_DJF = get_ds_DJF(ds)
    ds_DJF['LLS'] = ds_DJF['ta_850'] - ds_DJF['tas']
    ds_DJF_ts = flatten_to_timeseries(ds_DJF)
    ds_DJF_ts['Model'] = model
    return ds_DJF_ts
    
## this runs for a while, maybe an hour or two - might be worth
## running on sci server instead..

ds_list = []

for model in tqdm(mods_to_run):
    ds_DJF_ts = main(model)
    ds_DJF_ts.to_netcdf('int_ncs/{}.nc'.format(model))
    ds_list.append(ds_DJF_ts)
    DS_out = xr.concat(ds_list, dim='Model')
    DS_out.to_netcdf('All_mods_ts.nc')
