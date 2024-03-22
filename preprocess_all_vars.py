import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import json
import cftime
from itertools import product
from cftime import DatetimeNoLeap
import esmvalcore.preprocessor
import xesmf as xe
from xmip.preprocessing import promote_empty_dims, broadcast_lonlat, replace_x_y_nominal_lat_lon, rename_cmip6
from tqdm import tqdm
import dask

print('starting')



def read_in(dir, t_bnds, months, ocean = False, plev=False):
    files = []
    for x in os.listdir(dir): 
        if '_20' in x: #HACK - to prevent opening too many files and slowing down unneccesarily
            files.append(dir + x)
        elif '-2015' in x:
            files.append(dir + x)
        elif '-201412' in x:
            files.append(dir + x)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        if ocean:
            ds = replace_x_y_nominal_lat_lon(rename_cmip6(xr.open_mfdataset(files, chunks=50)))
        else: 
            #ds = rename_cmip6(xr.open_dataset(files[-1]))
            #ds = rename_cmip6(xr.open_mfdataset(files, parallel=True, chunks={"time": 50}))
            ds = rename_cmip6(xr.open_mfdataset(files, parallel=True, chunks=50))
    ds = ds.sel(time=slice(t_bnds[0], t_bnds[1]))
    ds = ds.where((ds['time.month'].isin(months)), drop=True)
    ds = ds.sel(y=slice(65,90))
    if plev:
        ds = ds.sel(plev=85000, method='nearest', tolerance=10)
    print(dir, ds['time.year'].min().values) #check that only selecting specific time files has worked ok
    return ds

def regrid(ds, target, method='bilinear'):
    ds.load()
    target.load()
    
    regridder = xe.Regridder(ds, target, method, periodic=True)
    out = regridder(ds)
    return out

def sea_ice_mask(in_ds, model, ensemble_member, t_bnds, months, conc_cutoff=95,
                 siconc_dir='/gws/nopw/j04/cpom/aduffey/siconc_regrids/out_siconca/',
                 reverse=False):
       
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        sea_ice = rename_cmip6(xr.open_mfdataset(siconc_dir+'*{m}*{e}*.nc'.format(
                                            m=model, e=ensemble_member)))

    sea_ice = sea_ice.sel(time=slice(t_bnds[0], t_bnds[1]))
    sea_ice = sea_ice.where((sea_ice['time.month'].isin(months)), drop=True)
    sea_ice = sea_ice.sel(y=slice(65,90))

    sea_ice = regrid(sea_ice, in_ds, method='bilinear')
    
    if reverse:
        try: 
            SI_mask = xr.where(sea_ice['siconc'] < conc_cutoff, True, False)
        except:
            SI_mask = xr.where(sea_ice['siconca'] < conc_cutoff, True, False)
    else:
        try:
            SI_mask = xr.where(sea_ice['siconc'] > conc_cutoff, True, False)
        except:
            SI_mask = xr.where(sea_ice['siconca'] > conc_cutoff, True, False)
    # resample up to needed time resolution
    SI_mask = SI_mask.reindex(time=in_ds['time'], method='nearest')

    masked_ds = in_ds.where(SI_mask)

    #    print('no siconc data in {}'.format(siconc_dir))
        
    return masked_ds

def resample(ds, freq, months):
    return ds.resample(time=freq).mean(dim='time').where((ds['time.month'].isin(months)), drop=True)

def process_inversions_with_wind(dir_ta, dir_tas, 
                       dir_rlds, dir_rlus, 
                       dir_uas, dir_vas,
                       dir_ua, dir_va, freq,
                       region, mod, ens):
    ## region must be 'sea_ice', 'open_ocean' or 'land'
    
    ds_ta = read_in(dir_ta, t_bnds=inversion_time_period, months=winter_months,
                    plev=True)
    ds_ta = resample(ds=ds_ta, freq=freq, months=winter_months)
    
    ds_tas = read_in(dir_tas, t_bnds=inversion_time_period, months=winter_months)
    ds_tas = resample(ds=ds_tas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_tas = ds_tas.resample(time=freq).mean(dim='time')

    ds_rlds = read_in(dir_rlds, t_bnds=inversion_time_period, months=winter_months)
    ds_rlds = resample(ds=ds_rlds, freq=freq, months=winter_months)
    
    #if not freq=='1D':
    #    ds_rlds = ds_rlds.resample(time=freq).mean(dim='time')

    ds_rlus = read_in(dir_rlus, t_bnds=inversion_time_period, months=winter_months)
    ds_rlus = resample(ds=ds_rlus, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_rlus = ds_rlus.resample(time=freq).mean(dim='time')

    ds_uas = read_in(dir_uas, t_bnds=inversion_time_period, months=winter_months)
    ds_uas = resample(ds=ds_uas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_uas = ds_uas.resample(time=freq).mean(dim='time')
    ds_uas = regrid(ds_uas, ds_tas)

    ds_vas = read_in(dir_vas, t_bnds=inversion_time_period, months=winter_months)
    ds_vas = resample(ds=ds_vas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_vas = ds_vas.resample(time=freq).mean(dim='time')
    ds_vas = regrid(ds_vas, ds_tas)

    ds_va = read_in(dir_va, t_bnds=inversion_time_period, months=winter_months, plev=True)
    ds_va = resample(ds=ds_va, freq=freq, months=winter_months)
    ds_va = regrid(ds_va, ds_ta)

    ds_ua = read_in(dir_ua, t_bnds=inversion_time_period, months=winter_months, plev=True)
    ds_ua = resample(ds=ds_ua, freq=freq, months=winter_months)
    ds_ua = regrid(ds_ua, ds_ta)


    ds_tas['LLS'] = ds_ta['ta'] -  ds_tas['tas']
    ds_tas = ds_tas.assign(uas=(['time', 'y', 'x'], ds_uas['uas'].values))
    ds_tas = ds_tas.assign(vas=(['time', 'y', 'x'], ds_vas['vas'].values))
    #ds_tas = ds_tas.assign(ua=(['time', 'y', 'x'], ds_ua['ua'].values))
    #ds_tas = ds_tas.assign(va=(['time', 'y', 'x'], ds_va['va'].values))
    ds_tas['rlns'] = ds_rlds['rlds'] - ds_rlus['rlus']

    ds_tas = xr.merge([ds_tas['LLS'], ds_tas['tas'], ds_tas['rlns'], 
                       ds_tas['uas'], ds_tas['vas'], ds_ua['ua'], ds_va['va']],
                       join='outer')
    
    print('made combined ds')
    
    if region == 'sea_ice':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months)
    elif region == 'open_ocean':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months, reverse=True)
    elif region == 'land':
        print('need to implement land masking')
        breakhere
    print('sea ice masking complete')
    
    #ds_tas['inv_strength'].isel(time=0).plot()
    
    ds_stacked_inv = ds_tas['LLS'].stack(z=('x', 'y','time'))
    ds_stacked_rlns = ds_tas['rlns'].stack(z=('x', 'y','time'))
    ds_stacked_tas = ds_tas['tas'].stack(z=('x', 'y','time'))
    ds_stacked_vas = ds_tas['vas'].stack(z=('x', 'y','time'))
    ds_stacked_uas = ds_tas['uas'].stack(z=('x', 'y','time'))
    ds_stacked_ua = ds_tas['ua'].stack(z=('x', 'y','time'))
    ds_stacked_va = ds_tas['va'].stack(z=('x', 'y','time'))
    
    
    print('stacked')
    out_df = pd.DataFrame({'LLS': ds_stacked_inv.values,
                           'rlns': ds_stacked_rlns.values,
                           'tas': ds_stacked_tas.values,
                           'vas': ds_stacked_vas.values,
                           'uas': ds_stacked_uas.values,
                           'va': ds_stacked_va.values,
                           'ua': ds_stacked_ua.values,
                          })
    
    out_df = out_df.dropna()
    
    # we don't need all 10s of millions of data points, so take a random subsample:
    out_df = out_df.sample(frac=0.01)

    #save
    out_df.to_csv("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens))
    #np.savetxt("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens), 
    #           out, delimiter=",")
    print(region +' done')


def process_inversions_with_wind_and_hfss(dir_ta, dir_tas, 
                       dir_rlds, dir_rlus, 
                       dir_uas, dir_vas,
                       dir_hfss,
                       region, mod, ens):
    ## region must be 'sea_ice', 'open_ocean' or 'land'
    freq = '6H'
    ds_ta = read_in(dir_ta, t_bnds=inversion_time_period, months=winter_months,
                    plev=True)
    ds_ta = resample(ds=ds_ta, freq=freq, months=winter_months)
    
    ds_tas = read_in(dir_tas, t_bnds=inversion_time_period, months=winter_months)
    ds_tas = resample(ds=ds_tas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_tas = ds_tas.resample(time=freq).mean(dim='time')

    ds_rlds = read_in(dir_rlds, t_bnds=inversion_time_period, months=winter_months)
    ds_rlds = resample(ds=ds_rlds, freq=freq, months=winter_months)
    
    #if not freq=='1D':
    #    ds_rlds = ds_rlds.resample(time=freq).mean(dim='time')

    ds_rlus = read_in(dir_rlus, t_bnds=inversion_time_period, months=winter_months)
    ds_rlus = resample(ds=ds_rlus, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_rlus = ds_rlus.resample(time=freq).mean(dim='time')

    ds_uas = read_in(dir_uas, t_bnds=inversion_time_period, months=winter_months)
    ds_uas = resample(ds=ds_uas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_uas = ds_uas.resample(time=freq).mean(dim='time')
    ds_uas = regrid(ds_uas, ds_tas)

    ds_vas = read_in(dir_vas, t_bnds=inversion_time_period, months=winter_months)
    ds_vas = resample(ds=ds_vas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_vas = ds_vas.resample(time=freq).mean(dim='time')
    ds_vas = regrid(ds_vas, ds_tas)

    ds_hfss = read_in(dir_hfss, t_bnds=inversion_time_period, months=winter_months)
    ds_hfss = resample(ds=ds_hfss, freq=freq, months=winter_months)
    ds_hfss = regrid(ds_hfss, ds_tas)

    ds_tas['LLS'] = ds_ta['ta'] -  ds_tas['tas']
    ds_tas = ds_tas.assign(uas=(['time', 'y', 'x'], ds_uas['uas'].values))
    ds_tas = ds_tas.assign(vas=(['time', 'y', 'x'], ds_vas['vas'].values))
    #ds_tas = ds_tas.assign(ua=(['time', 'y', 'x'], ds_ua['ua'].values))
    #ds_tas = ds_tas.assign(va=(['time', 'y', 'x'], ds_va['va'].values))
    ds_tas['rlns'] = ds_rlds['rlds'] - ds_rlus['rlus']
    

    ds_tas = xr.merge([ds_tas['LLS'], ds_tas['tas'], ds_tas['rlns'], 
                       ds_tas['uas'], ds_tas['vas'], ds_hfss['hfss']],
                       join='outer')
    
    print('made combined ds')
    
    if region == 'sea_ice':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months)
    elif region == 'open_ocean':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months, reverse=True)
    elif region == 'land':
        print('need to implement land masking')
        breakhere
    print('sea ice masking complete')
    
    #ds_tas['inv_strength'].isel(time=0).plot()
    
    ds_stacked_inv = ds_tas['LLS'].stack(z=('x', 'y','time'))
    ds_stacked_rlns = ds_tas['rlns'].stack(z=('x', 'y','time'))
    ds_stacked_tas = ds_tas['tas'].stack(z=('x', 'y','time'))
    ds_stacked_vas = ds_tas['vas'].stack(z=('x', 'y','time'))
    ds_stacked_uas = ds_tas['uas'].stack(z=('x', 'y','time'))
    ds_stacked_hfss = ds_tas['hfss'].stack(z=('x', 'y','time'))
    
    
    
    print('stacked')
    out_df = pd.DataFrame({'LLS': ds_stacked_inv.values,
                           'rlns': ds_stacked_rlns.values,
                           'tas': ds_stacked_tas.values,
                           'vas': ds_stacked_vas.values,
                           'uas': ds_stacked_uas.values,
                           'hfss': ds_stacked_hfss.values,
                          })
    
    out_df = out_df.dropna()
    
    # we don't need all 10s of millions of data points, so take a random subsample:
    out_df = out_df.sample(frac=0.01)

    #save
    out_df.to_csv("{r}_6hourly_all_vars/{m}_{e}.csv".format(r=region, m=mod, e=ens))
    #np.savetxt("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens), 
    #           out, delimiter=",")
    print(region +' done')


    
def process_inversions_without_wind(dir_ta, dir_tas, 
                       dir_rlds, dir_rlus, freq,
                       region, mod, ens):
    ## region must be 'sea_ice', 'open_ocean' or 'land'
    
    ds_ta = read_in(dir_ta, t_bnds=inversion_time_period, months=winter_months,
                    plev=True)
    ds_ta = resample(ds=ds_ta, freq=freq, months=winter_months)
    
    ds_tas = read_in(dir_tas, t_bnds=inversion_time_period, months=winter_months)
    ds_tas = resample(ds=ds_tas, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_tas = ds_tas.resample(time=freq).mean(dim='time')

    ds_rlds = read_in(dir_rlds, t_bnds=inversion_time_period, months=winter_months)
    ds_rlds = resample(ds=ds_rlds, freq=freq, months=winter_months)
    
    #if not freq=='1D':
    #    ds_rlds = ds_rlds.resample(time=freq).mean(dim='time')

    ds_rlus = read_in(dir_rlus, t_bnds=inversion_time_period, months=winter_months)
    ds_rlus = resample(ds=ds_rlus, freq=freq, months=winter_months)
    #if not freq=='1D':
    #    ds_rlus = ds_rlus.resample(time=freq).mean(dim='time')

    ds_tas['LLS'] = ds_ta['ta'] -  ds_tas['tas']
    ds_tas['rlns'] = ds_rlds['rlds'] - ds_rlus['rlus']

    ds_tas = xr.merge([ds_tas['LLS'], ds_tas['tas'], ds_tas['rlns']],
                       join='outer')
    
    print('made combined ds')
    
    if region == 'sea_ice':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months)
    elif region == 'open_ocean':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months, reverse=True)
    elif region == 'land':
        print('need to implement land masking')
        breakhere
    print('sea ice masking complete')
    
    #ds_tas['inv_strength'].isel(time=0).plot()
    
    ds_stacked_inv = ds_tas['LLS'].stack(z=('x', 'y','time'))
    ds_stacked_rlns = ds_tas['rlns'].stack(z=('x', 'y','time'))
    ds_stacked_tas = ds_tas['tas'].stack(z=('x', 'y','time'))
    
    
    print('stacked')
    out_df = pd.DataFrame({'LLS': ds_stacked_inv.values,
                           'rlns': ds_stacked_rlns.values,
                           'tas': ds_stacked_tas.values
                          })
    
    out_df = out_df.dropna()
    
    # we don't need all 10s of millions of data points, so take a random subsample:
    out_df = out_df.sample(frac=0.01)

    #save
    out_df.to_csv("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens))
    #np.savetxt("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens), 
    #           out, delimiter=",")
    print(region +' done')

def process_daily_inversions_and_cloud(dir_ta, dir_tas, 
                        dir_clivi, dir_clwvi,
                       region, mod, ens):
    ## region must be 'sea_ice', 'open_ocean' or 'land'
    freq = '1D'
    ds_ta = read_in(dir_ta, t_bnds=inversion_time_period, months=winter_months,
                    plev=True)
    ds_ta = resample(ds=ds_ta, freq=freq, months=winter_months)
    
    ds_tas = read_in(dir_tas, t_bnds=inversion_time_period, months=winter_months)
    ds_tas = resample(ds=ds_tas, freq=freq, months=winter_months)
    
    ds_clivi = read_in(dir_clivi, t_bnds=inversion_time_period, months=winter_months)
    ds_clivi = resample(ds=ds_clivi, freq=freq, months=winter_months)
    ds_clivi = regrid(ds_clivi, ds_tas)
    
    ds_clwvi = read_in(dir_clwvi, t_bnds=inversion_time_period, months=winter_months)
    ds_clwvi = resample(ds=ds_clwvi, freq=freq, months=winter_months)
    ds_clwvi = regrid(ds_clwvi, ds_tas)
    

    ds_tas['LLS'] = ds_ta['ta'] -  ds_tas['tas']
    
    ds_tas = xr.merge([ds_tas['LLS'], ds_tas['tas'], ds_clwvi['clwvi'], ds_clivi['clivi']],
                       join='outer')
    
    print('made combined ds')
    
    if region == 'sea_ice':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months)
    elif region == 'open_ocean':
        ds_tas = sea_ice_mask(ds_tas, mod, ens, 
                         t_bnds=inversion_time_period, 
                         months=winter_months, reverse=True)
    elif region == 'land':
        print('need to implement land masking')
        breakhere
    print('sea ice masking complete')
    
    #ds_tas['inv_strength'].isel(time=0).plot()
    
    ds_stacked_inv = ds_tas['LLS'].stack(z=('x', 'y','time'))
    ds_stacked_clivi = ds_tas['clivi'].stack(z=('x', 'y','time'))
    ds_stacked_clwvi = ds_tas['clwvi'].stack(z=('x', 'y','time'))
    ds_stacked_tas = ds_tas['tas'].stack(z=('x', 'y','time'))
    
    
    print('stacked')
    out_df = pd.DataFrame({'LLS': ds_stacked_inv.values,
                           'clivi': ds_stacked_clivi.values,
                           'clwvi': ds_stacked_clwvi.values,
                           'tas': ds_stacked_tas.values
                          })
    
    out_df = out_df.dropna()
    
    # we don't need all 10s of millions of data points, so take a random subsample:
    out_df = out_df.sample(frac=0.01)

    #save
    out_df.to_csv("{r}_daily_invs_and_clouds/{m}_{e}.csv".format(r=region, m=mod, e=ens))
    #np.savetxt("{r}_hourly_inversions/{m}_{e}.csv".format(r=region, m=mod, e=ens), 
    #           out, delimiter=",")
    print(region +' done')






inversion_time_period = ['2010', '2015']
winter_months = [11, 12, 1, 2, 3]



print('making dir lists')

### cell makes a dataframe containing paths to air temp data (below is just 1st ens mem)

exps = ["historical"]
#exps = ['ssp245']
#exps = ['ssp126']
dirs = []
var_path = "6hrPlevPt/ta"
#var_path = "SImon/siconca"
for experiment in exps:
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
                   'Ensemble_member': ensemble_member,
                   'freq': np.repeat('6H', len(dirs))})

#try an initial set of tables for all variables
df['tas_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/tas')
df['rlds_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/rlds')
df['rlus_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/rlus')
df['ua_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/ua')
df['va_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/va')
df['uas_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/uas')
df['vas_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/vas')
df['hfss_dirs'] = df['ta_dirs'].str.replace('6hrPlevPt/ta', '6hrPlevPt/hfss')



var_list = ['tas_dirs', 'hfss_dirs',
            'rlds_dirs', 'rlus_dirs', 
            'ua_dirs', 'va_dirs', 
            'uas_dirs', 'vas_dirs']

for var in var_list:
    for ind in df.index:
        dir = df[var][ind]
        if not os.path.isdir(dir):
            df[var][ind] = df[var][ind].replace('6hrPlevPt', '6hrPlev')
    
    for ind in df.index:
        dir = df[var][ind]
        if not os.path.isdir(dir):
            df[var][ind] = df[var][ind].replace('6hrPlev', '3hr')

    for ind in df.index:
        dir = df[var][ind]
        if not os.path.isdir(dir):
            for var_x in var_list: # use daily thoughout if need to go to it
                df[var_x][ind] = df[var_x][ind].replace('3hr', 'day')
                df[var_x][ind] = df[var_x][ind].replace('6hrPlevPt', 'day')
                df[var_x][ind] = df[var_x][ind].replace('6hrPlev', 'day')
            df['freq'][ind] = '1D'
        

    for ind in df.index:
        dir = df[var][ind]
        if not os.path.isdir(dir):
            print('no {s} path: {d}'.format(s=var, d=dir))
            df[var][ind] = 'nan: no sub-daily or daily data on CEDA archive'

vital_var_list = ['tas_dirs',
            'rlds_dirs', 'rlus_dirs']

for string in vital_var_list:
    rows_to_drop = []
    for ind in df.index:
        dir = df[string][ind]
        if not os.path.isdir(dir):
            rows_to_drop.append(ind)
    df.drop(rows_to_drop, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)  

    #check for any remaning errors:
    for dir in df[string]:
        if not os.path.isdir(dir):
            print('drop failed, ERROR on {s}: {d}'.format(s=string, d=dir))   
#df.to_csv('directories.csv')        
#print(len(df))

### now keep only the models with 6 hourly data:
resolutions = pd.read_csv('Highest_resolution_on_CEDA_for_vars.csv')
vars = ['tas', 'ta', 'hfss', 'vas', 'uas', 'sfcWind', 'rlds', 'rlus', 'ts', 'clivi', 'clwvi']
min_vars = ['tas', 'ta', 'hfss', 'vas', 'uas', 'rlds', 'rlus']
df_min = resolutions.drop(columns=[x for x in vars if x not in min_vars])
for var in min_vars:
    df_min = df_min[df_min[var].isin(['3hr', '6hrPlevPt'])]
print(len(df_min.Model.unique()))
mods_6hrly = df_min.Model.unique()


df = df[df['Model'].isin(mods_6hrly)].reset_index()



##########

print('now run')
### now run over all models:
failed = []
for i in tqdm(np.arange(0, len(df))):
    dir_ta = df['ta_dirs'][i]
    dir_tas = df['tas_dirs'][i]
    dir_rlds = df['rlds_dirs'][i]
    dir_rlus = df['rlus_dirs'][i]
    dir_uas = df['uas_dirs'][i]
    dir_vas = df['vas_dirs'][i]
    dir_hfss = df['hfss_dirs'][i]
    
    mod = df['Model'][i]
    ens = df['Ensemble_member'][i]
    freq = df['freq'][i]
    region = 'sea_ice'
    print(i)
    out_name = "{m}_{e}.csv".format(r=region, m=mod, e=ens)
    

    try:
        process_inversions_with_wind_and_hfss(dir_ta=dir_ta, dir_tas=dir_tas, 
                               dir_rlds=dir_rlds, dir_rlus=dir_rlus, 
                               dir_vas=dir_vas, dir_uas=dir_uas, dir_hfss=dir_hfss,
                               region=region, mod=mod, ens=ens)

    except:
   
        print('error on: ', mod)
        failed.append(mod+ens)

failed_df = pd.DataFrame({'Failed Models': failed})
failed_df.to_csv('Failed_models.csv') 
print(failed)


