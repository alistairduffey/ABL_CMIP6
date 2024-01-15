This is a standalone piece of analysis as part of the Arctic winter boundary layer in CMIP6 project. 

In this folder, we examine the winter-time relationships between arctic surface temperature, arctic temperature aloft, and arctic warming (amplification). 


Aims to answer for CMIP6 models: 

How does LLS change over time? 

How does AA change over time? 

LLS is strongly negatively correlated with arctic surface temperature in a given model on short timescales (hourly). Does this also hold on long timescales (decadal)?

Is there a correlation between LLS and AA across the MM ensemble, if not why not? How does this change depending on time-periods used. 

.......


processing.py creates

xarray datasets of:
* Global mean temp
* Arctic mean temp (>60N)
* Arctic mean temp (>66N)
* Arctic mean temp (>70N)
* Tropics temp (-30, 30)
* LLS (>75N) (maybe ocean masked?)
* LLs (>80N) (maybe ocean masked?)

..all annualy-resolved, DJF-only, 1850 - 2100, under historical then SSP245

Outputted to .nc files in /int_ncs  

nc_processing.py contains some standard utils. 
plotting_v2.ipynb plots the outputs and saves figs to /Figures


