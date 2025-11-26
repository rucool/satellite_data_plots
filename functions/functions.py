import os
import pandas as pd
import numpy as np
import xarray as xr
from erddapy import ERDDAP
from datetime import datetime, timezone
import re
from cool_maps.download import get_bathymetry

def get_bathymetry_file_info(bathyDir):
    """
    Get all files in bathymetry file directory and initialize dataframe with domain boundaries and resolution
    """
    if not bathyDir or not os.path.isdir(bathyDir):
        bathyfiles = None
        return bathyfiles
    bathyfiles=os.listdir(bathyDir)
    bathyfiles={'filenames':bathyfiles}
    bathyfiles=pd.DataFrame(bathyfiles)
    for f in bathyfiles.index:
        bathyfiles['filenames'][f] = os.path.join(bathyDir, bathyfiles['filenames'][f])
    bathyfiles['minlon']=np.nan
    bathyfiles['maxlon']=np.nan
    bathyfiles['minlat']=np.nan
    bathyfiles['maxlat']=np.nan
    bathyfiles['resolution']=np.nan
    bathyfiles=bathyfiles.set_index('filenames')

    # loop through bathymetry files
    for f in bathyfiles.index:
        # get resolution from filename (should be at end of file name)
        x = f.split('_res_')
        x0 = x[0]
        x1 = x[1]
        x = x1.split('m.')
        if x[0][-1]=='k':
            bathyfiles['resolution'].loc[f] = float(x[0][:-1]*1000)
        else:
            bathyfiles['resolution'].loc[f] = float(x[0])
        # get lon/lat of upper right corner of domain from filename
        x = x0.split('_UR_')
        x0 = x[0]
        x1 = x[1]
        x = x1.split('_')
        for n in range(len(x)):
            if x[n][-1]=='S':
                bathyfiles['maxlat'].loc[f] = -float(x[n][:-1])
            elif x[n][-1]=='N':
                bathyfiles['maxlat'].loc[f] = float(x[n][:-1])
            elif x[n][-1] == 'W':
                bathyfiles['maxlon'].loc[f] = -float(x[n][:-1])
            elif x[n][-1]=='E':
                bathyfiles['maxlon'].loc[f] = float(x[n][:-1])
        # get lon/lat of lower left corner of domain from filename
        x = x0.split('_LL_')
        x0 = x[0]
        x1 = x[1]
        x = x1.split('_')
        for n in range(len(x)):
            if x[n][-1] == 'S':
                bathyfiles['minlat'].loc[f] = -float(x[n][:-1])
            elif x[n][-1] == 'N':
                bathyfiles['minlat'].loc[f] = float(x[n][:-1])
            elif x[n][-1] == 'W':
                bathyfiles['minlon'].loc[f] = -float(x[n][:-1])
            elif x[n][-1] == 'E':
                bathyfiles['minlon'].loc[f] = float(x[n][:-1])
    return bathyfiles


def standardize_var_names(xrDataset,
                          preferred_names=None):
    """
    Map dataset variable names to preferred variable names.
    """
    if type(preferred_names)!=dict:
        print(f'{preferred_names} not provided as dictionary')
        return 1
    for k in preferred_names.keys():
        if 'standard_name' in preferred_names[k].keys() and type(preferred_names[k]['standard_name'])==str:
            preferred_names[k]['standard_name'] = re.split(', | |,', preferred_names[k]['standard_name'].lower())
        if 'alternate_names' in preferred_names[k].keys() and type(preferred_names[k]['alternate_names'])==str:
            preferred_names[k]['alternate_names'] = re.split(', | |,', preferred_names[k]['alternate_names'].lower())
    var_info = pd.DataFrame({'name': list(xrDataset.variables), 'standard_name': None, 'new_name': None})
    for vi in range(len(var_info)):
        v = var_info['name'][vi]
        if 'standard_name' in xrDataset[v].attrs.keys():
            var_info['standard_name'][vi] = xrDataset[v].attrs['standard_name']
    for k in preferred_names.keys():
        mapped_vars = []
        if 'standard_name' in preferred_names[k].keys():
            overlap = var_info['name'][var_info['standard_name'].isin(preferred_names[k]['standard_name'])]
            mapped_vars=np.append(mapped_vars,overlap)
        if 'alternate_names' in preferred_names[k].keys():
            overlap = var_info['name'][var_info['name'].isin(preferred_names[k]['alternate_names'])]
            mapped_vars=np.append(mapped_vars,overlap)
        mapped_vars = np.unique(mapped_vars)
        if len(mapped_vars)>1:
            print(f'Found more than one possible variable to map to preferred variable {k}')
        elif len(mapped_vars)==0:
            continue
        elif k==mapped_vars[0]:
            continue
        else:
            var_info['new_name'][var_info['name']==mapped_vars[0]] = k
    var_info = var_info[['name', 'new_name']].dropna(ignore_index=True)
    for vi in range(len(var_info)):
        xrDataset = xrDataset.rename({var_info['name'][vi]: var_info['new_name'][vi]})
    return xrDataset


def standardize_units(xrDataset,
                      temperature_units='degrees_C'):
    """
    Convert xarray dataset to preferred units
    """
    unit_list = dict()
    id_list = dict()
    unit_list['temperature'] = {'degrees_C': ['degrees_c', 'c', 'degrees_celsius', 'celsius', 'degree_c', 'degree_celsius'],
                                'degrees_F': ['degrees_f', 'f', 'degrees_fahrenheit', 'fahrenheit', 'degree_f', 'degree_fahrenheit'],
                                'degrees_K': ['degrees_k', 'k', 'degrees_kelvin', 'kelvin', 'degree_k', 'degree_kelvin']}
    id_list['temperature'] = {'names': ['sst', 'temperature'],
                              'name_includes': ['temperature'],
                              'standard_name_includes': ['temperature']}
    final_units={'temperature': temperature_units}
    
    for v in list(xrDataset.variables):
        id = None
        for k in id_list.keys():
            if 'names' in id_list[k].keys() and v in id_list[k]['names']:
                id = k
                break
            if 'name_includes' in id_list[k].keys():
                if type(id_list[k]['name_includes']) is not list:
                    id_list[k]['name_includes'] = [id_list[k]['name_includes']]
                for s in id_list[k]['name_includes']:
                    if s in v:
                        id = k
                        break
                if id:
                    break
            if 'standard_name_includes' in id_list[k].keys() and 'standard_name' in xrDataset[v].attrs.keys():
                if type(id_list[k]['standard_name_includes']) is not list:
                    id_list[k]['standard_name_includes'] = [id_list[k]['standard_name_includes']]
                for s in id_list[k]['standard_name_includes']:
                    if s in xrDataset[v].attrs['standard_name']:
                        id = k
                        break
                if id:
                    break
        if not id:
            continue
        if 'units' not in xrDataset[v].attrs.keys():
            print(f'Units not provided in dataset for variable {v}. Unable to attempt conversion.')
            continue
        if id not in unit_list.keys():
            print(f'Unit options not available for {id}, unable to attempt conversion.')
            continue
        if id not in final_units.keys():
            print(f'Preferred units not provided for {id}, skipping conversion.')
            continue
        from_units = None
        to_units = None
        for u in unit_list[id].keys():
            if final_units[id].lower() in unit_list[id][u]:
                to_units = u
            if xrDataset[v].attrs['units'].lower() in unit_list[id][u]:
                from_units = u
        if not from_units or not to_units:
            print(f'Unable to identify either origin or destination units for {v}, cannot convert')
            continue
        if from_units==to_units:
            continue
        converted_var = xrDataset[v].copy()
        if id=='temperature':
            if from_units=='degrees_K':
                converted_var.data = converted_var.data - 273.15
                converted_var.attrs['units'] = 'degrees_C'
            elif from_units=='degrees_F':
                converted_var.data = (converted_var.data - 32)*5/9
                converted_var.attrs['units'] = 'degrees_C'
            if to_units=='degrees_K':
                converted_var.data = converted_var.data + 273.15
                converted_var.attrs['units'] = to_units
            elif to_units=='degrees_F':
                converted_var.data = converted_var.data*9/5 + 32
                converted_var.attrs['units'] = to_units
        xrDataset[v] = converted_var.copy()
    return xrDataset


def get_climatology_layer(time,
                          tdslink='http://basin.ceoe.udel.edu/thredds/dodsC/aqua_clim_monthly.nc',
                          erddap_server='http://basin.ceoe.udel.edu/erddap/',
                          erddap_dataset='aqua_monthly_climatology',
                          priority=['thredds', 'erddap'],
                          variable_list=['sst'],
                          preferred_names=None):
    """
    Get climatology for provided time.
    """
    got_data = False

    priority_options = ['thredds', 'erddap']

    if type(priority)!=list:
        priority=[priority]
    if type(variable_list)!=list:
        variable_list=[variable_list]
    
    for datasettype in priority:
        if datasettype=='thredds':
            if not tdslink:
                print(f'Skipping {datasettype}, tdslink not provided.')
                continue
            try:
                climatology = xr.open_dataset(tdslink)
                climatology = standardize_var_names(climatology, preferred_names=preferred_names)
                t1 = climatology['time'].data
                for i in range(len(t1)):
                    t1[i] = pd.to_datetime(t1[i]).replace(year=pd.to_datetime(time).year)
                t0 = pd.to_datetime(np.append(pd.to_datetime(str(pd.to_datetime(time).year-1)+'-12-31T23:59'), t1[:-1]))
                climatology = climatology.sel(time=t1[np.logical_and(t0<pd.to_datetime(time),t1>=pd.to_datetime(time))])
                climatology = climatology[variable_list]
                climatology.close()
                got_data = True
            except:
                print(f'Unable to grab climatology from {tdslink}.')
        elif datasettype=='erddap':
            if not erddap_server or not erddap_dataset:
                print(f'Skipping {datasettype}, erddap info (server and/or dataset) not provided.')
                continue
            try:
                e = ERDDAP(server=erddap_server,
                        protocol='griddap')
                e.dataset_id = erddap_dataset
                e.griddap_initialize()
                e.variables = variable_list
                climatology = e.to_xarray()
                if preferred_names:
                    climatology = standardize_var_names(climatology, preferred_names=preferred_names)
                t1 = climatology['time'].data
                for i in range(len(t1)):
                    t1[i] = pd.to_datetime(t1[i]).replace(year=pd.to_datetime(time).year)
                t0 = pd.to_datetime(np.append(pd.to_datetime(str(pd.to_datetime(time).year-1)+'-12-31T23:59'), t1[:-1]))
                climatology = climatology.sel(time=t1[np.logical_and(t0<pd.to_datetime(time),t1>=pd.to_datetime(time))])
                climatology.close()
                got_data = True
            except:
                print(f'Unable to grab climatology from server {erddap_server} dataset {erddap_dataset}.')
        else:
            print(f'Skipping {datasettype}, not supported. Please choose from {" or ".join(priority_options)}')
            continue
        if got_data:
            break
    
    if got_data:
        climatology = standardize_units(climatology)
        return climatology
    else:
        print('Unable to get climatology.')
        return 1
    

def get_climatology_limits(extent, climatology,
                           varname='sst',
                           quantile_val=0.01,
                           offset_val=1.5):
    """
    Get temperature colormap bounds based on climatology SST (or other variable) and domain.
    """
    vmin = np.nan
    vmax = np.nan
    climatology_sub = climatology.copy().sel(longitude=slice(extent[0], extent[1]),
                                     latitude=slice(extent[2], extent[3]))
    vmin = np.nanquantile(climatology_sub[varname].data, quantile_val) - offset_val
    vmax = np.nanquantile(climatology_sub[varname].data, 1-quantile_val) + offset_val
    if np.isnan(vmin) or np.isnan(vmax):
        print(f'Unable to get limits for domain lon {" to ".join(extent[:2])} and lat {" to ".join(extent[2:])}')
        return 1
    else:
        return vmin,vmax


def get_satellite_layer(t0='now',
                        tdslink='http://basin.ceoe.udel.edu/thredds/dodsC/GOESNOAASST.nc',
                        erddap_server='http://basin.ceoe.udel.edu/erddap/',
                        erddap_dataset='NOAA_GOES19_SST',
                        max_tdiff=72,
                        variable_list=['sst'],
                        priority=['thredds', 'erddap'],
                        preferred_names=None):
    """
    Get satellite data at single time layer
    """
    got_data = False

    priority_options = ['thredds', 'erddap']

    if type(priority)!=list:
        priority=[priority]
    if type(variable_list)!=list:
        variable_list=[variable_list]
    
    if t0=='now':
        t0 = pd.to_datetime(datetime.now(timezone.utc)).replace(tzinfo=None)
    else:
        t0 = pd.to_datetime(t0)
    
    for datasettype in priority:
        if datasettype=='thredds':
            if not tdslink:
                print(f'Skipping {datasettype}, tdslink not provided.')
                continue
            try:
                print(f'Reading from {tdslink}')
                satdata = xr.open_dataset(tdslink)
                if preferred_names:
                    try:
                        satdata = standardize_var_names(satdata, preferred_names=preferred_names)
                    except:
                        print("Error remapping variable names.")
                        return 1
                t = pd.to_datetime(satdata['time'].data)
                satdata = satdata.sel(time=t[np.argmin(np.abs(t-t0))])
                satdata = satdata[variable_list]
                satdata.close()
                td = np.abs((pd.to_datetime(satdata['time'].data)-t0).total_seconds()/60/60)
                if td > max_tdiff:
                    print(f'{tdslink} data {td} hours off from {t0.strftime("%Y-%m-%d %H:%M")}.')
                else:
                    got_data = True
            except:
                print(f'Unable to grab data from {tdslink}.')
        elif datasettype=='erddap':
            if not erddap_server or not erddap_dataset:
                print(f'Skipping {datasettype}, erddap info (server and/or dataset) not provided.')
                continue
            try:
                print(f'reading from ERDDAP {erddap_server} dataset {erddap_dataset}')
                e = ERDDAP(server=erddap_server,
                        protocol='griddap')
                e.dataset_id = erddap_dataset
                e.griddap_initialize()
                # e.variables = variable_list
                e.constraints['time>='] = (t0-pd.Timedelta(hours=max_tdiff)).strftime('%Y-%m-%dT%H:%M%S')
                e.constraints['time<='] = (t0+pd.Timedelta(hours=max_tdiff)).strftime('%Y-%m-%dT%H:%M%S')
                satdata = e.to_xarray()
                if preferred_names:
                    try:
                        satdata = standardize_var_names(satdata, preferred_names=preferred_names)
                    except:
                        print("Error remapping variable names.")
                        return 1
                satdata = satdata[variable_list]
                t = pd.to_datetime(satdata['time'].data)
                satdata = satdata.sel(time=t[np.argmin(np.abs(t-t0))])
                satdata.close()
                td = np.abs((pd.to_datetime(satdata['time'].data)-t0).total_seconds()/60/60)
                if td > max_tdiff:
                    print(f'{os.path.join(erddap_server,erddap_dataset)} data {td} hours off from {t0.strftime("%Y-%m-%d %H:%M")}.')
                else:
                    got_data = True
            except:
                try:
                    print(f'reading from ERDDAP {erddap_server} dataset {erddap_dataset}')
                    # e = ERDDAP(server=erddap_server,
                    #         protocol='griddap')
                    # e.dataset_id = erddap_dataset
                    # e.griddap_initialize()
                    # e.variables = variable_list
                    # e.constraints['time>='] = (t0-pd.Timedelta(hours=max_tdiff)).strftime('%Y-%m-%dT%H:%M%S')
                    # e.constraints['time<='] = (t0+pd.Timedelta(hours=max_tdiff)).strftime('%Y-%m-%dT%H:%M%S')
                    satdata = xr.open_dataset(os.path.join(erddap_server, 'griddap', erddap_dataset))
                    if preferred_names:
                        try:
                            satdata = standardize_var_names(satdata, preferred_names=preferred_names)
                        except:
                            print("Error remapping variable names.")
                            return 1
                    satdata = satdata[variable_list]
                    t = pd.to_datetime(satdata['time'].data)
                    satdata = satdata.sel(time=t[np.argmin(np.abs(t-t0))])
                    satdata.close()
                    td = np.abs((pd.to_datetime(satdata['time'].data)-t0).total_seconds()/60/60)
                    if td > max_tdiff:
                        print(f'{os.path.join(erddap_server,erddap_dataset)} data {td} hours off from {t0.strftime("%Y-%m-%d %H:%M")}.')
                    else:
                        got_data = True
                except:
                    print(f'Unable to grab data from server {erddap_server} dataset {erddap_dataset}.')
        else:
            print(f'Skipping {datasettype}, not supported. Please choose from {" or ".join(priority_options)}')
            continue
        if got_data:
            break
    
    if got_data:
        satdata = standardize_units(satdata)
        return satdata
    else:
        print(f'Unable to get satellite data within {max_tdiff} hours.')
        return 1
    

def get_best_bathymetry(extent=(-100, -45, 5, 46),
                        file_list=None,
                        preferred_names=None
                        ):
    """
    Function to select bathymetry within a bounding box.
    This function pulls GEBCO 2014 bathy data from hfr.marine.rutgers.edu 
    OR from multi-source elevation/bathy data from CF compliant NetCDF file downloaded from https://www.gmrt.org/GMRTMapTool/
    Prioritizes highest resolution bathymetry file available that contains entire bounding box, or ERDDAP if none available.

    Args:
        extent (tuple, optional): Cartopy bounding box. Defaults to (-100, -45, 5, 46).
        file (str filename, optional): CF Compliant NetCDF file containing GMRT bathymetry
                                       if None (default) or if file is not found, will default to ERDDAP

    Returns:
        xarray.Dataset: xarray Dataset containing bathymetry data
    """

    file=None
    if type(file_list)==pd.DataFrame:
        bathyfiles_sub = file_list[file_list['minlon'] <= extent[0]]
        bathyfiles_sub = bathyfiles_sub[bathyfiles_sub['maxlon'] >= extent[1]]
        bathyfiles_sub = bathyfiles_sub[bathyfiles_sub['minlat'] <= extent[2]]
        bathyfiles_sub = bathyfiles_sub[bathyfiles_sub['maxlat'] >= extent[3]]
        if bathyfiles_sub.shape[0]>0:
            file = bathyfiles_sub['resolution'].idxmin()

    bathy = get_bathymetry(extent=extent, file=file)
    if preferred_names:
        bathy = standardize_var_names(bathy, preferred_names=preferred_names)
    return bathy
