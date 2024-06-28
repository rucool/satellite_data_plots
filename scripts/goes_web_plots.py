# Import from standard Python libraries
import os
import warnings
import argparse
import sys
import re

# Imports from required packages
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from erddapy import ERDDAP
import xarray as xr
import yaml

from functions.functions import get_satellite_layer, get_climatology_limits, get_climatology_layer, get_bathymetry_file_info, get_best_bathymetry

# Imports from cool_maps
from cool_maps.calc import calculate_ticks, dd2dms, fmt, calculate_colorbar_ticks
from cool_maps.download import get_bathymetry
import cool_maps.plot as cplt
import cool_maps.colormaps as cmcm

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")


def main(args):
    t0=args.t0
    regionFile=args.regionFile
    standard_names_file=args.standard_names_file
    bathyDir=args.bathyDir
    imgDir=args.imgDir
    cloudTemp=args.cloudTemp

    print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: Starting raw GOES imagery.')

    try:
        pd.to_datetime(t0)
    except:
        print(f'Time provided {t0} not recognized as a valid timestamp (or "now"). Unable to generate plots.')
        return 1
    if not regionFile or not os.path.isfile(regionFile):
        print(f'File defining regions ({regionFile}) not found. Unable to generate plots.')
        return 1
    if not imgDir or not os.path.isdir(imgDir):
        print(f'Image directory {imgDir} not found. Unable to generate plots.')
        return 1
    if not os.path.isfile(standard_names_file):
        print(f'File with preferred variable names {standard_names_file} not found. Continuing but will be unable to rename variables and may cause errors downstream.')
        standard_names_file = None
    if not os.path.isdir(bathyDir):
        print(f'Directory with bathymetry files {bathyDir} not found. Will default to ERDDAP bathymetry for any plots with isobath lines.')
        bathyDir = None

    proj = dict(
        map=ccrs.Mercator(), # the projection that you want the map to be in
        data=ccrs.PlateCarree() # the projection that the data is. 
        )

    try:
        with open(regionFile, 'r') as file:
            regions = yaml.safe_load(file)
    except:
        print(f'Unable to load file defining regions ({regionFile}) not found. Unable to generate plots.')
        return 1
    try:
        with open(standard_names_file, 'r') as file:
            preferred_names = yaml.safe_load(file)
    except:
        preferred_names = None
    
    print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: Reading GOES data nearest {pd.to_datetime(t0).strftime("%Y-%m-%d %H:%M")}.')
    satData_full = get_satellite_layer(t0=t0, preferred_names=preferred_names)
    if type(satData_full)!=xr.Dataset:
        print('Could not get satellite data. Exiting.')
        return 1
    print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: Reading climatology.')
    climatology_full = get_climatology_layer(satData_full['time'].data, preferred_names=preferred_names)
    bathyFileList = get_bathymetry_file_info(bathyDir)

    satData_full['sst'].data[satData_full['sst']<cloudTemp] = np.nan

    for region in regions.keys():
        print(f'\n{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: Starting {region}.')
        imgDirRegion = os.path.join(imgDir, region)
        if not os.path.isdir(imgDirRegion):
            print(f'Region {region} not available in directory {imgDir}. Skipping region.')
            continue
        hrfname = os.path.join(imgDirRegion, 'sst', 'noaa', pd.to_datetime(satData_full.time.data).strftime("%Y"), 'img', f'{pd.to_datetime(satData_full.time.data).strftime("%y%m%d.%j.%H%M")}.n00.jpg')
        thumbfname = os.path.join(imgDirRegion, 'sst', 'noaa', pd.to_datetime(satData_full.time.data).strftime("%Y"), 'thumb', f'{pd.to_datetime(satData_full.time.data).strftime("%y%m%d.%j.%H%M")}.n00thumb.jpg')
        if os.path.isfile(hrfname):
            print(f'Image {hrfname} for {region} already exists. Skipping.')
            continue
        if not os.path.isdir(os.path.split(hrfname)[0]):
            os.makedirs(os.path.split(hrfname)[0], mode = 0o775, exist_ok=True)
        if not os.path.isdir(os.path.split(thumbfname)[0]):
            os.makedirs(os.path.split(thumbfname)[0], mode = 0o775, exist_ok=True)
        if not set(['minLon','maxLon','minLat','maxLat']).issubset(regions[region].keys()):
            print(f'Complete domain limits for region {region} are not included in {regionFile}. Unable to generate plot for {region}.')
            continue
        if regions[region]['maxLon']<=regions[region]['minLon']:
            print(f'minLon exceeds maxLon for region {region}. Check domain listed in {regionFile}. Unable to generate plot for {region}.')
            continue
        if regions[region]['maxLat']<=regions[region]['minLat']:
            print(f'minLat exceeds maxLat for region {region}. Check domain listed in {regionFile}. Unable to generate plot for {region}.')
            continue
        if 'name' not in regions[region].keys():
            regions[region]['name'] = region
        extent = [regions[region]['minLon'], regions[region]['maxLon'], regions[region]['minLat'], regions[region]['maxLat']]
        v0,v1 = get_climatology_limits(extent, climatology_full)
        if 'minT' not in regions[region].keys():
            regions[region]['minT'] = v0
        try:
            regions[region]['minT'] = float(regions[region]['minT'])
        except:
            print(f'minimum temperature for {region} not recognized as numeric. Using default from climatology.')
        if 'maxT' not in regions[region].keys():
            regions[region]['maxT'] = v1
        try:
            regions[region]['maxT'] = float(regions[region]['maxT'])
        except:
            print(f'maximum temperature for {region} not recognized as numeric. Using default from climatology.')
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: subsetting SST.')
        sst_sub = satData_full.copy().sel(longitude=slice(extent[0]-.1, extent[1]+.1),
                                        latitude=slice(extent[2]-.1, extent[3]+.1))
        if 'cloudT' in regions[region].keys():
            try:
                sst_sub['sst'].data[sst_sub['sst']<float(regions[region]['cloudT'])] = np.nan
            except:
                print(f'Unable to remove clouds <{regions[region]['cloudT']} in {region}.')
        
        if (extent[1]-extent[0])<1.5 or (extent[3]-extent[2])<1.5:
            coastres='full'
        else:
            coastres='high'
        
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: initializing figure.')
        fig, ax = plt.subplots(figsize=(11,8), #12,9
                subplot_kw=dict(projection=proj['map'])
                )
        ax.set_extent(extent, crs=proj['data'])
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: adding land.')
        cplt.add_features(ax,oceancolor='none', coast=coastres)
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: adding bathymetry.')
        if 'isobaths' in regions[region].keys():
            bathy = get_best_bathymetry(extent=extent, file_list=bathyFileList)
            isobaths = re.split(', | |,', str(regions[region]['isobaths']))
            isobaths = np.sort(-np.array(isobaths).astype(float))
            cplt.add_bathymetry(ax, bathy['longitude'].data, bathy['latitude'].data, bathy['z'].data,method='shadedcontour',levels=isobaths, zorder=6, fontsize=9)
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: adding SST.')
        pch = ax.pcolormesh(sst_sub['longitude'], sst_sub['latitude'], 
                            np.squeeze(sst_sub['sst']),
                            vmin=regions[region]['minT'], vmax=regions[region]['maxT'], 
                            cmap='jet', transform=proj['data'], zorder=5)
        
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: formatting.')
        plt.title(f'GOES Sea Surface Temperature: {pd.to_datetime(sst_sub.time.data).strftime("%b %d %Y %H%M")} GMT\n', fontsize=13)
        plt.text(np.mean(extent[:2]), extent[3]+(extent[3]-extent[2])*.02, 'Courtesy of RUCOOL and U. Delaware ORB Labs, no cloud correction applied', fontsize=9, ha='center', transform=proj['data'])

        cplt.add_ticks(ax=ax, extent=extent, fontsize=11)
        cplt.add_double_temp_colorbar(ax=ax, h=pch, vmin=regions[region]['minT'], vmax=regions[region]['maxT'], fontsize=11)
        
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: saving hi-res to {hrfname}.')
        fig.savefig(hrfname, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: saving thumbnail to {thumbfname}.')
        fig.savefig(thumbfname, bbox_inches='tight', pad_inches=0.02, dpi=20)
        plt.close()
        print(f'{pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")}: {region}: complete.')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-t', '--time',
                            dest='t0',
                            default='now',
                            type=str,
                            help='End Date in format YYYYMMDDTHH:MM, or "now" for most recent')
    
    arg_parser.add_argument('-d', '--image_dir',
                            dest='imgDir',
                            default='',
                            type=str,
                            help='Directory to add images to')
    
    arg_parser.add_argument('-b', '--bathymetry_dir',
                            dest='bathyDir',
                            default='',
                            type=str,
                            help='Directory holding bathymetry files')

    arg_parser.add_argument('-f', '--region_file',
                            dest='regionFile',
                            default=os.path.join(os.getcwd(),'files','web_regions.yml'),
                            type=str,
                            help='File containing regions to plot, and settings for each region')
    
    arg_parser.add_argument('-s', '--standard_names_file',
                            dest='standard_names_file',
                            default=os.path.join(os.getcwd(),'files','standardized_variable_names.yml'),
                            type=str,
                            help='File containing preferred names for variables and corresponding CF standard names and/or alternate name options')
    
    arg_parser.add_argument('-ct', '--cloud_low_temperature',
                            dest='cloudTemp',
                            default=-1,
                            type=float,
                            help='Minimum temperature to include (assume below this value is clouds, default -1C)')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
