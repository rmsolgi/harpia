 # -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:20:41 2023

@author: Ryan
"""


import os
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import geopandas
from rasterio.features import rasterize
import rasterio
import pkg_resources
from harpia.tropomi.no2 import converter

def sat_to_df(data_file_path):
    
    file=Dataset(data_file_path, "r" , format="NETCDF4")
    
    
    qa_value=file.groups['PRODUCT'].variables['qa_value']
    tropospheric_no2=file.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column']
    latitude=file.groups['PRODUCT'].variables['latitude']
    longitude=file.groups['PRODUCT'].variables['longitude']

    

    qa_value_array=np.array(qa_value)
    tropospheric_no2_array=np.array(tropospheric_no2)
    latitude_array=np.array(latitude)
    longitude_array=np.array(longitude)
    
    #tropospheric_no2_array=(tropospheric_no2_array-np.min(tropospheric_no2_array))/(np.max(tropospheric_no2_array)-np.min(tropospheric_no2_array))
    
    
    lat_flat=latitude_array.flatten()
    long_flat=longitude_array.flatten()
    no2_flat=tropospheric_no2_array.flatten()
    qa_value_flat=qa_value_array.flatten()
    
    
    df_dict={}
    df_dict['long']=long_flat
    df_dict['lat']=lat_flat
    df_dict['no2']=no2_flat
    df_dict['qa_value']=qa_value_flat
    df=pd.DataFrame(df_dict)
    
    file.close()
    return df
    

def merge_granules(sat_data_directory, file_name_list):
    df=pd.DataFrame(columns=['lat','long','no2','qa_value'])
    for f in file_name_list:
        data_file_path=os.path.join(sat_data_directory,f)
        temp_df=converter.sat_to_df(data_file_path)
        df=pd.concat([df,temp_df],ignore_index=True)
    
    return df


def df_to_gdf(df,gdf_crs):
    
    df['geometry']=df[["long","lat"]].values.tolist()
    #df['geometry']=df['geometry'].apply(Point)
    df['geometry']=geopandas.points_from_xy(x=df['long'], y=df['lat'])
    df['no2']=df['no2'].values.tolist()
    df['qa_value']=df['qa_value'].values.tolist()
    
    gdf=geopandas.GeoDataFrame(df,crs=gdf_crs,geometry='geometry')
    
    return gdf
 
def df_filtering(df ,qa_value_threshold):
    
    df=df[df.qa_value>qa_value_threshold]
    df=df.reset_index()
    df=df.drop_duplicates(subset=['long','lat'])
    
    return df

def clip_gdf(gdf,zone_corners_coordinates):

    cliped_gdf=geopandas.clip(gdf,zone_corners_coordinates)
    
    return cliped_gdf


def gdf_to_raster(gdf, raster_template, file_path):
    geom_value=((geom,value) for geom,value in zip(gdf.geometry, gdf['no2']))
    rasterized=rasterize(geom_value,\
                         out_shape=raster_template.shape,\
                             transform=raster_template.transform,
                             fill=-1E+300,\
                                 all_touched=True,\
                                     dtype=rasterio.float64,\
                                         merge_alg=rasterio.enums.MergeAlg('REPLACE'))

    with rasterio.open(file_path,\
                        'w', driver='GTiff',\
                            dtype=rasterio.float64,\
                                count=1,\
                                    width=raster_template.shape[1],\
                                        height=raster_template.shape[0],\
                                            transform=raster_template.transform) as dst:
         dst.write(rasterized,indexes=1)
         
    return rasterized
    


def file_list_generator(sat_data_directory, granule_end_date):
    file_name_list=[]
    for f in os.listdir(sat_data_directory):
        if f.endswith('.nc') and f[36:44]==granule_end_date:
            file_name_list.append(f)
    return file_name_list



def set_region(region):
    
    if region=='130w60w20n55n':
        return [-130,20,-60,55]
    else:
        raise "the region is not supported"

        
def convert_to_df(sat_data_directory, writing_directory, granule_end_date, write_to_file=False):
    
    file_name_list=converter.file_list_generator(sat_data_directory, granule_end_date)

    df=converter.merge_granules(sat_data_directory,file_name_list)
    
    if write_to_file:
        save_file_name=granule_end_date+'_df.pickle'
        save_path=os.path.join(writing_directory,save_file_name)
        df.to_pickle(save_path)
        
    return df


def convert_to_raster(sat_data_directory, writing_directory, resolution_list, granule_end_date, qa_value_threshold, region='130w60w20n55n'):
    #resolution options: '005' for 0.05 degree, '010' for 0.10 degree, '025' for 0.25 degree, '050' for 0.50 degree, '100' for 1.0 degree
    #region must be '130w60w20n55n'
    #granule end date format: yyyymmdd and string like '20230615'
    #writing directory: the total uncliped and unfilterd dataframe is saved (not geodataframe) and the final clipped and filtered tif (raster) files.
    #sat_data_directory: read sat data downloaded from NASA
    #qa_value_threshold: filter sat data based on qa_value (usually 0.5 or 0.75) below threshold is filtered out.
    #this saves raster to file.
    
    
    
    
    
    df=converter.convert_to_df(sat_data_directory,writing_directory, granule_end_date)
    
    
    df=converter.df_filtering(df,qa_value_threshold)
    gdf=converter.df_to_gdf(df,gdf_crs=4326)
    
    region_corener=converter.set_region(region)
    
    gdf=converter.clip_gdf(gdf,region_corener)
    gdf=gdf.reset_index()
    if not gdf.empty:
        for res in resolution_list:
            
            raster_file_name='lib/raster_templates/raster_template_'+'130w60w20n55n'+'_'+res+'.tif'
            with pkg_resources.resource_stream(__name__, raster_file_name) as stream:
                raster_template=rasterio.open(stream)
            #raster_template=converter.get_raster_template('130w60w20n55n',res)
                new_raster_name=granule_end_date+'_'+res+'_'+str(qa_value_threshold)+'_raster.tif'
                new_raster=os.path.join(writing_directory, new_raster_name)
    
                rasterized=converter.gdf_to_raster(gdf,raster_template, new_raster)
        
                raster_template.close()
            

    del df
    del gdf

def converter_run(read_directory,write_directory, granule_end_date, qa_value_threshold, df_print=False):
 
    #if df_print:
    #        converter.convert_to_df(read_directory,write_directory,granule_end_date,write_to_file=True)
        
    converter.convert_to_raster(read_directory,write_directory, ['005','010','025','050','100'], granule_end_date, qa_value_threshold)


