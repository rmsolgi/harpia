# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:01:52 2023

@author: Ryan
"""



from datetime import datetime
from datetime import timedelta
import rasterio
import os
import numpy as np
import pkg_resources
from harpia.tropomi.no2 import utils

##############################################################################
##############################################################################
def generate_date_list(start_date,end_date):  
    
    date=datetime.strptime(start_date,'%Y%m%d')
    stop_date=datetime.strptime(end_date,'%Y%m%d')
    granule_end_date_list=[]

    while date<stop_date:
        granule_end_date=date.date().strftime('%Y%m%d')
        granule_end_date_list.append(granule_end_date)
        
        date+=timedelta(days=1)
    return granule_end_date_list
##############################################################################
##############################################################################
##############################################################################

def raster_to_array(raster_directory, start_date, end_date, resolution, qa_value):

    date_list=utils.generate_date_list(start_date,end_date)

    data=[]
    existed_list=[]
    missed_dates=[]
    for date in date_list:
        
        file_name=date+'_'+resolution+'_'+qa_value+'_'+'raster.tif'
        
        file_path=os.path.join(raster_directory,file_name)
        
        if os.path.exists(file_path):
        
            file_data=rasterio.open(file_path)
        
            array=file_data.read(1)
        
            data.append(array)
            
            existed_list.append(date)
        else:
            missed_dates.append(date)
        
    data_array=np.array(data)
    

    return data_array, existed_list, missed_dates
##############################################################################
##############################################################################
##############################################################################

def add_nans(data):
    
    data_array=data.copy()
    
    nan_indices=np.where(data_array==-1E+300) #the fill value for rasterization is replaced by nan
    data_array[nan_indices]=np.nan
    
    return data_array
##############################################################################
##############################################################################
##############################################################################
def molec_per_cm2(data):
    
    data_array=data.copy()
    real_indices=np.where(np.logical_not(np.isnan(data_array)))
    data_array[real_indices]=data_array[real_indices]*6.02214E+19
    return data_array
##############################################################################
##############################################################################
##############################################################################


def make_mask(data_array):
    
    mask=np.ones(data_array.shape)
    nan_indices=np.where(np.isnan(data_array))
    
    mask[nan_indices]=0
    
    #negative_indices=np.where(data_array<0)
    #mask[negative_indices]=0
    
    
    return mask
##############################################################################
##############################################################################
############################################################################## 

def generate_random_indices(mask,n):
    
    one_indices=np.where(mask==1)
    random_pos=np.random.choice(len(one_indices[0]),n, replace=False)
    array_one_indices=np.vstack(one_indices)
    
    
    random_init_list=[]
    for i in random_pos:
        random_init_list.append(array_one_indices[:,i])
    
    random_indices_array=np.vstack(random_init_list).T
    
    
    random_indices=[]
    for r in range(random_indices_array.shape[0]):
        random_indices.append(random_indices_array[r])
    
    
    return tuple(random_indices)
##############################################################################

##############################################################################
def read_added_missing_indices(file_name):
    file_path=os.path.join('lib/random_missing_indices', file_name)
    with pkg_resources.resource_stream(__name__, file_path) as stream:
        
        random_indices_array=np.load(stream)
        random_indices=tuple(random_indices_array.copy())
        
    return random_indices

##############################################################################
##############################################################################
##############################################################################


def generate_cloud_indices(mask, number_of_days, threshold,max_iteration=100): #threshold refers to the ration of available values already existed
    
    
    daily_ratio=[]
    for day in range(0,mask.shape[0]):
        daily_ratio.append(np.sum(mask[day,:,:])/(mask[day,:,:].size))
    daily_ratio=np.array(daily_ratio)
    above_threshold_indices=np.where(daily_ratio>=threshold)
    #potential_count=len(above_threshold_indices[0])
    pool=above_threshold_indices[0]
    selected=[]
    for i in range(number_of_days):
       counter=0
       condition=True
       while condition:
        counter+=1
        rand_index_1=np.random.choice(len(pool))
        rand_index_2=np.random.choice(len(pool))
        missing_indices=np.where(mask[pool[rand_index_1],:,:]==0)
        partial_day=mask[pool[rand_index_2],:,:][missing_indices]
        #print(np.size(partial_day))
        if rand_index_1!=rand_index_2 and np.size(partial_day)>0:
            if ((np.sum(partial_day)/np.size(partial_day))==1):
                #print(np.sum(partial_day)/np.size(partial_day))
                
                
                size=len(missing_indices[0])
                dim_0=[]
                for t in range(size):
                    dim_0.append(pool[rand_index_2])
                dim_0=np.array(dim_0)
                
                daily_cloud_inidices=np.vstack([dim_0,missing_indices[0],missing_indices[1]])
                
                
                selected.append(daily_cloud_inidices)
                pool=np.delete(pool,rand_index_2)
                condition=False

       
        
        if counter>max_iteration:
           condition=False
           print('no match found number of generated indices may be less than requested')
       
    cloud_indices=tuple(np.hstack(selected))

    
    return cloud_indices

   
##############################################################################
##############################################################################
##############################################################################

def index_of_agreement(observations,predictions): #Willmott 1981

    num=np.sum(np.square(observations-predictions))
    o_mean=np.mean(observations)
    denom=np.sum((abs(predictions-o_mean)+abs(observations-o_mean))**2)
    
    ioa=1-(num/denom)
    return ioa

def modified_index_of_agreement(observations,predictions): #Mccabe Jr (1999)
    num=np.sum(abs(observations-predictions))
    o_mean=np.mean(observations)
    denom=np.sum((abs(predictions-o_mean)+abs(observations-o_mean)))
    
    ioa=1-(num/denom)
    return ioa   

def mean_absolute_error(observations,predictions):
    return np.mean(abs(observations-predictions))


def root_mean_square_error(observations,predictions):
    return np.mean(np.square(observations-predictions))**(0.5)

def correlation_coefficient(observations,predictions):
    return np.corrcoef(observations,predictions)[0][1]

def mean_absolute_percentage_error(observations,predictions):
    return np.mean(abs((observations-predictions)/observations))



##############################################################################
##############################################################################
##############################################################################

