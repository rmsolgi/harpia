# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:27:33 2023

@author: Ryan
"""


from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import numpy as np   
import copy  
from harpia.tropomi.no2 import kriging

def make_mask(data_array, predict_indices):
    
    mask=np.ones(data_array.shape)
    nan_indices=np.where(np.isnan(data_array))
    
    mask[nan_indices]=-1
    
    #negative_indices=np.where(data_array<0)
    #mask[negative_indices]=-1
    
    mask[predict_indices]=0
    
    
    return mask
##############################################################################
##############################################################################
##############################################################################

def kriging_helper(array,mask,kriging_type, variogram_model):
    
    
    one_indices=np.where(mask==1)
    y_array=one_indices[0]
    x_array=one_indices[1]
    
    
    
    k_y=y_array.tolist()
    k_x=x_array.tolist()
    
    data=array[one_indices]
    k_data=data.tolist()
    if kriging_type=='ordinary':
        k_model=OrdinaryKriging(k_x,k_y,k_data,variogram_model)
    elif kriging_type=='universal':
        k_model=UniversalKriging(k_x,k_y,k_data,variogram_model)
    else:
        raise "only universal or ordinary kriging are supported"
    zero_indices=np.where(mask==0)

    y_predict_array=zero_indices[0]
    x_predict_array=zero_indices[1]
    
    yy=y_predict_array.astype('float64')
    xx=x_predict_array.astype('float64')
    k_p_y=yy.tolist()
    k_p_x=xx.tolist()
    
    z,ss=k_model.execute('points',k_p_x,k_p_y)
   
    return z

##############################################################################
##############################################################################
##############################################################################
  
def main_kriging(data, predict_indices,kriging_type, variogram_model): #data must have negative and nan values as they are
    
    
    mask=kriging.make_mask(data,predict_indices)
    
    #data_array=(data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))
    k_data=copy.deepcopy(data)
    
    for i in range(data.shape[0]):
        slice_mask=mask[i,:,:]
        slice_data=k_data[i,:,:]
        indices=np.where(slice_mask==0)
        
        z=kriging.kriging_helper(slice_data,slice_mask,kriging_type, variogram_model)
        slice_data[indices]=z
        k_data[i,:,:]=slice_data
        print(i)
    
    #k_data=k_data*(np.nanmax(data)-np.nanmin(data))+np.nanmin(data)
    return k_data[predict_indices]
##############################################################################
##############################################################################
##############################################################################
def run_kriging_helper(data,random_mask,row_num, column_num,row_size, column_size,kriging_type, variogram_model):
    k_data=copy.deepcopy(data)
    for i in range(0,row_num):
        for j in range(0,column_num):
            data_array=data[:,(row_size*i):(row_size*(i+1)),(column_size*j):(column_size*(j+1))]
            mask_array=random_mask[:,(row_size*i):(row_size*(i+1)),(column_size*j):(column_size*(j+1))]
            
            random_indices=np.where(mask_array==0)
            z=kriging.main_kriging(data_array.copy(),random_indices,kriging_type,variogram_model)
            
            data_array[random_indices]=z
            
            
            k_data[:,(row_size*i):(row_size*(i+1)),(column_size*j):(column_size*(j+1))]=data_array
        #print(i)
    return k_data  

    
def run_kriging_helper_custom(data,random_mask,rows, cols, kriging_type, variogram_model):
    k_data=copy.deepcopy(data)
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            data_array=data[:,rows[i]:rows[i+1],cols[j]:cols[j+1]]
            mask_array=random_mask[:,rows[i]:rows[i+1],cols[j]:cols[j+1]]
    
            random_indices=np.where(mask_array==0)
            z=kriging.main_kriging(data_array.copy(),random_indices,kriging_type,variogram_model)
            
            data_array[random_indices]=z
            k_data[:,rows[i]:rows[i+1],cols[j]:cols[j+1]]=data_array
            
    return k_data
##############################################################################
##############################################################################
##############################################################################


def run_kriging(data,random_mask,resolution, kriging_type, variogram_model, partitioning=False):
    if resolution=='050':
        k_data=kriging.run_kriging_helper(data,random_mask,1,1,70,140,kriging_type, variogram_model)
    elif resolution=='025':
        if partitioning:
            k_data=kriging.run_kriging_helper(data,random_mask,3,1,47,281,kriging_type,variogram_model)
        else:
            k_data=kriging.run_kriging_helper(data,random_mask,1,1,141,281,kriging_type,variogram_model)
    else:
        raise "resolution is not recognized"
    return k_data


##############################################################################

def run_custom_kriging(data,random_mask,resolution, rows, cols, kriging_type, variogram_model): #rows and cols determines the partitioning
    k_data=kriging.run_kriging_helper_custom(data,random_mask,rows, cols, kriging_type, variogram_model)
    return k_data
    

    
    
