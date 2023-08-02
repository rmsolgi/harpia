#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:03:42 2022

@author: Ryan Solgi
"""
import numpy as np
from tensorlearn.decomposition import candecomp_parafac as cp
from tensorlearn.operations import tensor_operations as top
from tensorlearn.operations import matrix_operations as mop

from harpia.tropomi.no2 import cp_completion


def cp_completion_als(tensor, samples, rank, iteration, cp_iteration=100, verbose=True):
    
    if tensor.shape!=samples.shape:
        raise Exception ("samples and tensor must have the same shape")
    
    tensor_shape=tensor.shape
    tensor_dim=np.ndim(tensor)
    
    projected_tensor=np.multiply(tensor,samples)
    
    
    weights,factors=cp.cp_als_rand_init(projected_tensor, rank, cp_iteration)
    
    #factors=[]
    #for i in range(tensor_dim):
    #    array=np.ones((tensor_shape[i],rank))
    #    factors.append(array)
    #weights=np.ones(tensor_dim)
    sample_indices=np.where(samples==1)
    sampe_entries=projected_tensor[sample_indices]
    log=[]
    reterived_tensor=top.cp_to_tensor(weights.copy(),factors.copy())
    sample_reterived_entries=reterived_tensor[sample_indices]
    error=sample_reterived_entries-sampe_entries
    error_ratio=np.linalg.norm(error)/np.linalg.norm(sampe_entries)
    log.append(error_ratio)
    
    if verbose:
        print('Initilization is done')
        print('Error', error_ratio)
    
    for t in range(0,iteration):

        for n in range(tensor_dim):
            
            projected_tensor_n=top.unfold(projected_tensor,n)
            
            samples_n=top.unfold(samples,n)
            
            for element in range(tensor_shape[n]):
                
                right_hand_b = projected_tensor_n[element,:]
                
                rao_product=np.ones(shape=(1,rank))
                
                for i in range(tensor_dim):
                
                    if i!=n:
                        rao_product=mop.column_wise_kronecker(rao_product,factors[i])
                        
                ones=np.ones(rank)
                A_matrix=np.multiply(np.outer(samples_n[element,:],ones),rao_product)
                
                factors[n][element,:]=np.dot(np.linalg.pinv(A_matrix),right_hand_b)
                
            weights=np.linalg.norm(factors[n],axis=0)
            
            factors[n]=factors[n]/weights
                
    
        
        reterived_tensor=top.cp_to_tensor(weights.copy(),factors.copy())
        sample_reterived_entries=reterived_tensor[sample_indices]
        error=sample_reterived_entries-sampe_entries
        error_ratio=np.linalg.norm(error)/np.linalg.norm(sampe_entries)
        log.append(error_ratio)
        if verbose:
            print('Error', error_ratio)
            
               
    
    return weights, factors, log

def run_cp_completion(data,mask,rank,iteration):
    data_array=data.copy()
    data_array[np.where(np.isnan(data_array))]=0
    weights, factors, error_log=cp_completion.cp_completion_als(data_array, mask, rank, iteration)
    reterived_array=top.cp_to_tensor(weights.copy(), factors.copy())
    return reterived_array, error_log

